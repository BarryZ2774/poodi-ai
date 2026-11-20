#!/usr/bin/env python3
"""Works out which of the map's own buildings and bridge roads lie under each landmark model.

    python3 -I tools/landmark_overlap.py WorldTest/Resources

Reads Resources/landmarks/index.json and each <slug>/<slug>.glb (as data: positions only), and the
building tiles (sf-buildings.pmtiles) at zoom 14 through `tippecanoe-decode`. Writes
Resources/landmarks/overlap.json:

    { "<slug>": { "buildings": [tile ids], "parts": [tile ids], "bridgeOutline": [[lon, lat], ...] | null,
                  "unsupported": "<reason>" (only when the file cannot be read) } }

A map building (or building part) is listed when the model covers at least half of its footprint
and some of the model over it comes down below the building's top (plus 3 m). So the map's own
stacked boxes of Coit Tower or a bridge tower are listed, while a low building under a high bridge
deck, or one that merely stands next to a model, is not. Tile ids are the same at every zoom, so
the app can leave these out with one filter that holds while tiles reload.

For bridges, `bridgeOutline` is the model's outline seen from above (convex hull, widened by 20 m),
stretched to take in the basemap's bridge-flagged roads that cross the model (up to 200 m past it).
The app hides the basemap's road lines that are flagged as bridges and lie wholly inside it.

For buildings, `viewBearing` is the direction for the Landmarks menu's three-quarter view in which the
fewest tall map buildings stand between the camera and the model.
"""
import json
import math
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

CELL = 2.0          # metres per grid cell of the model's footprint
MARGIN = 3.0        # metres: the model must come down below a building's top + this
COVER = 0.5         # share of a building's footprint the model must cover
WIDEN = 20.0        # metres added around a bridge outline
VIEW_RADIUS = 400.0 # metres around a model searched for buildings that would block the view
REACH = 200.0       # metres beyond a bridge model that its bridge roads may reach and still be hidden
EARTH = 40_075_016.685_578_49


def read_glb(path):
    """Triangles of a glTF binary in model space (metres, Y up, +X east, -Z north)."""
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError("not a glTF 2.0 binary")
    offset, doc, binary = 12, None, b""
    while offset + 8 <= len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8: offset + 8 + length]
        if kind == 0x4E4F534A:
            doc = json.loads(chunk)
        elif kind == 0x004E4942:
            binary = chunk
        offset += 8 + length
    if doc is None:
        raise ValueError("no JSON chunk")
    if doc.get("extensionsRequired"):
        raise ValueError("needs " + ", ".join(doc["extensionsRequired"]))
    accessors, views = doc.get("accessors", []), doc.get("bufferViews", [])
    sizes = {5120: ("b", 1), 5121: ("B", 1), 5122: ("h", 2), 5123: ("H", 2), 5125: ("I", 4), 5126: ("f", 4)}
    widths = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}

    def read(index):
        a = accessors[index]
        fmt, size = sizes[a.get("componentType", 5126)]
        width, count = widths[a.get("type", "SCALAR")], a.get("count", 0)
        view = views[a["bufferView"]]
        start = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = view.get("byteStride", width * size)
        return [struct.unpack_from("<" + fmt * width, binary, start + i * stride) for i in range(count)]

    def matmul(a, b):
        return [[sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)] for r in range(4)]

    def local(node):
        if "matrix" in node:
            m = node["matrix"]
            return [[m[c * 4 + r] for c in range(4)] for r in range(4)]
        x, y, z, w = node.get("rotation", [0, 0, 0, 1])
        sx, sy, sz = node.get("scale", [1, 1, 1])
        tx, ty, tz = node.get("translation", [0, 0, 0])
        r = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
        return [[r[0][0] * sx, r[0][1] * sy, r[0][2] * sz, tx],
                [r[1][0] * sx, r[1][1] * sy, r[1][2] * sz, ty],
                [r[2][0] * sx, r[2][1] * sy, r[2][2] * sz, tz], [0, 0, 0, 1]]

    triangles = []
    nodes, meshes = doc.get("nodes", []), doc.get("meshes", [])

    def visit(i, parent, seen):
        if i in seen:
            return
        seen.add(i)
        node = nodes[i]
        world = matmul(parent, local(node))
        if "mesh" in node:
            for prim in meshes[node["mesh"]].get("primitives", []):
                if prim.get("mode", 4) != 4 or "POSITION" not in prim.get("attributes", {}):
                    continue
                if prim.get("extensions"):
                    raise ValueError("needs " + ", ".join(prim["extensions"]))
                pts = [tuple(sum(world[r][k] * v for k, v in enumerate((*p, 1.0))) for r in range(3))
                       for p in read(prim["attributes"]["POSITION"])]
                idx = [int(v[0]) for v in read(prim["indices"])] if "indices" in prim else list(range(len(pts)))
                for t in range(0, len(idx) - 2, 3):
                    triangles.append((pts[idx[t]], pts[idx[t + 1]], pts[idx[t + 2]]))
        for child in node.get("children", []):
            visit(child, world, seen)

    identity = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
    scenes = doc.get("scenes", [])
    roots = scenes[doc.get("scene", 0)]["nodes"] if scenes else range(len(nodes))
    for root in roots:
        visit(root, identity, set())
    return triangles


def footprint(triangles):
    """Grid cell (ix, iz) -> lowest model height over it, from every triangle seen from above."""
    low = {}

    def mark(ix, iz, y):
        key = (ix, iz)
        if y < low.get(key, math.inf):
            low[key] = y

    for a, b, c in triangles:
        y = min(a[1], b[1], c[1])
        area = (b[0] - a[0]) * (c[2] - a[2]) - (c[0] - a[0]) * (b[2] - a[2])
        # Every edge, so walls (which are lines from above) count too.
        for p, q in ((a, b), (b, c), (c, a)):
            steps = max(1, int(math.hypot(q[0] - p[0], q[2] - p[2]) / (CELL / 2)))
            for s in range(steps + 1):
                t = s / steps
                mark(math.floor((p[0] + (q[0] - p[0]) * t) / CELL), math.floor((p[2] + (q[2] - p[2]) * t) / CELL), y)
        if abs(area) < 1e-6:
            continue
        x0, x1 = math.floor(min(a[0], b[0], c[0]) / CELL), math.floor(max(a[0], b[0], c[0]) / CELL)
        z0, z1 = math.floor(min(a[2], b[2], c[2]) / CELL), math.floor(max(a[2], b[2], c[2]) / CELL)
        for ix in range(x0, x1 + 1):
            px = (ix + 0.5) * CELL
            for iz in range(z0, z1 + 1):
                pz = (iz + 0.5) * CELL
                w0 = (b[0] - px) * (c[2] - pz) - (c[0] - px) * (b[2] - pz)
                w1 = (c[0] - px) * (a[2] - pz) - (a[0] - px) * (c[2] - pz)
                w2 = (a[0] - px) * (b[2] - pz) - (b[0] - px) * (a[2] - pz)
                if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                    mark(ix, iz, y)
    return low


class Frame:
    """Model metres <-> lon/lat around the anchor, the same way the app places models (Web Mercator)."""

    def __init__(self, lat, lon):
        self.lat, self.lon = lat, lon
        self.x0 = (lon + 180) / 360
        self.y0 = (1 - math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) / math.pi) / 2
        self.k = 1 / (EARTH * math.cos(math.radians(lat)))   # world units (0..1) per metre

    def to_lonlat(self, x, z):
        mx, my = self.x0 + x * self.k, self.y0 + z * self.k
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * my))))
        return mx * 360 - 180, lat

    def to_model(self, lon, lat):
        mx = (lon + 180) / 360
        my = (1 - math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)) / math.pi) / 2
        return (mx - self.x0) / self.k, (my - self.y0) / self.k


def hull(points):
    pts = sorted(set(points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def widen(points, by):
    return hull([(x + by * math.cos(a * math.pi / 8), z + by * math.sin(a * math.pi / 8))
                 for x, z in points for a in range(16)])


def distance(px, pz, ring):
    """Metres from a point to a convex outline (0 inside)."""
    if inside(px, pz, ring):
        return 0.0
    best = math.inf
    for (ax, az), (bx, bz) in zip(ring, ring[1:] + ring[:1]):
        dx, dz = bx - ax, bz - az
        t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / (dx * dx + dz * dz or 1)))
        best = min(best, math.hypot(px - ax - t * dx, pz - az - t * dz))
    return best


def tiles_for(bounds, z=14):
    (west, south), (east, north) = bounds
    n = 2 ** z

    def tile(lon, lat):
        return (int((lon + 180) / 360 * n),
                int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n))
    (xa, ya), (xb, yb) = tile(west, north), tile(east, south)
    return [(z, x, y) for x in range(xa, xb + 1) for y in range(ya, yb + 1)]


def decode(pmtiles, z, x, y, layer=None, need_id=True):
    out = subprocess.run(["tippecanoe-decode", *(["-l", layer] if layer else []), str(pmtiles), str(z), str(x), str(y)],
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        return []
    features = []
    for layer in json.loads(out.stdout).get("features", []):
        name = layer["properties"]["layer"]
        for f in layer["features"]:
            if "id" in f or not need_id:
                features.append((name, f))
    return features


def top_of(layer, props):
    """The height the style draws a building or building part to (porcelain-style.json)."""
    if layer == "building":
        if props.get("height") is not None:
            return props["height"]
        return props["num_floors"] * 3.2 if props.get("num_floors") is not None else 6
    if props.get("height") is not None:
        return props["height"]
    return (props.get("min_height") or 0) + (props.get("num_floors") or 2) * 3.2


def inside(px, pz, ring):
    hit = False
    j = len(ring) - 1
    for i in range(len(ring)):
        (xi, zi), (xj, zj) = ring[i], ring[j]
        if (zi > pz) != (zj > pz) and px < (xj - xi) * (pz - zi) / (zj - zi) + xi:
            hit = not hit
        j = i
    return hit


def view_bearing(xs, zs, buildings):
    """The camera direction (degrees, as the camera faces) with the least of the map's buildings standing
    between a three-quarter camera (35° above the horizon) and the model. Prefers 30° on a tie."""
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    radius = math.hypot(max(xs) - min(xs), max(zs) - min(zs)) / 2
    best = None
    for b in range(0, 360, 30):
        facing = (math.sin(math.radians(b)), -math.cos(math.radians(b)))      # +X east, -Z north
        score = 0.0
        for (bx, bz), top in buildings.values():
            vx, vz = bx - cx, bz - cz
            along = -(vx * facing[0] + vz * facing[1])                          # towards the camera
            lateral = abs(vx * facing[1] - vz * facing[0])
            if along <= 0 or lateral > radius + along * math.tan(math.radians(30)):
                continue
            gap = max(1.0, along - radius)
            score += max(0.0, top - gap * math.tan(math.radians(35)))
        rank = (round(score), abs(((b - 30 + 180) % 360) - 180))
        if best is None or rank < best[0]:
            best = (rank, b)
    return best[1]


def overlap(entry, folder, pmtiles, basemap):
    frame = Frame(entry["anchor"]["lat"], entry["anchor"]["lon"])
    low = footprint(read_glb(folder / f"{entry['slug']}.glb"))
    xs = [ix * CELL for ix, _ in low] + [(ix + 1) * CELL for ix, _ in low]
    zs = [iz * CELL for _, iz in low] + [(iz + 1) * CELL for _, iz in low]
    corners = [frame.to_lonlat(min(xs), max(zs)), frame.to_lonlat(max(xs), min(zs))]

    covered, total = defaultdict(int), defaultdict(int)
    tops, centres = {}, {}
    pad = VIEW_RADIUS
    around = [frame.to_lonlat(min(xs) - pad, max(zs) + pad), frame.to_lonlat(max(xs) + pad, min(zs) - pad)]
    for tile in tiles_for(around):
        for layer, f in decode(pmtiles, *tile):
            props, geom = f["properties"], f["geometry"]
            if props.get("is_underground"):
                continue
            key = (layer, f["id"])
            top = top_of(layer, props)
            tops[key] = top
            polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
            for poly in polys:
                rings = [[frame.to_model(lon, lat) for lon, lat in ring] for ring in poly]
                outer = rings[0]
                centres[key] = (sum(p[0] for p in outer) / len(outer), sum(p[1] for p in outer) / len(outer))
                x0, x1 = min(p[0] for p in outer), max(p[0] for p in outer)
                z0, z1 = min(p[1] for p in outer), max(p[1] for p in outer)
                if x1 < min(xs) or x0 > max(xs) or z1 < min(zs) or z0 > max(zs):
                    continue
                # Sample the footprint every metre (at least its centre for tiny ones).
                samples = [(x0 + (i + 0.5) * (x1 - x0) / max(1, round(x1 - x0)),
                            z0 + (j + 0.5) * (z1 - z0) / max(1, round(z1 - z0)))
                           for i in range(max(1, round(x1 - x0))) for j in range(max(1, round(z1 - z0)))]
                for px, pz in samples:
                    if not inside(px, pz, outer) or any(inside(px, pz, hole) for hole in rings[1:]):
                        continue
                    total[key] += 1
                    y = low.get((math.floor(px / CELL), math.floor(pz / CELL)))
                    if y is not None and y < top + MARGIN:
                        covered[key] += 1
    hidden = sorted(k for k in total if covered[k] >= COVER * total[k])
    result = {"buildings": [i for layer, i in hidden if layer == "building"],
              "parts": [i for layer, i in hidden if layer == "building_part"],
              "bridgeOutline": None, "viewBearing": None}
    if entry.get("kind") != "bridge":
        result["viewBearing"] = view_bearing(xs, zs, {k: (centres[k], tops[k]) for k in centres if k not in hidden})
    if entry.get("kind") == "bridge":
        cells = hull([(ix * CELL + dx, iz * CELL + dz) for ix, iz in low for dx in (0, CELL) for dz in (0, CELL)])
        widened = widen(cells, WIDEN)
        # The map's bridge roads often run a little past the model's ends (onto the abutments). The app
        # can only hide a line that lies wholly inside the outline, so the outline also takes in the
        # points of bridge-flagged roads that cross the model, as far as REACH metres beyond it.
        extra = []
        for tile in [t for z in (12, 13, 14, 15) for t in tiles_for(corners, z)]:
            for layer, f in decode(basemap, *tile, layer="roads", need_id=False):
                if not f["properties"].get("is_bridge"):
                    continue
                g = f["geometry"]
                lines = g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]
                for line in lines:
                    pts = [frame.to_model(lon, lat) for lon, lat in line]
                    if any(inside(x, z, widened) for x, z in pts):
                        extra += [(x, z) for x, z in pts if distance(x, z, widened) <= REACH]
        widened = widen(hull(cells + extra), WIDEN)
        ring = [frame.to_lonlat(x, z) for x, z in widened]
        result["bridgeOutline"] = [[round(lon, 7), round(lat, 7)] for lon, lat in ring + ring[:1]]
    return result


def main():
    resources = Path(sys.argv[1] if len(sys.argv) > 1 else "WorldTest/Resources")
    root = resources / "landmarks"
    pmtiles = resources / "sf-buildings.pmtiles"
    out = {}
    for entry in json.loads((root / "index.json").read_text()):
        slug = entry["slug"]
        try:
            out[slug] = overlap(entry, root / slug, pmtiles, resources / "sf-base.pmtiles")
            print(f"{slug}: hides {len(out[slug]['buildings'])} buildings, {len(out[slug]['parts'])} parts"
                  + (", bridge roads inside its outline" if out[slug]["bridgeOutline"] else ""))
        except (ValueError, KeyError, IndexError, struct.error) as error:
            out[slug] = {"buildings": [], "parts": [], "bridgeOutline": None, "unsupported": str(error)}
            print(f"{slug}: cannot read the model ({error}); nothing hidden")
    (root / "overlap.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {root / 'overlap.json'}")


if __name__ == "__main__":
    main()
