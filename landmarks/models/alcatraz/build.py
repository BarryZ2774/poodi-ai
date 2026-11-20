"""Generate the landmark in metres. Local coordinates are across, along and up."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import manifold3d as md
import numpy as np

HERE = Path(__file__).resolve().parent
FACTS = json.loads((HERE / "facts.json").read_text())

def union(parts):
    return md.Manifold.batch_boolean(parts, md.OpType.Add)


def box(x0, x1, s0, s1, bottom, top):
    return md.Manifold.cube((x1 - x0, s1 - s0, top - bottom)).translate((x0, s0, bottom))


def section(outline):
    return md.CrossSection([outline], md.FillRule.EvenOdd)


def loft(rings, cap=None):
    """Skin aligned rings, CCW about the sweep axis. Convex caps can use a fan."""
    n = len(rings[0])
    if cap is None:
        cap = [(0, i, i+1) for i in range(1, n-1)]
    faces = [(c, b, a) for a, b, c in cap]
    for k in range(len(rings) - 1):
        for i in range(n):
            a, b = k * n + i, k * n + (i + 1) % n
            faces.extend([(a, b, b + n), (a, b + n, a + n)])
    end = n * (len(rings) - 1)
    faces.extend((end+a, end+b, end+c) for a, b, c in cap)
    return md.Manifold(md.Mesh64(np.vstack(rings), np.array(faces, dtype=np.uint64)))


def map_axes():
    a = math.radians(FACTS["bearing_deg"])
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [0, 0, 1], [s, -c, 0]])


def mass(outline, low, high):
    return section(outline).extrude(high-low).translate((0, 0, low))


def drum(x, y, levels, sides=32):
    """Revolve horizontal elliptical sections (elevation, X radius, Y radius)."""
    angles = np.arange(sides) * math.tau / sides
    rings = [[(x + w*math.cos(t), y + d*math.sin(t), z) for t in angles]
             for z, w, d in levels]
    return loft(rings)


def bar(start, end, width, depth=None):
    """Rectangular member joining two points."""
    if depth is None:
        depth = width
    a, b = np.asarray(start), np.asarray(end)
    u = (b-a)/np.linalg.norm(b-a)
    side = np.cross(u, (0, 0, 1) if abs(u[2]) < 0.9 else (0, 1, 0))
    side /= np.linalg.norm(side)
    other = np.cross(u, side)
    matrix = np.column_stack((side, other, u, (a+b)/2))
    return md.Manifold.cube((width, depth, np.linalg.norm(b-a)), center=True).transform(matrix)


def hip(x, y, width, length, eave, ridge, ridge_length=0):
    pts = [(x+i*width/2, y+j*length/2, eave) for i in (-1,1) for j in (-1,1)]
    pts.extend([(x,y-ridge_length/2,ridge),(x,y+ridge_length/2,ridge)])
    return md.Manifold.hull_points(pts)


def finish(hard, curved):
    base = union(hard).as_original().simplify(0.002).calculate_normals(0, 0.5)
    round_parts = [s.calculate_normals(0, angle) for s, angle in curved]
    write_glb(union([base, *round_parts]).simplify(0.002))


def write_glb(solid):
    if solid.status() != md.Error.NoError or solid.volume() <= 0:
        raise ValueError("Solid failed its manifold check")
    mesh = solid.to_mesh64()
    basis = map_axes()
    props = mesh.vert_properties
    positions = (props[:, :3] @ basis.T).astype("<f4")
    normals = props[:, 3:6] @ basis.T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    # glTF's standard quantized normal format packs a unit vector into one
    # aligned word. Positions remain ordinary float32 metre coordinates.
    packed = np.zeros((len(normals), 4), dtype=np.int8)
    packed[:, :3] = np.rint(normals * 127).astype(np.int8)
    # Hard edges need separate corners. Reuse corners that became identical
    # after encoding, including those created on either side of a boolean seam.
    keys = np.hstack((positions, packed))
    _, first, remap = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    positions, packed = positions[first], packed[first]
    if len(positions) >= 65536:
        raise ValueError("Index range exceeded")
    indices = remap[mesh.tri_verts].astype("<u2").ravel()
    views, data = [], bytearray()
    for array, target in [(positions, 34962), (packed, 34962), (indices, 34963)]:
        views.append(dict(buffer=0, byteOffset=len(data), byteLength=array.nbytes, target=target))
        data.extend(array.tobytes())
        data.extend(bytes(-len(data) % 4))
    views[1]["byteStride"] = 4
    linear = [((c/255+0.055)/1.055)**2.4 for c in (244, 241, 236)]
    model = {
        "extensionsUsed": ["KHR_mesh_quantization"],
        "extensionsRequired": ["KHR_mesh_quantization"],
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": FACTS["name"]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1},
                                     "indices": 2, "material": 0}]}],
        "materials": [{"name": "body", "pbrMetallicRoughness": {
            "baseColorFactor": linear+[1], "metallicFactor": 0, "roughnessFactor": 1}}],
        "accessors": [
            dict(bufferView=0, componentType=5126, count=len(positions), type="VEC3",
                 min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist()),
            dict(bufferView=1, componentType=5120, normalized=True, count=len(packed), type="VEC3"),
            dict(bufferView=2, componentType=5123, count=len(indices), type="SCALAR")],
        "bufferViews": views, "buffers": [{"byteLength": len(data)}]}
    metadata = json.dumps(model, separators=(",", ":"), allow_nan=False).encode()
    metadata += b" " * (-len(metadata) % 4)
    length = 28 + len(metadata) + len(data)
    glb = (struct.pack("<4sIII4s", b"glTF", 2, length, len(metadata), b"JSON") +
           metadata + struct.pack("<I4s", len(data), b"BIN\0") + data)
    if len(glb) > 400_000 or len(indices) // 3 > (25_000 if FACTS["kind"] == "bridge" else 6_000):
        raise ValueError(f"Mobile budget exceeded: {len(indices)//3} triangles, {len(glb)} bytes")
    (HERE / (FACTS["slug"] + ".glb")).write_bytes(glb)
    print(f"{len(indices)//3:,} triangles, {len(glb):,} bytes")


def build():
    plans = FACTS["plans"]
    def points(name):
        return np.asarray(plans[name]["outline"])
    def centre(name):
        return plans[name]["centre"]
    summit = 39.624
    coast = points("coast")
    crest = np.vstack([points(k) for k in ("cellhouse","mess","administration","lighthouse")])
    # Clip a convex rock mass to the surveyed shoreline. The summit and intermediate
    # benches control the relief, with a low wharf terrace to the east.
    relief = [(x*1.08,y*1.08,-2) for x,y in coast]
    relief += [(x*1.04,y*1.04,4+1.5*math.cos(y/29)) for x,y in coast]
    relief += [(x*1.07+0.6,y*1.04,summit) for x,y in crest]
    relief += [(x+0.2,y+1.1,29.4) for x,y in points("tank")]
    relief += [(x+1.1,y+0.7,12.4) for x,y in points("new_industries")]
    island = md.Manifold.hull_points(relief) ^ mass(coast, 0, summit+2)
    # Cut the low dock and factory benches into the hillside so the long
    # waterfront elevations stand clear of the rock.
    for name, elevation, apron in [("barracks",4.8,6.5),("new_industries",12.5,2.5),
                                    ("model_industries",9.5,2.5),("powerhouse",6,2)]:
        terrace=section(points(name)).offset(apron, md.JoinType.Square)
        island -= terrace.extrude(55).translate((0,0,elevation))
    hard = [island]
    for name, roof in [("cellhouse",55.5),("mess",51.5),("administration",54),
                       ("barracks",20),("new_industries",21.5),("model_industries",20.5),
                       ("powerhouse",18)]:
        outline = points(name)
        hard.append(mass(outline,1.8,roof))
        if name in ("cellhouse","barracks","mess","new_industries"):
            x,y=centre(name)
            # A shallow projecting cornice preserves the long eave line.
            hard.append(mass((outline-[x,y])*1.018+[x,y],roof-1,roof+0.7))
    hard.append(box(-14,14,-28,28,55,58))
    hard.append(hip(0,0,29.2,59.5,57.7,60.1,48))
    for y in (-22,-7,8,23):
        hard.append(box(-4,4,y-1.8,y+1.8,59,61))
    lx,ly=centre("lighthouse")
    base=summit
    h=84*0.3048
    hard.append(drum(lx,ly,[(base-1,3,3),(base+1,3,3),(base+1,2.5,2.5),
                             (base+h-5.7,1.75,1.75),(base+h-5.2,2.6,2.6),
                             (base+h-4.3,2.6,2.6),(base+h-4.3,1.55,1.55),
                             (base+h-1.6,1.55,1.55),(base+h-1.6,2,2),
                             (base+h,0.35,0.35)],8))
    x,y=centre("tank")
    foot,top=29,29+94*0.3048
    for a,b in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        hard.append(bar((x+a*5.1,y+b*5.1,foot-1.4),(x+a*4,y+b*4,top-9),1.1))
    for s in (-1,1):
        for a in (-1,1):
            hard.append(bar((x+a*4.8,y+s*4.8,foot+2),(x-a*4.1,y+s*4.1,top-10),0.85))
            hard.append(bar((x+s*4.8,y+a*4.8,foot+2),(x+s*4.1,y-a*4.1,top-10),0.85))
    tank=drum(x,y,[(top-11,2,2),(top-9,6.6,6.6),(top-1.8,6.6,6.6),
                   (top,0.5,0.5)],24)
    finish(hard,[(tank,25)])


if __name__ == "__main__":
    build()
