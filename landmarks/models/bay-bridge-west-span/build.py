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


def side_plate(shape, left, right):
    """Extrude a station/elevation section across the bridge."""
    return shape.extrude(right - left).transform(((0, 0, 1, left), (1, 0, 0, 0), (0, 1, 0, 0)))


def front_plate(outline, station, depth):
    return section(outline).extrude(depth).rotate((90, 0, 0)).translate((0, station + depth / 2, 0))


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


def pedestal(x, s, levels):
    """Clipped-corner courses. Equal elevations produce horizontal setbacks."""
    rings = []
    for z, width, depth, chamfer in levels:
        w, d, c = width / 2, depth / 2, chamfer
        corners = [(-w+c, -d), (w-c, -d), (w, -d+c), (w, d-c),
                   (w-c, d), (-w+c, d), (-w, d-c), (-w, -d+c)]
        rings.append([(x + u, s + v, z) for u, v in corners])
    return loft(rings)


def brace(x0, z0, x1, z1, width, station, depth):
    length = math.hypot(x1 - x0, z1 - z0)
    dx, dz = (z0 - z1) * width / (2 * length), (x1 - x0) * width / (2 * length)
    return front_plate([(x0+dx, z0+dz), (x0-dx, z0-dz),
                        (x1-dx, z1-dz), (x1+dx, z1+dz)], station, depth)


def map_axes():
    a = math.radians(FACTS["bearing_deg"])
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [0, 0, 1], [s, -c, 0]])


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


def tube(path, radius, sides=8):
    """Sweep a round cable along a station/elevation path at a fixed offset."""
    rings = []
    for i, (x, y, z) in enumerate(path):
        v = np.subtract(path[min(i+1, len(path)-1)], path[max(i-1, 0)])
        v = v/np.linalg.norm(v)
        across = np.cross(v, (0, 0, 1))
        if np.linalg.norm(across) < 1e-8:
            across = np.array([1., 0., 0.])
        across /= np.linalg.norm(across)
        normal = np.cross(v, across)
        angles = np.arange(sides)*math.tau/sides
        rings.append([np.array([x,y,z])+radius*(math.cos(t)*across+math.sin(t)*normal) for t in angles])
    return loft(rings)


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
    triangles = mesh.tri_verts.copy()
    # A boolean intersection can leave sub-millimetre slivers. Collapse only
    # short mesh edges; all shading corners at an endpoint move together.
    points, owners = np.unique(positions, axis=0, return_inverse=True)
    spatial = owners[triangles]
    edges = np.sort(np.vstack((spatial[:, [0,1]], spatial[:, [1,2]], spatial[:, [2,0]])), axis=1)
    short = edges[np.linalg.norm(points[edges[:,0]]-points[edges[:,1]], axis=1) < 0.0012]
    parent = np.arange(len(points))
    for a, b in short:
        while parent[a] != a:
            a = parent[a]
        while parent[b] != b:
            b = parent[b]
        parent[max(a,b)] = min(a,b)
    while np.any(parent != parent[parent]):
        parent = parent[parent]
    positions = points[parent[owners]]
    corners = parent[spatial]
    triangles = triangles[np.all(corners != np.roll(corners, 1, axis=1), axis=1)]
    normals = normals.astype("<f4")
    # Preserve seams while sharing identical position/normal pairs.
    keys = np.hstack((positions, normals))
    _, first, remap = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    positions, normals = positions[first], normals[first]
    if len(positions) >= 65536:
        raise ValueError("Index range exceeded")
    indices = remap[triangles].astype("<u2").ravel()
    views, data = [], bytearray()
    for array, target in [(positions, 34962), (normals, 34962), (indices, 34963)]:
        views.append(dict(buffer=0, byteOffset=len(data), byteLength=array.nbytes, target=target))
        data.extend(array.tobytes())
        data.extend(bytes(-len(data) % 4))
    linear = [((c/255+0.055)/1.055)**2.4 for c in (244, 241, 236)]
    model = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": FACTS["name"]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1},
                                     "indices": 2, "material": 0}]}],
        "materials": [{"name": "body", "pbrMetallicRoughness": {
            "baseColorFactor": linear+[1], "metallicFactor": 0, "roughnessFactor": 1}}],
        "accessors": [
            dict(bufferView=0, componentType=5126, count=len(positions), type="VEC3",
                 min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist()),
            dict(bufferView=1, componentType=5126, count=len(normals), type="VEC3"),
            dict(bufferView=2, componentType=5123, count=len(indices), type="SCALAR")],
        "bufferViews": views, "buffers": [{"byteLength": len(data)}]}
    metadata = json.dumps(model, separators=(",", ":"), allow_nan=False).encode()
    metadata += b" " * (-len(metadata) % 4)
    length = 28 + len(metadata) + len(data)
    glb = (struct.pack("<4sIII4s", b"glTF", 2, length, len(metadata), b"JSON") +
           metadata + struct.pack("<I4s", len(data), b"BIN\0") + data)
    if len(glb) > (1_500_000 if FACTS["kind"] == "bridge" else 400_000) or len(indices) // 3 > (25_000 if FACTS["kind"] == "bridge" else 6_000):
        raise ValueError(f"Mobile budget exceeded: {len(indices)//3} triangles, {len(glb)} bytes")
    (HERE / (FACTS["slug"] + ".glb")).write_bytes(glb)
    print(f"{len(indices)//3:,} triangles, {len(glb):,} bytes")


def build():
    w2, w3, _, w5, w6 = FACTS["tower_stations_m"]
    side = 1160 * 0.3048
    half_anchor = (abs(w3)+w5)/2 - side
    east = w6 + side
    west = east - 3140
    width, depth = 66*.3048,35*.3048
    # Coast Pilot lists underside clearances at the four towers and midspans.
    mhw=.643
    deck_at=lambda clearance: clearance*.3048+mhw+depth
    crown=260.76*.3048
    anchorage_top=286.6*.3048
    cable_entry=269.5*.3048
    mid_w,mid_e=(w2+w3)/2,(w5+w6)/2
    stations=[west,w2-side,w2,mid_w,w3,-half_anchor,half_anchor,w5,mid_e,w6,east]
    # Continue the measured outer slopes to the two shore landings.
    slope=(deck_at(204)-deck_at(174))/(mid_w-w2)
    grades=[deck_at(174)+(west-w2)*slope,deck_at(174)-side*slope,deck_at(174),
            deck_at(204),deck_at(217),crown,crown,deck_at(218),deck_at(204),
            deck_at(175),deck_at(175)-(east-w6)*(deck_at(204)-deck_at(175))/(w6-mid_e)]
    def road(s):
        return float(np.interp(s, stations, grades))
    hard, curved = [], []
    for down in (0, depth-1.5):
        slab = [(s,road(s)-down) for s in stations]
        slab += [(s,road(s)-down-1.5) for s in reversed(stations)]
        hard.append(side_plate(section(slab), -width/2, width/2))
    # A single folded web on either edge reveals both traffic levels.
    ys = np.linspace(west+1,east-1,63)
    for x in (-9.10,9.10):
        for i, (a,b) in enumerate(zip(ys,ys[1:])):
            z1, z2 = (0.9, depth-0.9) if i%2 else (depth-0.9, 0.9)
            hard.append(bar((x,a,road(a)-z1),(x,b,road(b)-z2),1.15))
    for i,s in enumerate((w2,w3,w5,w6)):
        outer = i in (0,3)
        top=(474 if outer else 519)*.3048
        base=top-(414 if outer else 458)*.3048
        hard.append(pedestal(0,s,[(-3,42,23,5),(base,37,19,4)]))
        def leg_x(z):
            return 15.1 - (z-base)/(top-base)*4.3
        for sign in (-1,1):
            hard.append(bar((sign*leg_x(base),s,base-0.3),(sign*leg_x(top),s,top-1.6),3.7,6))
        for a,b in zip([base+1,road(s)-depth,road(s)+1,top-42,top-20],
                       [road(s)-depth,road(s)+1,top-42,top-20,top-3]):
            for sign in (-1,1):
                hard.append(brace(sign*leg_x(a),a,-sign*leg_x(b),b,2.1,s,3.2))
        hard.append(box(-13,13,s-4.4,s+4.4,top-4.5,top))
    # Centre anchor rises between the two independent pairs of cables.
    hard.append(pedestal(0,0,[(-3,34,72,6),(14,34,72,6),(52,29,61,4)]))
    hard.append(pedestal(0,0,[(50,29,61,4),(crown-depth,28,2*half_anchor,2)]))
    for x in (-12.1,12.1):
        # Separate the two tall side housings around the through roadway.
        hard.append(pedestal(x,0,[(crown-depth-1,5.2,2*half_anchor,1.2),
                                  (cable_entry+1.1,5.2,2*half_anchor,1.2),
                                  (anchorage_top-1.2,5.0,2*half_anchor-8,1.1)]))
        for station in (-.60*half_anchor,-.20*half_anchor,.20*half_anchor,.60*half_anchor):
            hard.append(box(x-2.9,x+2.9,station-1.3,station+1.3,crown-depth,anchorage_top))
    hard.append(box(-width/2,width/2,-half_anchor,half_anchor,crown-1.5,crown))
    for s in (w2-side,east-14.7):
        hard.append(pedestal(0,s,[(-1.5,32,29,3),(road(s)-3.3,25,25,2)]))
    # The two-level roadway meets solid, grounded abutments on the flat map.
    for s,direction in ((west,1),(east,-1)):
        a,b=sorted((s,s+5*direction))
        hard.append(box(-width/2,width/2,a,b,0,min(road(a),road(b))-.7))
    for s in (w2-side,east-7):
        for x in (-9.4,9.4):
            hard.append(box(x-1.55,x+1.55,s-2.2,s+2.2,road(s)-4.4,road(s)+2.15))
    for s in np.linspace(west+2,w2-side,5)[1:-1]:
        for sign in (-1,1):
            hard.append(bar((sign*12.8,s,0),(sign*9.5,s,road(s)-1),2.1,3))
            hard.append(brace(sign*11,3,-sign*10,road(s)-depth,1.6,s,2.2))
    for a,b,c,d in [(w2-side,w2,w3,-half_anchor),(half_anchor,w5,w6,east)]:
        tower_z = lambda s: (474 if s in (w2,w6) else 519)*.3048-1.4
        nodes = [a,b,c,d]
        heights = [cable_entry if a>0 else road(a)+1.8,tower_z(b),tower_z(c),
                   cable_entry if d<0 else road(d)+1.8]
        pts = []
        for i,(u,v) in enumerate(zip(nodes,nodes[1:])):
            n = 14 if i==1 else 6
            sag = (min(heights[1:3])+max(heights[1:3]))/2-road((u+v)/2)-3 if i==1 else 8
            for t in np.linspace(0,1,n+1)[0 if i==0 else 1:]:
                pts.append((u+(v-u)*t,(1-t)*heights[i]+t*heights[i+1]-4*sag*t*(1-t)))
        if d == east:
            pts[-1] = (east-7, road(east)+1.8)
        for x in (-9.4,9.4):
            def cable_x(s):
                # Flare the last station into the side housing above the roadway.
                t=max(0,1-(abs(s)-half_anchor)/55)
                return x+math.copysign(2.7*t,x)
            curved.append((tube([(cable_x(s),s,z) for s,z in pts],.85,6),73))
            for u,v in zip(nodes,nodes[1:]):
                hangers=np.linspace(u,v,round((v-u)/28)+1)[1:-1]
                for s in hangers:
                    z=np.interp(s,*zip(*pts))
                    bottom=road(s)-0.4
                    if z-bottom>0.8:
                        curved.append((md.Manifold.cylinder(z-bottom,0.5,circular_segments=5)
                                       .translate((cable_x(s),s,bottom)),78))
    finish(hard, curved)


if __name__ == "__main__":
    build()
