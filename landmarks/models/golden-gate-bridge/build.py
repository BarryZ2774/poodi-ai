"""Build a metre-scale Golden Gate Bridge. Run with Python 3.11+."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import manifold3d as md
import numpy as np

HERE = Path(__file__).resolve().parent
FACTS = json.loads((HERE / "facts.json").read_text())
TOWER = 640.0                        # stations are measured north from midspan
PYLON = TOWER + 343.0
ARCH_SPAN = 320 * 0.3048
NORTH_HOUSING = 350 * 0.3048
NORTH_END = PYLON + NORTH_HOUSING + 5 * 175 * 0.3048
SOUTH_END = NORTH_END - 2737
# District elevation drawing is in feet above MLLW. Model elevations use MSL.
MLLW_TO_MSL = FACTS["water_datum"]["msl_above_mllw_m"]
MID_ROAD = 266.70 * 0.3048 - MLLW_TO_MSL
TOWER_ROAD = 246.08 * 0.3048 - MLLW_TO_MSL
TRUSS_DEPTH = 25 * 0.3048
TRUSS_DROP = 1.5   # deck crown to the upper chord
CABLE_X = 13.72
PANEL = 50 * 0.3048
S2 = -1094.6


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


def tower(s):
    parts = [pedestal(0, s, [(-3, 48.7, 24.4, 4), (13.41, 40.84, 19.81, 3)])]
    if s < 0:
        # The concrete fender stands clear of the south pier at the surface.
        outer = md.CrossSection.circle(1, 32).scale((45.72, 23.62))
        inner = md.CrossSection.circle(1, 32).scale((42.7, 20.6))
        parts.append((outer - inner).extrude(7.57).translate((0, s, -3)))
    levels = [(13.3, 10.0584, 16.4592, 1.5), (74, 10.0584, 16.4592, 1.5),
              (74, 8.7, 13.8, 1.3), (133, 8.7, 13.8, 1.3),
              (133, 7.4, 11.6, 1.2), (173, 7.4, 11.6, 1.2),
              (173, 6.3, 9.4, 1), (201, 6.3, 9.4, 1),
              (201, 5.6, 7.4, 0.9), (223.3, 5.6, 7.4, 0.9)]
    for sign in (-1,1):
        # A common inner face preserves vertical jambs. Only the outside steps.
        inner=CABLE_X-2.8
        for start,stop in zip(levels,levels[1:]):
            z,w,d,ch=start
            if stop[0]==z:
                continue
            x=sign*(inner+w/2)
            leg=pedestal(x,s,[start,stop])
            for face in (-1,1):
                v=s+face*d/2
                leg-=box(x-.48,x+.48,v-.78,v+.78,z+.9,stop[0]-.9)
            parts.append(leg)
        # Raised cheeks contain the cable. The bearing follows its crest.
        x=sign*CABLE_X
        parts.append(pedestal(x,s,[(223.3,4.9,6.9,.7),(225.32,4.9,3.6,.7)]))
        for offset in (-1.68,1.68):
            parts.append(box(x+offset-.66,x+offset+.66,s-1.25,s+1.25,224.8,227))
    for bottom, top in [(15, 40), (40, 66)]:
        for sign in (-1, 1):
            parts.append(brace(sign*12, bottom, -sign*12, top, 2.7, s, 5.2))
    for top, height, depth in [(134, 9, 8.2), (174, 8, 7), (202, 7, 6), (223, 6, 5)]:
        # The undersides of the transverse struts have two corbel steps.
        half = [(0, top-height), (7, top-height), (7, top-height-1.5),
                (9.5, top-height-1.5), (9.5, top-height-3.5), (13.9, top-height-3.5)]
        soffit = [(-x, z) for x, z in reversed(half[1:])] + half
        strut=front_plate(soffit + [(13.9, top), (-13.9, top)], s, depth)
        for u in (-5.5,5.5):
            for v in (s-depth/2,s+depth/2):
                strut-=box(u-.55,u+.55,v-.78,v+.78,top-height,top)
        parts.append(strut)
        # The proud middle face catches light and marks the tower centre.
        parts.append(pedestal(0, s, [(top-height, 6.8, depth+1.5, 1.2),
                                     (top, 6.8, depth+1.5, 1.2)]))
    return parts


def road(s):
    a = abs(s)
    if a <= PYLON:
        return MID_ROAD - (MID_ROAD - TOWER_ROAD) * (a / TOWER)**2
    return road(PYLON) - 0.03 * (a - PYLON)


def ribbon(stations, upper, lower):
    return section([(s, upper(s)) for s in stations] +
                   [(s, lower(s)) for s in reversed(stations)])


def deck():
    # Both tower stations and the navigational crown are exact section cuts.
    spans = [(SOUTH_END, -PYLON, 8), (-PYLON, -TOWER, 10), (-TOWER, TOWER, 32),
             (TOWER, PYLON, 10), (PYLON, NORTH_END, 8)]
    bays = np.unique(np.concatenate([np.linspace(a, b, n+1) for a, b, n in spans]))
    stations = bays[::2]
    profile = [(-13.5, -1.7), (-13.5, 0.7), (-9.5, 0.7), (-9.5, 0),
               (9.5, 0), (9.5, 0.7), (13.5, 0.7), (13.5, -1.7)]
    cap = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 7), (7, 4, 5), (7, 5, 6)]
    crown = [s for s in stations if abs(s) <= PYLON or s in (SOUTH_END, NORTH_END)]
    parts = [loft([[(x, s, road(s)+z) for x, z in profile] for s in crown], cap)]
    bottom = lambda s: road(s)-TRUSS_DROP-TRUSS_DEPTH
    top = lambda s: road(s)-TRUSS_DROP
    web = ribbon(crown, top, bottom)
    # Each 50 ft bay contains a V with a 25 ft half-run and a 25 ft rise.
    # Profile ordinates use the same piecewise grade as the slab.
    openings=[]
    for i in range(math.ceil(-PYLON/(PANEL/2)),math.floor(PYLON/(PANEL/2))-1):
        a=i*PANEL/2
        b,c=a+PANEL/2,a+PANEL
        high,low=(top,bottom) if i%2==0 else (bottom,top)
        tri=section([(a,high(a)),(c,high(c)),(b,low(b))]).offset(-(.73+.045*(i%3)),md.JoinType.Miter)
        openings.extend(tri.to_polygons())
    # The shorter approach viaducts use plain plate girders.
    web-=md.CrossSection(openings,md.FillRule.EvenOdd)
    for a,b in [(S2,-PYLON),(PYLON,PYLON+NORTH_HOUSING)]:
        web-=section([(a,0),(b,0),(b,100),(a,100)])
    for x in (-12.1, 10.8):
        parts.append(side_plate(web, x, x+1.3))
    # Transverse floor bracing gives the underside its full navigational depth.
    parts.append(box(-11.6, 11.6, -0.8, 0.8, bottom(0), bottom(0)+1.5))
    return parts


def approaches():
    parts = []
    south_outer = S2
    north_outer = PYLON + NORTH_HOUSING
    for s in (south_outer, -PYLON, PYLON, north_outer):
        z = road(s)
        parts.append(front_plate([(-16, z-4), (-16, z-14), (-9, z-14),
                                  (-7, z-10), (7, z-10), (9, z-14),
                                  (16, z-14), (16, z-4)], s, 8))
        for x in (-15, 15):
            parts.append(pedestal(x, s, [(0, 9, 12, 1.6), (z+2, 7.8, 9.5, 1.4),
                                         (z+2, 6.3, 7.5, 1), (z+6, 6.3, 7.5, 1)]))
            parts.append(box(x-2.2, x+2.2, s-2.8, s+2.8, z+6, z+9))
    for a, b, width, height, upper in [
        (south_outer-370*0.3048, south_outer, 120*0.3048, 60*0.3048, 35),
        (PYLON, north_outer, 130*0.3048, 110*0.3048, road(north_outer)-1.2)
    ]:
        if a > 0:
            # The roof is the deck. Both exterior faces follow its grade.
            housing = ribbon([a, b], lambda s: road(s)-1.5, lambda s: road(s)-1.5-height)
            parts.append(side_plate(housing, -width/2, width/2))
            parts.append(side_plate(ribbon([a,b], lambda s: road(s)-1.5-height+.2, lambda s: 0), -width/2+1,width/2-1))
        else:
            parts.append(pedestal(0,(a+b)/2,[(0,width-2,b-a-2,3),(upper-height+.2,width-2,b-a-2,3)]))
            parts.append(pedestal(0, (a+b)/2, [(upper-height, width, b-a, 3),
                                                (upper, width, b-a, 3)]))
    # Four shallow polygonal arch ribs over Fort Point, with open spandrels.
    arch_stations = np.linspace((south_outer-PYLON-ARCH_SPAN)/2, (south_outer-PYLON+ARCH_SPAN)/2, 13)
    arch_mid = (south_outer-PYLON)/2
    def arch(s):
        return 9 + 38 * (1 - ((s-arch_mid)/(ARCH_SPAN/2))**2)
    rib = ribbon(arch_stations, lambda s: arch(s)+2, lambda s: arch(s)-2)
    for x in (-11.5, -4, 4, 11.5):
        parts.append(side_plate(rib, x-0.85, x+0.85))
        for s in arch_stations[3:-1:3]:
            parts.append(box(x-0.85, x+0.85, s-0.8, s+0.8, arch(s), road(s)-1))
    for s in (south_outer, -PYLON):
        parts.append(box(-17, 17, s-8, s+8, 0, 11))
    # Ground the approach bents on the flat map.
    for a, b, count in [(SOUTH_END, south_outer, 7), (north_outer, NORTH_END, 5)]:
        for i,s in enumerate(np.linspace(a,b,count+1)[1:-1]):
            top = road(s)-2
            ground = 0
            for x in (-10.9, 10.9):
                parts.append(box(x-1.5, x+1.5, s-2.4, s+2.4, ground, top+1))
            sign = 1 if i % 2 else -1
            parts.append(brace(sign*10.9,ground+1,-sign*10.9,top-2,1.6,s,2.6))
    for s, inward in [(SOUTH_END, 1), (NORTH_END, -1)]:
        a, b = sorted([s, s+inward*5])
        parts.append(box(-13.2, 13.2, a, b, 0, road(s)-1))
    return parts


def cable_height(s):
    a = abs(s)
    if a <= TOWER:
        return 226 - (470 * 0.3048) * (1 - (a/TOWER)**2)
    t = (a-TOWER) / (PYLON-TOWER)
    return (1-t)*226 + t*(road(PYLON)+1) - 25*t*(1-t)


def cables():
    parts = []
    stations = np.arange(-64,65)*PANEL
    path = [(S2-70,27),(S2,road(S2)-4)]
    curve_stations = np.r_[np.linspace(-PYLON, -TOWER, 7)[:-1],
                          np.linspace(-TOWER, TOWER, 25)[:-1], np.linspace(TOWER, PYLON, 7)]
    path.extend((s, cable_height(s)) for s in curve_stations)
    path.append((PYLON+215*0.3048, road(PYLON)-28))
    cable_height_samples = [cable_height(s) for s in curve_stations]
    for x in (-CABLE_X, CABLE_X):
        rings = []
        for i, (s, z) in enumerate(path):
            direction = np.subtract(path[min(i+1, len(path)-1)], path[max(i-1, 0)])
            ds, dz = direction/np.linalg.norm(direction)
            # Broad soffit facet gives each vertical suspender a clean joint.
            outline=[(.82,0),(.60,.78),(-.60,.78),(-.82,0),(-.60,-.78),(.60,-.78)]
            rings.append([(x+u,s+v*dz,z-v*ds) for u,v in outline])
        parts.append(loft(rings).calculate_normals(0, 65))
        for s in stations:
            if abs(abs(s)-TOWER) < 4:
                continue
            low, high = road(s), np.interp(s, curve_stations, cable_height_samples)
            rope=box(x-.55,x+.55,s-.55,s+.55,low-.45,high)
            parts.append(rope.calculate_normals(0,1))
    return parts


def map_axes():
    south, north = FACTS["checkpoints"][:2]
    phi = math.radians(FACTS["anchor"]["lat"])
    # WGS84 tangent-plane scale at the midpoint, then rotate the actual vertices.
    e2 = 0.00669437999014
    east = (north["lon"]-south["lon"]) * math.cos(phi)
    northward = (north["lat"]-south["lat"]) * (1-e2)/(1-e2*math.sin(phi)**2)
    sin_a, cos_a = np.array([east, northward])/math.hypot(east, northward)
    return np.array([[cos_a, sin_a, 0], [0, 0, 1], [sin_a, -cos_a, 0]])


def write_glb(solid):
    if solid.status() != md.Error.NoError or solid.volume() <= 0:
        raise ValueError("Bridge solid failed its manifold check")
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
        "nodes": [{"mesh": 0, "name": "Golden Gate Bridge"}],
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
    if len(glb) > (1_500_000 if FACTS["kind"] == "bridge" else 400_000) or len(indices) // 3 > 25_000:
        raise ValueError(f"Bridge exceeds the mobile asset budget: {len(indices)//3} triangles, {len(glb)} bytes")
    (HERE / "golden-gate-bridge.glb").write_bytes(glb)
    print(f"{len(indices)//3:,} triangles, {len(glb):,} bytes")


if __name__ == "__main__":
    steel = union(deck() + approaches() + tower(-TOWER) + tower(TOWER))
    # Remove buried intersections before assigning crisp structural normals.
    steel = steel.as_original().simplify(0.012).calculate_normals(0, 1.2)
    write_glb(union([steel, *cables()]).simplify(0.006))
