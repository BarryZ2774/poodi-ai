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
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True),1e-12)
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
    # Clipping a plateau on a shared edge can cancel an interpolated corner
    # normal. Give that corner the normal of its actual exported triangle.
    pa,pb,pc=positions[triangles].astype(float).transpose(1,0,2)
    fn=np.cross(pb-pa,pc-pa)
    fn/=np.maximum(np.linalg.norm(fn,axis=1,keepdims=True),1e-12)
    faces,corner=np.where(np.einsum("ij,ikj->ik",fn,normals[triangles])<.02)
    new_ids=np.arange(len(faces))+len(positions)
    positions=np.vstack((positions,positions[triangles[faces,corner]]))
    normals=np.vstack((normals,fn[faces])).astype("<f4")
    triangles[faces,corner]=new_ids
    used,idx=np.unique(triangles,return_inverse=True)
    triangles=idx.reshape(-1,3)
    positions,normals=positions[used],normals[used]
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


def island_slope():
    """Skin sparse USGS bare-earth transects, then clip to the actual coastline."""
    survey = FACTS["terrain_survey"]
    rows = survey["transects"]
    # Extend the survey surface past the shore before trimming its ragged edges.
    north_south = [dict(rows[0],v=-255), *rows, dict(rows[-1],v=300)]
    vertices = []
    for row in north_south:
        samples = row["samples"]
        extended = [(-115,samples[0][1]), *samples, (145,samples[-1][1])]
        for u,z in extended:
            level = z - survey["datum_shift_m"]
            # Close the survey triangles across the level parade-ground pad.
            if -215 < row["v"] < -95 and -8 < u < 86:
                level = 19.01
            vertices.append((u,row["v"],level))
    columns = len(north_south[0]["samples"])+2
    top_count = len(vertices)
    vertices.extend([(u,v,-1) for u,v,z in vertices])
    faces = []
    for j in range(len(north_south)-1):
        for i in range(columns-1):
            a = j*columns+i
            b, c, d = a+1, a+columns+1, a+columns
            for tri in ((a,b,c),(a,c,d)):
                faces.append(tri)
                faces.append(tuple(top_count+k for k in reversed(tri)))
    rim = [*range(columns), *range(2*columns-1,top_count,columns),
           *range(top_count-2,top_count-columns-1,-1),
           *range(top_count-2*columns,0,-columns)]
    for a,b in zip(rim,np.roll(rim,-1)):
        faces.extend([(a,a+top_count,b+top_count),(a,b+top_count,b)])
    return md.Manifold(md.Mesh64(np.array(vertices),np.array(faces,dtype=np.uint64)))


def build():
    plans = FACTS["plans"]
    def points(name):
        return np.asarray(plans[name]["outline"])
    def centre(name):
        return plans[name]["centre"]
    summit = 134*.3048
    coast = points("coast")
    crest = np.vstack([points(k) for k in ("cellhouse","mess","administration","lighthouse")])
    # Preserve the wharf, yard, factories and parade ground as separate level
    # sites. Everywhere else the surveyed rock rises directly from the water.
    shore = section(coast)
    dock = section([(59,-128),(158,-128),(158,-16),(62,-16)])
    factories = section([(-120,123),(-17,123),(-16,207),(60,200),(65,310),(-120,310)])
    plateau = md.CrossSection.hull_points(crest).offset(4.5,md.JoinType.Square)
    yard = section(points("yard"))
    yard_floor = 32.365
    terrain = island_slope() ^ shore.extrude(summit)
    for plan,level in [(factories,13),(dock,4.8),(plateau,summit),
                       (yard,yard_floor),(section(points("tank")).offset(1.3),31.2)]:
        area = plan ^ shore
        terrain -= area.extrude(75-level).translate((0,0,level))
        terrain += area.extrude(level)
    hard = [terrain]
    wall = yard - yard.offset(-1.2,md.JoinType.Square)
    hard.append(wall.extrude(4.9).translate((0,0,yard_floor)))
    cx,cy=centre("chimney")
    hard.append(drum(cx,cy,[(12.8,1.65,1.65),(13+91*.3048,.74,.74)],12))
    for name, roof in [("cellhouse",55.5),("mess",51.5),("administration",54),
                       ("barracks",20),("new_industries",21.5),("model_industries",20.5),
                       ("powerhouse",18)]:
        outline = points(name)
        if name in ("mess","administration"):
            # Neighbouring survey polygons nearly touch along the party walls.
            hard.append(section(outline).offset(.18,md.JoinType.Square).extrude(roof-1.8).translate((0,0,1.8)))
        else:
            block=mass(outline,1.8,roof)
            if name in ("cellhouse","barracks"):
                for a,b in zip(outline,np.roll(outline,-1,axis=0)):
                    v=b-a
                    length=np.linalg.norm(v)
                    if length < (45 if name=="cellhouse" else 68):
                        continue
                    if name=="barracks" and (a[0]+b[0])/2<90:
                        continue
                    angle=math.degrees(math.atan2(v[1],v[0]))
                    bays=int(length/5.4)
                    for t in (np.arange(bays)+.5)/bays:
                        for z in ((44.3,49.8) if name=="cellhouse" else (7.0,11.6,16.1)):
                            bay=box(-1.35,1.35,-1.4,1.4,z,z+2.45)
                            block-=bay.rotate((0,0,angle)).translate((*(a+v*t),0))
            hard.append(block)
        if name in ("cellhouse","barracks","mess","new_industries"):
            x,y=centre(name)
            # A shallow projecting cornice preserves the long eave line.
            hard.append(mass((outline-[x,y])*1.018+[x,y],roof-1,roof+0.7))
    # Roof monitors articulate the long factory frontage above its own terrace.
    for station in (141,164,187,209):
        band=section(points("new_industries")) ^ section([(-120,station-1),(-5,station-1),(-5,station+1),(-120,station+1)])
        x0,_,x1,_=band.bounds()
        x=(x0+x1)/2
        hard.append(box(x-3.5,x+3.5,station-4.5,station+4.5,22.1,23.1))
        hard.append(hip(x,station,7,9,23.1,24.5,7))
    hard.append(box(-14,14,-28,28,55,58))
    hard.append(hip(0,0,27,55,57.7,60.1,48))
    for y in (-22,-7,8,23):
        hard.append(box(-4,4,y-1.8,y+1.8,59,61))
    lx,ly=centre("lighthouse")
    base=summit
    h=84*0.3048
    hard.append(drum(lx,ly,[(base-1,3,3),(base+1,3,3),(base+1,2.5,2.5),
                             (base+h-5.7,1.75,1.75),(base+h-5.2,2.6,2.6),
                             (base+h-4.3,2.6,2.6),(base+h-4.3,1.55,1.55),
                             (base+h-1.6,1.55,1.55),(base+h-1.6,2,2),(base+h-.85,2,2),
                             (base+h,0.35,0.35)],8))
    x,y=centre("tank")
    foot,top=31.2,31.2+94*.3048
    for a,b in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        u,v=x+a*4.45,y+b*4.45
        hard.append(box(u-.66,u+.66,v-.66,v+.66,foot-1.4,top-8))
    for side in (-1,1):
        for a in (-1,1):
            member=bar((a*4.45,side*4.45,foot+1.5),(-a*4.45,side*4.45,top-9),1.05)
            hard.append(member.translate((x,y,0)))
            hard.append(member.rotate((0,0,90)).translate((x,y,0)))
    tank=drum(x,y,[(top-11,2,2),(top-9,6.6,6.6),(top-1.8,6.6,6.6),
                   (top,0.5,0.5)],24)
    finish(hard,[(tank,25)])


if __name__ == "__main__":
    build()
