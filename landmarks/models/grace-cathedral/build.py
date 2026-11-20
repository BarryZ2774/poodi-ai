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
    # Trimming a round member can introduce a small corner at a junction.
    # Split that corner when its interpolated normal points through the surface.
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
    pa, pb, pc = positions[triangles].astype(np.float64).transpose(1, 0, 2)
    face_normals = np.cross(pb-pa, pc-pa)
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-20)
    faces, corners = np.where(np.einsum("ij,ikj->ik", face_normals, normals[triangles]) < 0.05)
    if len(faces):
        positions = np.vstack((positions, positions[triangles[faces, corners]]))
        normals = np.vstack((normals, face_normals[faces]))
        triangles[faces, corners] = np.arange(len(faces)) + len(props)
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


def lancet(x,bottom,spring,rise,width,y,depth):
    r=width/2
    return front_plate([(x-r,bottom),(x+r,bottom),(x+r,spring),
                        (x+.62*r,spring+.57*rise),(x,spring+rise),
                        (x-.62*r,spring+.57*rise),(x-r,spring)],y,depth)


def build():
    p=FACTS["plans"]
    plan=p["base"]["outline"]
    aisle_top = 20.8
    eaves = 29.4
    ridge = (247-117)*.3048
    body=mass(plan,0,aisle_top)
    for side in (-1,1):
        for y in np.linspace(-6,29,7):
            body-=lancet(y,3.8,10.7,2.9,3.9,-side*13.4,3.6).rotate((0,0,90))
    body-=lancet(0,2.2,7.4,4.4,6.4,44.25,4.2)
    hard=[body]
    apse_start = -41.8
    front = 40
    nave=front_plate([(-7.0104,aisle_top-.2),(7.0104,aisle_top-.2),(7.0104,eaves),
                      (.55,ridge),(-.55,ridge),(-7.0104,eaves)],(front+apse_start)/2,front-apse_start)
    # Radial roof facets converge on the end of the straight choir ridge.
    apse_plan=[(-7.0104,apse_start),(-7.0104,-43.4),(-5.1,-47.4),(-2.6,-49.68),
               (0,-50.5),(2.6,-49.68),(5.1,-47.4),(7.0104,-43.4),(7.0104,apse_start)]
    apse_walls=mass(apse_plan,aisle_top-.2,eaves)
    apse_roof=md.Manifold.hull_points([(u,v,eaves) for u,v in apse_plan]+[(u,apse_start,ridge) for u in (-.55,.55)])
    hard.append(apse_walls+apse_roof)
    for side in (-1,1):
        for pos in np.linspace(-6,29,7):
            nave-=lancet(pos,21.50,25.25,2.5,2.5,-side*7,2.1).rotate((0,0,90))
        for pos in np.linspace(-8.65,26.35,7):
            # Stepped aisle buttresses follow the divisions between windows.
            ends=sorted((side*12.8,side*14.7))
            hard.append(box(*ends,pos-.64,pos+.64,0,10.2))
            ends=sorted((side*12.8,side*14.1))
            hard.append(box(*ends,pos-.64,pos+.64,9.9,19.8))
    hard.append(nave ^ mass(plan,0,45))
    x,y=p["spire"]["centre"]
    transept=front_plate([(-6.5,20.6),(6.5,20.6),(6.5,29.4),(0,37.1),(-6.5,29.4)],0,44)
    transept=transept.rotate((0,0,90)).translate((0,y,0)) ^ mass(plan,0,42)
    for side in (-1,1):
        cut=lancet(y,21.5,25.6,3.35,6.4,-side*21.6,4.0).rotate((0,0,90))
        transept-=cut
        ends=sorted((side*19.5,side*21.35))
        hard.append(box(*ends,y-.46,y+.46,20.6,28.65))
    hard.append(transept)
    for key in ("north","south"):
        u,v=p[key]["centre"]
        tower=box(u-4.05,u+4.05,v-5.25,v+5.25,15.7,52.55)
        for dx in (-1.72,1.72):
            tower-=lancet(u+dx,34.8,46.5,3.6,2.25,v,12)
        hard.append(tower)
        for i in (-1,1):
            for j in (-1,1):
                px,py=u+i*3.38,v+j*4.53
                hard.append(pedestal(px,py,[(16,1.62,1.62,.16),(51.1,1.62,1.62,.16),
                                            (52.45,1.42,1.42,.15),(53.0352,.85,.85,.12)]))
    face=front_plate([(-5.15,0),(5.15,0),(5.15,30.8),(0,37.2),(-5.15,30.8)],41.6,7)
    face-=lancet(0,2.2,7.4,4.4,6.4,44.1,3.8)
    rose=md.Manifold.cylinder(2.2,3.81,circular_segments=24).rotate((90,0,0)).translate((0,46,22.5))
    hard.append(face-rose)
    hard.append(drum(x,y,[(28.5,2.35,2.35),(39.4,2.35,2.35),(42,1.55,1.55),
                          (52,1.5,1.5),(54,2.0,2.0),(57,1.35,1.35),(75.2856,.38,.38)],8))
    finish(hard,[])


if __name__=="__main__":
    build()
