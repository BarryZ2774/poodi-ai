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


def arch_cut(x, bottom, spring, radius, station, depth):
    """Round-headed opening in a wall normal to the station axis."""
    arc = [(x+radius*math.cos(t), spring+radius*math.sin(t))
           for t in np.linspace(0, math.pi, 13)]
    return front_plate([(x-radius,bottom),(x+radius,bottom)]+arc, station, depth)


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


def build():
    p=FACTS["plans"]
    adobe_plan=p["adobe"]["outline"]
    adobe=mass(adobe_plan,0,6.5)
    # Rotate the chapel within the basilica's coordinate system to follow
    # its independently mapped, slightly oblique, Dolores Street frontage.
    ax,ay=15.4,8.0
    chapel=front_plate([(-6.1,6.3),(6.1,6.3),(0,9.6)],0,45)
    adobe+=chapel.rotate((0,0,3.15)).translate((ax,ay,0)) ^ mass(adobe_plan,0,11)
    screen=front_plate([(-5.9,0),(5.9,0),(5.9,6.75),(3.5,8.35),(1.45,10.76),
                        (-1.45,10.76),(-3.5,8.35),(-5.9,6.75)],29.4,2.7).translate((13.9,0,0))
    adobe+=screen
    adobe-=arch_cut(13.9,.65,3.15,1.35,30.4,3.8)
    for u in (-2.35,0,2.35):
        adobe-=arch_cut(13.9+u,6.1 if u else 7.15,7.0 if u else 8.2,.68,30.3,3.4)
    hard=[adobe]
    plan=p["basilica"]["outline"]
    body=mass(plan,0,12.2)
    nave=front_plate([(-16.75,12.1),(3.6,12.1),(3.6,14.9),(-6.5,20.1),(-16.75,14.9)],-1,55)
    body+=nave ^ mass(plan,0,23)
    gable=[(-18.5,0),(5.5,0),(5.5,16.7),(2.3,16.7),(.6,18.2),(-.2,20.7),
           (-2.8,21.4),(-4.8,24.0),(-6.5,25.1),(-8.2,24),(-10.2,21.4),
           (-12.8,20.7),(-13.6,18.2),(-15.3,16.7),(-18.5,16.7)]
    body+=front_plate(gable,25.4,2.6)
    for u,r in [(-10.9,1.2),(-6.5,1.85),(-2.1,1.2)]:
        body-=arch_cut(u,.6,4.1,r,26.5,4.3)
    body-=arch_cut(-6.5,12.6,15.8,1.75,26.2,3.6)
    hard.append(body)
    for x,tall in [(-15.2,True),(2.35,False)]:
        y=23.15
        belfry=pedestal(x,y,[(12.0,6.5,6.8,.45),(19.6,6.5,6.8,.45),
                             (20.4,6.9,7.1,.5),(21.1,6.9,7.1,.5),
                             (21.1,5.6,5.85,.5),(26.4,5.6,5.85,.5)])
        for angle in (0,90):
            belfry-=arch_cut(0,21.8,24.1,1.25,0,9).rotate((0,0,angle)).translate((x,y,0))
        hard.append(belfry)
        z=26.4
        if tall:
            upper=pedestal(x,y,[(z,6,6,.8),(z+.85,6,6,.8),(z+1.5,4.45,4.45,.7),
                                (z+6,4.45,4.45,.7)])
            upper-=arch_cut(x,28.5,30.2,.83,y,7)
            hard.append(upper)
            z+=6
        hard.append(drum(x,y,[(z,2.75,2.75),(z+.8,2.75,2.75),(z+1.7,2.35,2.35),
                              (z+2.6,1.8,1.8),(z+3.6,.78,.78),(z+4.1,.62,.62)],12))
        hard.append(box(x-.35,x+.35,y-.35,y+.35,z+4.1,z+6.64))
        hard.append(box(x-1.1,x+1.1,y-.35,y+.35,z+5.45,z+6.12))
    finish(hard,[])


if __name__=="__main__":
    build()
