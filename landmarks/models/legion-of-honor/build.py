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


def map_axes():
    a = math.radians(FACTS["bearing_deg"])
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [0, 0, 1], [s, -c, 0]])


def drum(x, y, levels, sides=32):
    """Revolve horizontal elliptical sections (elevation, X radius, Y radius)."""
    angles = np.arange(sides) * math.tau / sides
    rings = [[(x + w*math.cos(t), y + d*math.sin(t), z) for t in angles]
             for z, w, d in levels]
    return loft(rings)


def hip(x, y, width, length, eave, ridge, ridge_length=0):
    pts = [(x+i*width/2, y+j*length/2, eave) for i in (-1,1) for j in (-1,1)]
    pts.extend([(x,y-ridge_length/2,ridge),(x,y+ridge_length/2,ridge)])
    return md.Manifold.hull_points(pts)


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
    plans=FACTS["plans"]
    def sheet(key):
        return section(plans[key]["outline"])
    ring=sheet("base")-sheet("court")
    hard=[ring.extrude(.8)]
    # Exterior galleries stand behind open colonnades on three sides.
    # Use one footprint for the masonry galleries; mapped part boundaries
    # differ by centimetres at the party walls.
    def rect(x0,x1,y0,y1):
        return section([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])
    gallery_plan=ring ^ (rect(-40,40,-55,1.55)+rect(-34,-18.15,0,56)+rect(18.15,34,0,56)+rect(-6.3,6.3,0,5.25))
    galleries=gallery_plan.extrude(12.3)
    for sign in (-1,1):
        for y in np.linspace(-29,42,10):
            cut=arch_cut(y,3.1,6.7,1.65,-sign*29.8,3.7)
            galleries-=cut.rotate((0,0,90))
        axis=sign*24
        hard.append(hip(axis,6.4,11.4,86.3,12.3,16.3,76))
        # A low hipped cover carries raised runs of gallery skylights.
        for centre,length in [(-21,16.5),(.5,19.0),(23,17),(39.1,10.0)]:
            hard.append(hip(axis,centre,4.3,length,13.25,17.0,length-2.7))
    hard.append(galleries)
    hard.append(hip(0,-15.7,12.2,38.4,12.3,17.35,32))
    cx,cy=plans["rotunda"]["centre"]
    # The shallow cupola tapers continuously above its cylindrical drum.
    rotunda=drum(cx,cy,[(12.3,6.65,6.65),(17.65,6.65,6.65),
                        (18.95,6.20,6.20),(20.2,5.15,5.15),(21.0,3.7,3.7),
                        (21.4,1.5,1.5),(22.05,.72,.72)],24)
    for k,sign in (("northWalk",-1),("southWalk",1)):
        hard.append(sheet(k).extrude(1.56).translate((0,0,6.64)))
        for y in np.linspace(5.8,46.9,11):
            x=sign*16.38
            hard.append(drum(x,y,[(.72,.95,.95),(1.7,.95,.95),(1.7,.69,.69),
                                  (5.85,.67,.67),(6.64,.95,.95)],12))
        for x in (8.1,12.2):
            hard.append(drum(sign*x,47.15,[(.72,.94,.94),(6.64,.82,.82)],12))
    for extent,level in [(12.35,.80),(11.28,1.51)]:
        hard.append(box(-6.75,6.75,3.65,extent,0,level))
    for x in np.linspace(-5.14,5.14,6):
        hard.append(drum(x,9.25,[(.7,.71,.71),(9.4,.63,.63),(9.92,.80,.80)],10))
    # A level cornice caps the six-column portico below the gallery gable.
    hard.append(box(-6.38,6.30,5.05,10.85,9.92,13.0))
    gate=box(-5.46,5.23,44.88,49.65,0,10.84)
    gate-=arch_cut(-.11,-.2,6.52,2.36,47.15,7)
    arch=arch_cut(-.11,0,6.52,3.25,49.48,1.64)-arch_cut(-.11,-.2,6.52,2.36,49.48,2)
    gate+=arch
    gate+=box(-5.52,5.29,44.8,50.0,9.4,10.15)
    for u in (-4.45,4.22):
        gate+=box(u-.55,u+.55,49.25,50.35,0,9.95)
    hard.extend([gate,box(-5.8,5.56,44.58,49.88,10.84,11.6)])
    finish(hard,[(rotunda,24)])


if __name__=="__main__":
    build()
