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


def roof_profile(radius,eave,rise):
    # Curved glasshouse rafters rise steeply at the eaves then flatten.
    return [(-radius,2), (radius,2)]+[(radius*math.cos(t),eave+rise*math.sin(t))
            for t in np.linspace(0,math.pi,13)]


def build():
    p=FACTS["plans"]["base"]["outline"]
    base=mass(p,0,3.25)
    clip=mass(p,0,25)
    hard=[base]
    for sign in (-1,1):
        wing=front_plate(roof_profile(5.65,4.3,4.2),sign*18.2,25)
        hard.append(wing.translate((1.95,.3,0)) ^ clip)
        for y in (11.3,16.2,21.2,25.8):
            band=front_plate(roof_profile(5.77,4.3,4.5),sign*y+.3,.84)
            hard.append(band.translate((1.95,0,0)) ^ clip)
        end=front_plate(roof_profile(5.5,4.7,5.35),-6.45,20.5)
        hard.append(end.rotate((0,0,90)).translate((0,sign*31.6+.3,0)) ^ clip)
        hard.append(drum(9.4,sign*31.6+.3,[(9.55,1.4,1.4),(10.9,.9,.9),(12.3,.38,.38)],8))
    # The low octagonal palm house gives way to a second raised dome.
    x,y=2.0,.3
    hard.append(pedestal(x,y,[(3,17.4,17.1,4.15),(6.6,17.4,17.1,4.15),
                              (9.4,15.8,15.6,3.9),(11.2,13.4,13.4,3.65)]))
    dome=drum(x,y,[(10.8,6.9,6.9),(12.0,6.6,6.6),(14.1,5.25,5.25),
                   (15.8,4.8,4.8),(17.2,3.85,3.85),(18.45,2.35,2.35),
                   (19,1.1,1.1),(19.7,1.1,1.1),(20.7264,.37,.37)],24)
    porch=front_plate(roof_profile(3.9,4.2,3.6),-13,12).rotate((0,0,90)).translate((0,y,0))
    hard.append(porch ^ clip)
    finish(hard,[(dome,24)])


if __name__=="__main__":
    build()
