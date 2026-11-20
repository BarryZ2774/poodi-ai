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


def arch_cut(x, bottom, spring, radius, station, depth):
    """Round-headed opening in a wall normal to the station axis."""
    arc = [(x+radius*math.cos(t), spring+radius*math.sin(t))
           for t in np.linspace(0, math.pi, 9)]
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
    outer=section(p["base"]["outline"])
    courtyard=section(p["court"]["outline"])
    walls=(outer-courtyard).extrude(13.716)
    # All courtyard faces are built from the same mapped ring. Place the
    # openings along the actual wall direction, clear of the corner stairwells.
    court=np.array(p["court"]["outline"])
    for a,b in zip(court,np.roll(court,-1,axis=0)):
        v=b-a
        length=np.linalg.norm(v)
        if length<8:
            continue
        angle=math.degrees(math.atan2(v[1],v[0]))
        bays=int((length-2)/4.35)
        for t in (np.arange(bays)+.5)/bays:
            centre=a+t*v
            for z in (0,4.3,8.6):
                cut=arch_cut(0,z+.65,z+2.25,1.37,0,5.8)
                walls-=cut.rotate((0,0,angle)).translate((*centre,0))
    # The exterior has narrow rectangular openings aligned by storey.
    outline=np.array(p["base"]["outline"])
    for a,b in zip(outline,np.roll(outline,-1,axis=0)):
        v=b-a
        length=np.linalg.norm(v)
        if length<18:
            continue
        angle=math.degrees(math.atan2(v[1],v[0]))
        count=int(length/6.2)
        for t in (np.arange(count)+.5)/count:
            xy=a+t*v
            for z in (1.5,5.8,10.1):
                if np.array_equal(a,outline[-1]) and abs(t-.5)*length<3 and z<6:
                    continue
                walls-=box(-1.02,1.02,-1.1,1.1,z,z+1.85).rotate((0,0,angle)).translate((*xy,0))
    edge=outline[0]-outline[-1]
    angle=math.degrees(math.atan2(edge[1],edge[0]))
    centre=(outline[0]+outline[-1])/2
    portal=arch_cut(0,.68,3.3,2.23,0,9.2).rotate((0,0,angle))
    walls-=portal.translate((*centre,0))
    parapet=(outer-outer.offset(-1.8288,md.JoinType.Miter)).extrude(1.524).translate((0,0,13.716))
    hard=[walls,parapet]
    for u,v in [(-8.1,24.5),(17.6,-27)]:
        penthouse=front_plate([(-1.9,13.65),(1.9,13.65),(1.9,15.6),
                               (0,16.75),(-1.9,15.6)],0,4.5)
        hard.append(penthouse.translate((u,v,0)))
    x,y=p["light"]["centre"]
    light=drum(x,y,[(13.3,1.72,1.72),(14.55,1.72,1.72),(14.55,1.54,1.54),
                    (18.1,1.25,1.25),(18.1,1.94,1.94),(18.85,1.94,1.94),
                    (18.85,1.38,1.38),(20.7,1.38,1.38),(21.9456,.4,.4)],9)
    finish([*hard,light],[])


if __name__=="__main__":
    build()
