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
    base=FACTS["plans"]["base"]["outline"]
    x,y=FACTS["plans"]["shaft"]["centre"]
    high=64.008
    plinth=mass(base,0,9.7536)-box(-3.55,3.55,-14,-7.35,1,7.1)
    # Deep reveals follow the larger sides of the irregular ground plan.
    outline=np.asarray(base)
    for a,b in zip(outline,np.roll(outline,-1,axis=0)):
        edge=b-a
        length=np.linalg.norm(edge)
        if length<3.4 or (a[1]+b[1])/2 < -7:
            continue
        opening=box(-.85,.85,-.95,.95,3,5.8).rotate((0,0,math.degrees(math.atan2(edge[1],edge[0]))))
        plinth-=opening.translate((*(a+b)/2,0))
    crown_segments=80
    hard=[plinth,drum(x,y,[(9.65,6.45,6.45),(10.55,6.45,6.45),(11.45,6.42,6.42)],crown_segments)]
    for u in (-2.55,2.55):
        hard.append(md.Manifold.cylinder(6.45,0.78,circular_segments=16).translate((u,-9.75,0.8)))
    # Alternating channels follow the depth of the fluting to the belvedere.
    angles=np.arange(crown_segments)*math.tau/crown_segments
    rings=[]
    for z,r,relief in [(9.5,6.45,.06),(12.5,6.4,.38),(45.6,5.86,.38),(46.6,5.8,0)]:
        rings.append([(x+(r-relief*(1-math.cos(8*t))/2)*math.cos(t),
                       y+(r-relief*(1-math.cos(24*t))/2)*math.sin(t),z) for t in angles])
    shaft=loft(rings)
    # Build the round arcade sector by sector. The reveals run radially and
    # retain their full wall depth all the way round each arch.
    deck=high-32*.3048
    # The 80-station ring leaves a third solid station between each of the
    # sixteen intermediate lights and six between the larger crown arches.
    crown=[drum(x,y,[(46.6,5.8,5.8),(47.6,5.8,5.8)],crown_segments),
           drum(x,y,[(47.56,3.66,3.66),(deck,3.66,3.66)],crown_segments)]
    def window_tier(z0,z1,ri,ro,sill,spring,half_steps,repeat):
        # Indexed rings keep every reveal radial. Neighbouring bays share their
        # endpoints, including on the back of the lantern wall.
        vertices=[]
        faces=[]
        count=crown_segments
        period=count//repeat
        def offset(i):
            return (i+period/2)%period-period/2
        def idx(i,row,side):
            return (i%count)*8+row*2+side
        for i in range(count):
            t=i*math.tau/count
            h=math.sqrt(max(0,1-(offset(i)/half_steps)**2))
            head=spring+h*ro*half_steps*math.tau/count
            for z in (z0,sill,head,z1):
                for radius in (ri,ro):
                    vertices.append((x+radius*math.cos(t),y+radius*math.sin(t),z))
        def quad(a,b,c,d):
            faces.extend([(a,b,c),(a,c,d)])
        def present(i,row):
            return 0<=row<3 and (row!=1 or abs(offset(i+.5))>half_steps)
        for i in range(count):
            for row in range(3):
                if not present(i,row):
                    continue
                a,b,c,d=[idx(k,j,1) for k,j in ((i,row),(i+1,row),(i+1,row+1),(i,row+1))]
                quad(a,b,c,d)
                quad(d-1,c-1,b-1,a-1)
                if not present(i,row-1): quad(b,a,a-1,b-1)
                if not present(i,row+1): quad(d,c,c-1,d-1)
                if not present(i-1,row): quad(a-1,a,d,d-1)
                if not present(i+1,row): quad(c-1,c,b,b-1)
        mesh=md.Mesh64(np.array(vertices),np.array(faces,dtype=np.uint64))
        crown.append(md.Manifold(mesh))
    window_tier(47.6,deck+.015,3.64,5.8,48.4,51.65,2,8)
    window_tier(deck,58.52,4.12,5.65,55.45,56.70,1,16)
    window_tier(58.5,high,4.08,5,59.25,61.65,2,8)
    crown=union(crown)
    # The concrete rim is hollow. Its inner edge lies inside the checkpoint radius.
    finish(hard+[shaft],[(crown,16)])


if __name__ == "__main__":
    build()
