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
    # Resolve the final clipped lattice joints as planar faces.
    positions = positions[triangles].reshape(-1,3)
    normals = np.repeat(np.round(face_normals,6),3,axis=0).astype("<f4")
    triangles = np.arange(len(positions)).reshape(-1,3)
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
    names=FACTS["ground_keys"]
    feet=np.array([plans[k]["centre"] for k in names])
    top=np.array([plans["top"+k]["centre"] for k in names])
    waist=feet/np.linalg.norm(feet,axis=1)[:,None]*(18.288/math.sqrt(3))
    knee=550*.3048
    platform=762*.3048
    def locations(z):
        if z<=knee:
            return feet+(waist-feet)*z/knee
        return waist+(top-waist)*(z-knee)/(platform-knee)
    hard=[]
    for i,name in enumerate(names):
        pier_half = 14*.3048/2
        x, y = feet[i]
        hard.append(md.Manifold.cube((2*pier_half,2*pier_half,2)).translate((x-pier_half,y-pier_half,0)))
        # The main leg envelopes bend inward at the waist and flare to the aerial heads.
        radial = feet[i] / np.linalg.norm(feet[i])
        tangent = np.array([-radial[1],radial[0]])
        # S1.0 shows a triangular section with a broad clad exterior face.
        shape = np.array([radial*.61-tangent*2.275, radial*.61+tangent*2.275,
                          -radial*.12+tangent*2.275, -radial*1.2188+tangent*.48,
                          -radial*1.2188-tangent*.48, -radial*.12-tangent*2.275])
        rings=[]
        for z in (0,knee,platform):
            rings.append([(*uv,z) for uv in locations(z)[i]+shape])
        hard.append(loft(rings))
        x,y=top[i]
        tip = .93 if name == "west" else 1.025
        hard.append(drum(x,y,[(platform-1.5,1.95,1.95),(platform+3.5,1.95,1.95),
                              (platform+6.8,1.35,1.35),(257,1.35,1.35),(257,1.025,1.025),
                              (279,1.025,1.025),(279,tip,tip),(297.7896,tip,tip)],12))
    previous=2.
    for height in (180,375,550,650,755):
        centre=height*.3048
        bottom,upper=centre-1.5,centre+1.5
        if height == 755:
            upper = platform-.75
        for i in range(3):
            j=(i+1)%3
            if height in (180,375,550):
                p=locations(centre)
                hard.append(bar((*p[i],centre),(*p[j],centre),1.65,3.3))
            if height==755:
                p=locations(upper)
                direction=(p[j]-p[i])/np.linalg.norm(p[j]-p[i])
                hard.append(bar((*(p[i]-3.6*direction),upper),
                                (*(p[j]+3.6*direction),upper),1.5))
            for z in (bottom,upper):
                if height == 755 and z == upper:
                    continue  # The full outrigger already includes this chord.
                p=locations(z)
                hard.append(bar((*p[i],z),(*p[j],z),1.25))
            a,b=locations(bottom),locations(upper)
            for k in range(4):
                t=k/4; u=(k+1)/4
                low=(*((1-t)*a[i]+t*a[j]),bottom)
                high=(*((1-u)*b[i]+u*b[j]),upper)
                hard.append(bar(low,high,.9))
            # Long stay diagonals cross the otherwise open rectangular bays.
            for left,right in ((i,j),(j,i)):
                hard.append(bar((*locations(previous)[left],previous),
                                (*locations(bottom)[right],bottom),.95))
        previous=upper
    finish(hard,[])


if __name__=="__main__":
    build()
