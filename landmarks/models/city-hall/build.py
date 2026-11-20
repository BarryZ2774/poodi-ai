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


def hip(x, y, width, length, eave, ridge, ridge_length=0):
    pts = [(x+i*width/2, y+j*length/2, eave) for i in (-1,1) for j in (-1,1)]
    pts.extend([(x,y-ridge_length/2,ridge),(x,y+ridge_length/2,ridge)])
    return md.Manifold.hull_points(pts)


def arch_cut(x, bottom, spring, radius, station, depth):
    """Round-headed opening in a wall normal to the station axis."""
    arc = [(x+radius*math.cos(t), spring+radius*math.sin(t))
           for t in np.linspace(0, math.pi, 8)]
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
    plan=FACTS["plans"]["base"]["outline"]
    plan=section(plan).simplify(.48).to_polygons()[0]
    terrace=mass(plan,0,1.35)
    length=390*.3048
    width=273.25*.3048
    main=box(-width/2,width/2,-length/2,length/2,0,24.384)
    porticoes=box(-46.4,46.4,-13,13,0,24.384)
    wall_plan=main+porticoes
    wings=(mass(plan,0,24.384) ^ wall_plan)+terrace
    for side in (-1,1):
        # Deep porticoes are cut into the block above a continuous flight of steps.
        u=45*side
        wings-=box(min(u-9,u+9),max(u-9,u+9),-11,11,2,22.4)
        for y in (-54,-45,-36,-27,-19,20,28,36,44,52):
            cut=arch_cut(y,3.1,9.2,2.25,-side*42,9).rotate((0,0,90))
            wings-=cut
    # Three treads are cut into the existing entrance footprint.
    for face in (-1,1):
        for distance, height in [(46.4,1.45),(47.75,.9),(49.1,.68)]:
            ends=sorted((face*distance,face*54))
            wings-=box(*ends,-10.8,10.8,height,2.05)
        for u in (-29,-21,-13,-5,5,13,21,29):
            ends=sorted((face*57.5,face*68))
            wings-=box(u-1.9,u+1.9,*ends,12.2,17.9)
            wings-=box(u-1.65,u+1.65,*ends,4.3,9.6)
        end=face*(length/2)
        bay=box(-7.15,7.15,min(end,end+face*1.25),max(end,end+face*1.25),1.3,21.0)
        for u in (-4.4,0,4.4):
            bay-=box(u-1.05,u+1.05,min(end-1.5,end+2),max(end-1.5,end+2),9.0,17.85)
        wings+=bay
    hard=[wings]
    for side in (-1,1):
        x=side*42.5
        for y in np.linspace(-9,9,6):
            hard.append(drum(x,y,[(1.8,1.15,1.15),
                                  (20.7,1.02,1.02),(22.7,1.65,1.65)],8))
        hard.append(hip(x,0,12,32,24.1,29.5,0) ^ wall_plan.scale((1,1,2)) ^ mass(plan,0,31))
    belt=mass((np.asarray(plan)*[1.008,1.005]).tolist(),19.15,20.25) ^ wall_plan.scale((1.01,1.01,1))
    for face in (-1,1):
        ends=sorted((face*35.8,face*54))
        belt-=box(*ends,-11,11,18.5,21)
    hard.append(belt)
    for y in (-42,42):
        hard.append(hip(0,y,80,33,23.9,28,18) ^ wall_plan.scale((1,1,2)) ^ mass(plan,0,31))
    x,y=FACTS["plans"]["drum"]["centre"]
    hard.append(pedestal(x,y,[(23.9,56,54,7),(32,51,49,7),(35.3,44.5,43,6)]))
    hard.append(drum(x,y,[(34.7,21.3,21.3),(37.4,20.7,20.7),(40.3,18.75,18.75),
                         (41.5,16,16),(55.9,15.5,15.5),(55.9,18.95,18.95),(57.1,18.95,18.95),
                         (58.2168,13.1064,13.1064)],32))
    for t in np.arange(16)*math.tau/16:
        u=x+16.1*math.cos(t);v=y+16.1*math.sin(t)
        hard.append(drum(u,v,[(40.3,.96,.96),
                              (53.7,.9,.9),(55.9,1.45,1.45)],8))
    rings=[]
    for z,r,relief in [(58.2168,12.8064,.3),(61.5,12.55,.45),(65,11.75,.5),(68.7,10.05,.55),
                       (72.1,7.55,.5),(74.4,5.4,.35),(75.65,4.05,.2)]:
        rings.append([(x+(r+relief*max(0,math.cos(16*t)))*math.cos(t),
                       y+(r+relief*max(0,math.cos(16*t)))*math.sin(t),z)
                      for t in np.arange(48)*math.tau/48])
    hard.append(drum(x,y,[(75.55,4.25,4.25),(76.65,4.25,4.25),(77.5,3.42,3.42),
                          (78.35,3.42,3.42)],16))
    for t in np.arange(8)*math.tau/8:
        u=x+3.15*math.cos(t);v=y+3.15*math.sin(t)
        hard.append(drum(u,v,[(78.3,.72,.72),(83.25,.68,.68)],8))
    hard.append(drum(x,y,[(83.25,3.85,3.85),(84.15,3.85,3.85),(85.6,2.55,2.55),
                          (87.75,1.35,1.35),(89.1,1.15,1.15),(93.726,.38,.38)],16))
    finish(hard,[(loft(rings),18)])


if __name__ == "__main__":
    build()
