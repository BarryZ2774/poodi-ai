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


def map_axes():
    a = math.radians(FACTS["bearing_deg"])
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0], [0, 0, 1], [s, -c, 0]])


def mass(outline, low, high):
    return section(outline).extrude(high-low).translate((0, 0, low))


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
    # Join the mapped party walls before extrusion, closing narrow survey seams.
    ground = md.CrossSection.batch_boolean(
        [section(FACTS["plans"][k]["outline"]).offset(.05) for k in FACTS["ground_keys"]],
        md.OpType.Add).offset(-.05)
    houses = [ground.extrude(.8)]
    for address in FACTS["ground_keys"]:
        parts=[]
        survey=FACTS["survey"][address]
        grade=survey["ground_m"]-FACTS["datum_m"]
        peak=survey["height_m"]
        ridge=min(13.15,peak-.55)
        plan = np.asarray(FACTS["plans"][address]["outline"])
        left, front = plan.min(axis=0)
        right, back = plan.max(axis=0)
        x = (left + right) / 2
        # The mapped footprint includes the front landing. Pull the first
        # storeys behind it so the bay and the steps articulate the street edge.
        w = right - left - .7
        facade = front + 1.5
        eaves = 8.65 + .12 * (int(address) % 3)
        body = front_plate([(x-w/2,.8), (x+w/2,.8), (x+w/2,eaves),
                            (x,ridge), (x-w/2,eaves)], (facade+back)/2, back-facade)
        # Recess the entry and attic window into the solid elevations.
        body -= box(x+w/2-2.25, x+w/2-.65, facade-.5, facade+1.1, 2.4, 5.3)
        body -= box(x-.66,x+.66,facade-.4,facade+.8,9.25,10.35)
        parts.append(body)
        # A small stack interrupts each roof slope near the rear of the house.
        parts.append(box(x+1.25,x+2.13,back-5.1,back-3.8,eaves-.2,peak-.75))
        parts.append(box(x+1.12,x+2.26,back-5.22,back-3.68,peak-.75,peak))
        parts.append(box(x-1.26,x+1.26,facade-.78,facade+.2,10.4,11.22))
        bay_x = x-1.05
        bay = [(bay_x-1.63,facade+.25),(bay_x-1.63,front+.45),
               (bay_x-1.30,front-.28),(bay_x+1.30,front-.28),
               (bay_x+1.63,front+.45),(bay_x+1.63,facade+.25)]
        bay_solid=mass(bay,1.8,eaves-.35)
        for z in (2.5,5.65):
            bay_solid-=box(bay_x-.59,bay_x+.59,front-.5,front+.55,z,z+1.75)
        parts.append(bay_solid)
        cap=section([(u-bay_x,v-facade) for u,v in bay])
        parts.append(cap.extrude(1.04,scale_top=(.72,.65))
                     .translate((bay_x,facade,eaves-.49)))
        for level in range(3):
            parts.append(box(x+w/2-2.4,x+w/2-.45,
                              front-.90+level*.74,facade+.2,-grade,1.05+level*.70))
        if grade>0:
            houses.append(mass(plan,0,grade+.81))
        houses.append(union(parts).as_original().translate((0,0,grade)))
    finish(houses,[])


if __name__ == "__main__":
    build()
