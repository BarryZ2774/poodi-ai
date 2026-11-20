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
    pa, pb, pc = positions[triangles].astype(np.float64).transpose(1, 0, 2)
    face_normals = np.cross(pb-pa, pc-pa)
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-20)
    faces, corners = np.where(np.einsum("ij,ikj->ik", face_normals, normals[triangles]) < 0.05)
    if len(faces):
        positions = np.vstack((positions, positions[triangles[faces, corners]]))
        normals = np.vstack((normals, face_normals[faces]))
        triangles[faces, corners] = np.arange(len(faces)) + len(props)
    # glTF's standard quantized normal format packs a unit vector into one
    # aligned word. Positions remain ordinary float32 metre coordinates.
    packed = np.zeros((len(normals), 4), dtype=np.int8)
    packed[:, :3] = np.rint(normals * 127).astype(np.int8)
    # Hard edges need separate corners. Reuse corners that became identical
    # after encoding, including those created on either side of a boolean seam.
    keys = np.hstack((positions, packed))
    _, first, remap = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    positions, packed = positions[first], packed[first]
    if len(positions) >= 65536:
        raise ValueError("Index range exceeded")
    indices = remap[triangles].astype("<u2").ravel()
    views, data = [], bytearray()
    for array, target in [(positions, 34962), (packed, 34962), (indices, 34963)]:
        views.append(dict(buffer=0, byteOffset=len(data), byteLength=array.nbytes, target=target))
        data.extend(array.tobytes())
        data.extend(bytes(-len(data) % 4))
    views[1]["byteStride"] = 4
    linear = [((c/255+0.055)/1.055)**2.4 for c in (244, 241, 236)]
    model = {
        "extensionsUsed": ["KHR_mesh_quantization"],
        "extensionsRequired": ["KHR_mesh_quantization"],
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": FACTS["name"]}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1},
                                     "indices": 2, "material": 0}]}],
        "materials": [{"name": "body", "pbrMetallicRoughness": {
            "baseColorFactor": linear+[1], "metallicFactor": 0, "roughnessFactor": 1}}],
        "accessors": [
            dict(bufferView=0, componentType=5126, count=len(positions), type="VEC3",
                 min=positions.min(axis=0).tolist(), max=positions.max(axis=0).tolist()),
            dict(bufferView=1, componentType=5120, normalized=True, count=len(packed), type="VEC3"),
            dict(bufferView=2, componentType=5123, count=len(indices), type="SCALAR")],
        "bufferViews": views, "buffers": [{"byteLength": len(data)}]}
    metadata = json.dumps(model, separators=(",", ":"), allow_nan=False).encode()
    metadata += b" " * (-len(metadata) % 4)
    length = 28 + len(metadata) + len(data)
    glb = (struct.pack("<4sIII4s", b"glTF", 2, length, len(metadata), b"JSON") +
           metadata + struct.pack("<I4s", len(data), b"BIN\0") + data)
    if len(glb) > 400_000 or len(indices) // 3 > (25_000 if FACTS["kind"] == "bridge" else 6_000):
        raise ValueError(f"Mobile budget exceeded: {len(indices)//3} triangles, {len(glb)} bytes")
    (HERE / (FACTS["slug"] + ".glb")).write_bytes(glb)
    print(f"{len(indices)//3:,} triangles, {len(glb):,} bytes")


def build():
    plans=FACTS["plans"]
    hard,curved=[],[]
    for name in FACTS["ground_keys"]:
        hard.append(section(plans[name]["outline"]).simplify(0.95).extrude(.8))
    for i in range(8):
        outline=plans["pier"+str(i)]["outline"]
        x,y=plans["pier"+str(i)]["centre"]
        theta=math.atan2(y,x)
        hard.append(mass(outline,.6,3.1))
        for sign in (-1,1):
            u=x-sign*1.8*math.sin(theta);v=y+sign*1.8*math.cos(theta)
            hard.append(drum(u,v,[(2.9,1.55,1.55),
                                  (23.4,1.3,1.3),(24.7,2.2,2.2)],8))
        hard.append(mass(outline,24.2,27.6))
    def ring(r,inside,low,high,n=32):
        return (md.CrossSection.circle(r,n)-md.CrossSection.circle(inside,n)).extrude(high-low).translate((0,0,low))
    # Wide round arches bridge the gaps between the paired supports.
    arcade=ring(23.5,16.1,22.8,33.5)
    theta=math.atan2(*reversed(plans["pier0"]["centre"]))
    for i in range(8):
        cut=arch_cut(0,20,23,6.2,20,13)
        arcade-=cut.rotate((0,0,math.degrees(theta)-90+22.5+i*45))
    hard.extend([arcade,ring(24.384,16,31.8,33.1),ring(18.6,15.4,32.8,36.2)])
    # An open soffit follows the dome. The inner shell ends below its apex.
    radii=[(35.9,17.8),(37.5,17.5),(40.7,16.2),(44.3,13.4),
           (47.1,9.6),(48.8,5.2),(49.3776,.65)]
    outer=drum(0,0,[(z,r,r) for z,r in radii],28)
    inner=drum(0,0,[(35,16.6,16.6)]+[(z-1.1,r-1.05,r-1.05) for z,r in radii[1:-1]]+[(48.2,.65,.65)],28)
    curved.append((outer-inner,24))
    paths={"north":[(-34.4,23),(-32.8,32),(-28.8,45),(-23.2,56),(-17.2,65),
                    (-8.8,73.8),(2,79),(16,83),(13,99),(11,110)],
           "south":[(-25.5,-32),(-20.3,-42.8),(-12.5,-52.8),(-5,-60.6),
                    (4.7,-67.6),(14,-72.9),(25.8,-76),(41.3,-73.6),(44.7,-89),(47,-99)]}
    for name,locations in paths.items():
        hard.append(section(plans[name]["outline"]).simplify(1.2).extrude(2.2 if name=="north" else 3.2).translate((0,0,16.8)))
        for x,y in locations:
            hard.append(drum(x,y,[(.65,1.1,1.1),
                                  (15.7,.93,.93),(17.3,1.65,1.65)],8))
    finish(hard,curved)


if __name__ == "__main__":
    build()
