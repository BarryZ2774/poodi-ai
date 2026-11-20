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
    base=FACTS["plans"]["base"]["outline"]
    x,y=FACTS["plans"]["tower"]["centre"]
    hall=mass(base,0,15.5)
    # Along the Embarcadero the central pavilion projects beyond the side wings.
    for s in np.linspace(-95,95,31):
        front=-31.45 if -20.7<s<22.4 else -22.55
        cut=arch_cut(s,0.8,6.2,2.25,-front-2.4,10).rotate((0,0,90))
        hall -= cut
    hard=[hall,hip(3,0,28,201.168,15.25,20.2,194)]
    # Two shallow hipped roofs flank the centre of the raised skylit nave.
    for s in (-55,55):
        hard.append(hip(-14,s,14.5,82,15.2,17.8,73))
    hard.append(box(x-6.2,x+6.2,y-6.2,y+6.2,14.9,46))
    hard.append(box(x-6.6,x+6.6,y-6.6,y+6.6,44.5,46.6))
    clock=box(-5.9,5.9,-5.9,5.9,45.9,55)
    for k in range(4):
        dial=md.Manifold.cylinder(2,6.7056/2,circular_segments=32)
        dial=dial.rotate((90,0,0)).translate((0,-4.95,50.6))
        clock -= dial.rotate((0,0,90*k))
        for a,b in [((0,-5.28,50.6),(1.7,-5.28,52.6)),((0,-5.28,50.6),(-1.6,-5.28,51.15))]:
            hand=bar(a,b,0.67,0.75).rotate((0,0,90*k)).translate((x,y,0))
            hard.append(hand)
    hard.append(clock.translate((x,y,0)))
    bell=box(-4.8,4.8,-4.8,4.8,54.8,62.8)
    for k in (0,90):
        bell-=arch_cut(0,55.9,59.3,2.1,0,12).rotate((0,0,k))
    hard.append(bell.translate((x,y,0)))
    hard.append(pedestal(x,y,[(62.5,10,10,.75),(63.5,10,10,.75),
                              (68,5.5,5.5,.7),(69.3,5.2,5.2,.65)]))
    hard.append(drum(x,y,[(69,2.1,2.1),(71.2,2.1,2.1),(74.676,.4,.4)],8))
    finish(hard,[])


if __name__ == "__main__":
    build()
