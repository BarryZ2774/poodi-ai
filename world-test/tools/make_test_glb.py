#!/usr/bin/env python3 -I
"""Writes a test landmark model: WorldTest/Resources/test-tower.glb

A stepped tower, 60 m wide at the base and 200 m tall, in four square tiers.
Axes as agreed for landmark models: 1 unit = 1 metre, Y up, +X east, -Z north,
origin at the centre of the base. One material: base colour #F4F1EC, metallic 0,
roughness 1, no textures. Flat-shaded (each face has its own vertices and normal).

Usage: python3 tools/make_test_glb.py [out.glb]
"""
import json, os, struct, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "WorldTest/Resources/test-tower.glb")

# (width in metres, top of the tier in metres); each tier starts where the one below ends.
TIERS = [(60, 90), (46, 140), (32, 180), (16, 200)]

def srgb_to_linear(c):
    c /= 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

def box(x0, x1, y0, y1, z0, z1):
    """Six quads (4 vertices each, counter-clockwise seen from outside) with outward normals."""
    faces = [
        ((1, 0, 0), [(x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1)]),   # east
        ((-1, 0, 0), [(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)]),  # west
        ((0, 1, 0), [(x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)]),   # top
        ((0, -1, 0), [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)]),  # bottom
        ((0, 0, 1), [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]),   # south (+Z)
        ((0, 0, -1), [(x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0)]),  # north (-Z)
    ]
    return faces

def main():
    positions, normals, indices = [], [], []
    base = 0.0
    for width, top in TIERS:
        h = width / 2
        for normal, quad in box(-h, h, base, top, -h, h):
            start = len(positions)
            positions += quad
            normals += [normal] * 4
            indices += [start, start + 1, start + 2, start, start + 2, start + 3]
        base = top

    pos_bytes = b"".join(struct.pack("<3f", *p) for p in positions)
    nrm_bytes = b"".join(struct.pack("<3f", *n) for n in normals)
    idx_bytes = b"".join(struct.pack("<H", i) for i in indices)
    idx_bytes += b"\0" * (-len(idx_bytes) % 4)
    binary = pos_bytes + nrm_bytes + idx_bytes

    xs, ys, zs = zip(*positions)
    gltf = {
        "asset": {"version": "2.0", "generator": "World Test make_test_glb.py"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "test-tower", "mesh": 0}],
        "meshes": [{"name": "test-tower", "primitives": [{
            "attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0, "mode": 4}]}],
        "materials": [{
            "name": "porcelain",
            "pbrMetallicRoughness": {
                "baseColorFactor": [round(srgb_to_linear(c), 6) for c in (0xF4, 0xF1, 0xEC)] + [1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 1.0,
            },
        }],
        "buffers": [{"byteLength": len(binary)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bytes), "byteLength": len(nrm_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_bytes) + len(nrm_bytes), "byteLength": len(indices) * 2, "target": 34963},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3",
             "min": [min(xs), min(ys), min(zs)], "max": [max(xs), max(ys), max(zs)]},
            {"bufferView": 1, "componentType": 5126, "count": len(normals), "type": "VEC3"},
            {"bufferView": 2, "componentType": 5123, "count": len(indices), "type": "SCALAR"},
        ],
    }
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(OUT, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(binary), b"BIN\0") + binary)
    print(f"{OUT}: {total} bytes, {len(positions)} vertices, {len(indices) // 3} triangles")

if __name__ == "__main__":
    main()
