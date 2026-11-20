"""Build the sample tower: an invented building that shows how a model folder works.

Run:  python build.py
It writes _sample-tower.glb next to this file. Running it again gives the same bytes.

The tower is one closed surface. Each step's roof is a ring that meets the walls
of the step above along shared edges, so there are no hidden faces inside the
solid and no faces lying on top of each other.

Axes: 1 unit = 1 metre, +Y up, +X east, -Z north. The origin is the anchor in
facts.json, at ground level.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pygltflib

HERE = Path(__file__).resolve().parent
SLUG = HERE.name

# Base colour #F4F1EC, converted from sRGB to the linear values glTF stores.
def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


BASE_COLOUR = [round(srgb_to_linear(v / 255), 6) for v in (0xF4, 0xF1, 0xEC)] + [1.0]

# ---- Dimensions (metres). These are the numbers facts.json reports. ----
PODIUM = dict(half=15.0, notch=3.0, y0=0.0, y1=10.0)
SHAFT_A = dict(half=10.0, notch=2.0, y0=10.0, y1=46.0)
SHAFT_B = dict(half=8.0, notch=1.6, y0=46.0, y1=70.0)
SHAFT_C = dict(half=6.0, notch=1.2, y0=70.0, y1=86.0)
DRUM = dict(radius=4.5, y0=86.0, y1=94.0)          # accent
SPIRE = dict(r_base=2.2, r_mid=0.9, r_tip=0.35, y0=94.0, y_mid=112.0, y_tip=124.0)
# The tip is a flat disc 0.7 m across, not a point: rule 8 wants nothing thinner than 0.6 m.
SEGMENTS = 64                                       # for the round parts


# ---- Outlines in the ground plane, as (x, z) points around the vertical axis ----
def notched_square(half: float, notch: float) -> np.ndarray:
    """Square with a small step cut out of each corner (12 points)."""
    h, n = half, notch
    corner = [(h, h - n), (h - n, h - n), (h - n, h)]       # one corner, +x +z
    pts = []
    for k in range(4):                                       # rotate it four times
        c, s = [(1, 0), (0, 1), (-1, 0), (0, -1)][k]
        pts += [(c * x - s * z, s * x + c * z) for x, z in corner]
    return np.array(pts)


def circle(radius: float, segments: int = SEGMENTS) -> np.ndarray:
    a = np.arange(segments) * (2 * math.pi / segments)
    return np.stack([radius * np.cos(a), radius * np.sin(a)], axis=1)


# ---- A small mesh builder: triangles with one normal per corner ----
class Mesh:
    def __init__(self):
        self.tris: list[np.ndarray] = []      # each (3, 3) positions
        self.norms: list[np.ndarray] = []     # each (3, 3) normals

    def add(self, a, b, c, facing, normals=None):
        """Add triangle a, b, c turned so its front faces along `facing`."""
        p = np.array([a, b, c], dtype=np.float64)
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.dot(n, facing) < 0:
            p = p[[0, 2, 1]]
            if normals is not None:
                normals = [normals[0], normals[2], normals[1]]
            n = -n
        if normals is None:                    # hard edge: the face's own normal
            normals = [n / np.linalg.norm(n)] * 3
        self.tris.append(p)
        self.norms.append(np.array(normals, dtype=np.float64))


def xyz(p2, y):
    return np.array([p2[0], y, p2[1]])


def outward(p2):
    return np.array([p2[0], 0.0, p2[1]])


def walls(mesh: Mesh, outline, y0, y1):
    """Vertical walls with hard edges."""
    for i in range(len(outline)):
        a, b = outline[i], outline[(i + 1) % len(outline)]
        mid = outward((a + b) / 2)
        mesh.add(xyz(a, y0), xyz(b, y0), xyz(b, y1), mid)
        mesh.add(xyz(a, y0), xyz(b, y1), xyz(a, y1), mid)


def round_walls(mesh: Mesh, lower, y0, upper, y1):
    """Smooth walls between two circles with the same number of points."""
    m = len(lower)
    for i in range(m):
        j = (i + 1) % m

        def normal(k):
            d = lower[k] / np.linalg.norm(lower[k])
            slope = (np.linalg.norm(lower[k]) - np.linalg.norm(upper[k])) / (y1 - y0)
            v = np.array([d[0], slope, d[1]])
            return v / np.linalg.norm(v)

        mid = outward((lower[i] + lower[j]) / 2)
        mesh.add(xyz(lower[i], y0), xyz(lower[j], y0), xyz(upper[j], y1), mid,
                 [normal(i), normal(j), normal(j)])
        mesh.add(xyz(lower[i], y0), xyz(upper[j], y1), xyz(upper[i], y1), mid,
                 [normal(i), normal(j), normal(i)])


def ring(mesh: Mesh, outer, inner, y):
    """Flat roof between two outlines around the axis, facing up.

    Both outlines are walked around in order of angle, each step taking the
    shorter of the two possible diagonals. Every point of both outlines is used,
    so the roof's edges match the walls above and below exactly.
    """
    def by_angle(pts):
        ang = np.arctan2(pts[:, 1], pts[:, 0])
        order = np.argsort(ang, kind="stable")
        return pts[order], ang[order]

    o, oa = by_angle(outer)
    n, na = by_angle(inner)
    # Close both loops by repeating the first point one turn later.
    o = np.vstack([o, o[:1]])
    oa = np.append(oa, oa[0] + 2 * math.pi)
    n = np.vstack([n, n[:1]])
    na = np.append(na, na[0] + 2 * math.pi)
    i = j = 0
    up = np.array([0.0, 1.0, 0.0])
    while i < len(o) - 1 or j < len(n) - 1:
        if i == len(o) - 1:
            advance_outer = False
        elif j == len(n) - 1:
            advance_outer = True
        else:
            advance_outer = (np.linalg.norm(o[i + 1] - n[j]) <= np.linalg.norm(o[i] - n[j + 1]))
        if advance_outer:
            mesh.add(xyz(o[i], y), xyz(o[i + 1], y), xyz(n[j], y), up)
            i += 1
        else:
            mesh.add(xyz(o[i], y), xyz(n[j + 1], y), xyz(n[j], y), up)
            j += 1


def disc(mesh: Mesh, outline, y, facing_up: bool):
    """Flat cap as a fan from the centre: the underside, or the top of the spire."""
    facing = np.array([0.0, 1.0 if facing_up else -1.0, 0.0])
    for i in range(len(outline)):
        a, b = outline[i], outline[(i + 1) % len(outline)]
        mesh.add(xyz(a, y), xyz(b, y), np.array([0.0, y, 0.0]), facing)


def build():
    body, accent = Mesh(), Mesh()
    steps = [PODIUM, SHAFT_A, SHAFT_B, SHAFT_C]
    outlines = [notched_square(s["half"], s["notch"]) for s in steps]

    disc(body, outlines[0], 0.0, facing_up=False)
    for k, step in enumerate(steps):
        walls(body, outlines[k], step["y0"], step["y1"])
        if k + 1 < len(steps):
            ring(body, outlines[k], outlines[k + 1], step["y1"])

    drum = circle(DRUM["radius"])
    spire_base = circle(SPIRE["r_base"])
    spire_mid = circle(SPIRE["r_mid"])
    spire_tip = circle(SPIRE["r_tip"])
    ring(body, outlines[-1], drum, SHAFT_C["y1"])
    round_walls(accent, drum, DRUM["y0"], drum, DRUM["y1"])
    ring(accent, drum, spire_base, DRUM["y1"])
    round_walls(body, spire_base, SPIRE["y0"], spire_mid, SPIRE["y_mid"])
    round_walls(body, spire_mid, SPIRE["y_mid"], spire_tip, SPIRE["y_tip"])
    disc(body, spire_tip, SPIRE["y_tip"], facing_up=True)
    return {"body": body, "accent": accent}


# ---- Writing the .glb ----
def indexed(mesh: Mesh):
    """Share corners that have the same position and normal."""
    pos = np.concatenate(mesh.tris).astype(np.float32)
    nrm = np.concatenate(mesh.norms).astype(np.float32)
    key = np.concatenate([pos, nrm], axis=1)
    uniq, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)                  # keep first-seen order, stable output
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return pos[first[order]], nrm[first[order]], remap[inverse.ravel()].astype(np.uint32)


def write_glb(parts: dict[str, Mesh], path: Path):
    blob = bytearray()
    views, accessors, primitives, materials = [], [], [], []

    def add_view(data: bytes, target: int) -> int:
        while len(blob) % 4:
            blob.append(0)
        views.append(pygltflib.BufferView(buffer=0, byteOffset=len(blob),
                                          byteLength=len(data), target=target))
        blob.extend(data)
        return len(views) - 1

    for name, mesh in parts.items():
        if not mesh.tris:
            continue
        pos, nrm, idx = indexed(mesh)
        index_type = np.uint16 if len(pos) < 65536 else np.uint32
        v_pos = add_view(pos.tobytes(), pygltflib.ARRAY_BUFFER)
        v_nrm = add_view(nrm.tobytes(), pygltflib.ARRAY_BUFFER)
        v_idx = add_view(idx.astype(index_type).tobytes(), pygltflib.ELEMENT_ARRAY_BUFFER)
        accessors.append(pygltflib.Accessor(
            bufferView=v_pos, componentType=pygltflib.FLOAT, count=len(pos), type="VEC3",
            min=[float(v) for v in pos.min(axis=0)], max=[float(v) for v in pos.max(axis=0)]))
        accessors.append(pygltflib.Accessor(
            bufferView=v_nrm, componentType=pygltflib.FLOAT, count=len(nrm), type="VEC3"))
        accessors.append(pygltflib.Accessor(
            bufferView=v_idx, count=len(idx), type="SCALAR",
            componentType=pygltflib.UNSIGNED_SHORT if index_type is np.uint16
            else pygltflib.UNSIGNED_INT))
        materials.append(pygltflib.Material(
            name=name,
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=BASE_COLOUR, metallicFactor=0.0, roughnessFactor=1.0)))
        a = len(accessors)
        primitives.append(pygltflib.Primitive(
            attributes=pygltflib.Attributes(POSITION=a - 3, NORMAL=a - 2),
            indices=a - 1, material=len(materials) - 1, mode=pygltflib.TRIANGLES))

    while len(blob) % 4:
        blob.append(0)
    gltf = pygltflib.GLTF2(
        asset=pygltflib.Asset(version="2.0"),
        scene=0,
        scenes=[pygltflib.Scene(nodes=[0])],
        nodes=[pygltflib.Node(name=SLUG, mesh=0)],
        meshes=[pygltflib.Mesh(name=SLUG, primitives=primitives)],
        materials=materials,
        accessors=accessors,
        bufferViews=views,
        buffers=[pygltflib.Buffer(byteLength=len(blob))],
    )
    gltf.set_binary_blob(bytes(blob))
    gltf.save_binary(str(path))


def main():
    parts = build()
    out = HERE / f"{SLUG}.glb"
    write_glb(parts, out)
    tris = sum(len(m.tris) for m in parts.values())
    print(f"wrote {out.name}: {tris} triangles, {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
