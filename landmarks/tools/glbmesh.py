"""Read a .glb file into plain numpy arrays.

Shared by check.py and render.py. Only numpy and the standard library are used,
so the reader also reports exactly what is wrong with a broken file.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field

import numpy as np

GLB_MAGIC = 0x46546C67  # b"glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942

COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
COMPONENT_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


class GLBError(Exception):
    """The file is not a readable glTF 2.0 binary."""


def parse_glb(data: bytes) -> tuple[dict, bytes | None]:
    """Split a GLB container into its JSON document and its BIN chunk."""
    if len(data) < 20:
        raise GLBError("file is shorter than a GLB header")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise GLBError("not a GLB file (the first four bytes are not 'glTF')")
    if version != 2:
        raise GLBError(f"GLB container version {version}, expected 2")
    if length != len(data):
        raise GLBError(f"GLB header says {length} bytes but the file has {len(data)}")
    offset = 12
    chunks: list[tuple[int, bytes]] = []
    while offset < length:
        if offset + 8 > length:
            raise GLBError("truncated chunk header")
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        if offset + 8 + chunk_len > length:
            raise GLBError("a chunk runs past the end of the file")
        if chunk_len % 4:
            raise GLBError("a chunk length is not a multiple of 4 bytes")
        chunks.append((chunk_type, data[offset + 8: offset + 8 + chunk_len]))
        offset += 8 + chunk_len
    if not chunks or chunks[0][0] != CHUNK_JSON:
        raise GLBError("the first chunk is not the JSON chunk")
    try:
        gltf = json.loads(chunks[0][1].decode("utf-8").rstrip(" \x00"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GLBError(f"the JSON chunk does not parse: {exc}") from None
    if not isinstance(gltf, dict):
        raise GLBError("the JSON chunk is not an object")
    bin_chunk = None
    if len(chunks) > 1 and chunks[1][0] == CHUNK_BIN:
        bin_chunk = chunks[1][1]
    version_str = str(gltf.get("asset", {}).get("version", ""))
    if version_str != "2.0":
        raise GLBError(f"asset.version is '{version_str}', expected '2.0'")
    return gltf, bin_chunk


def read_accessor(gltf: dict, bin_chunk: bytes | None, index: int) -> np.ndarray:
    """Return accessor `index` as an array of shape (count, components)."""
    try:
        acc = gltf["accessors"][index]
    except (KeyError, IndexError, TypeError):
        raise GLBError(f"accessor {index} does not exist") from None
    if "sparse" in acc:
        raise GLBError(f"accessor {index} is sparse; write plain accessors")
    if acc.get("componentType") not in COMPONENT_DTYPES or acc.get("type") not in COMPONENT_COUNT:
        raise GLBError(f"accessor {index} has an unknown component type or type")
    dtype = np.dtype(COMPONENT_DTYPES[acc["componentType"]]).newbyteorder("<")
    ncomp = COMPONENT_COUNT[acc["type"]]
    count = int(acc.get("count", 0))
    if "bufferView" not in acc:
        return np.zeros((count, ncomp), dtype=dtype.newbyteorder("="))
    try:
        view = gltf["bufferViews"][acc["bufferView"]]
        buffer = gltf["buffers"][view["buffer"]]
    except (KeyError, IndexError, TypeError):
        raise GLBError(f"accessor {index} points at a missing bufferView or buffer") from None
    if "uri" in buffer or view["buffer"] != 0:
        raise GLBError("geometry lives outside the GLB binary chunk")
    if bin_chunk is None:
        raise GLBError("the file has no binary chunk")
    elem = dtype.itemsize * ncomp
    stride = view.get("byteStride") or elem
    view_start = view.get("byteOffset", 0)
    view_end = view_start + view.get("byteLength", 0)
    start = view_start + acc.get("byteOffset", 0)
    if count:
        end = start + stride * (count - 1) + elem
        if end > view_end or view_end > len(bin_chunk):
            raise GLBError(f"accessor {index} reads past the end of its data")
    arr = np.ndarray(shape=(count, ncomp), dtype=dtype, buffer=bin_chunk,
                     offset=start, strides=(stride, dtype.itemsize)) if count else \
        np.zeros((0, ncomp), dtype=dtype)
    arr = np.array(arr, dtype=dtype.newbyteorder("="))
    if acc.get("normalized") and dtype.kind in "iu":
        info = np.iinfo(dtype)
        arr = np.maximum(arr.astype(np.float64) / info.max, -1.0)
    return arr


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    t = np.array(node.get("translation", [0, 0, 0]), dtype=np.float64)
    x, y, z, w = node.get("rotation", [0, 0, 0, 1])
    s = np.array(node.get("scale", [1, 1, 1]), dtype=np.float64)
    r = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    m = np.eye(4)
    m[:3, :3] = r * s[None, :]
    m[:3, 3] = t
    return m


@dataclass
class Primitive:
    mesh: int
    index: int
    material: int | None
    mode: int
    attributes: list[str]
    positions: np.ndarray            # (n, 3) float64, world space
    normals: np.ndarray | None       # (n, 3) float64, world space
    faces: np.ndarray                # (m, 3) int64, into positions
    normal_lengths: np.ndarray | None = None  # length of each normal as stored in the file


@dataclass
class Model:
    gltf: dict
    file_size: int
    primitives: list[Primitive] = field(default_factory=list)
    unused_meshes: list[int] = field(default_factory=list)
    # Everything joined: vertices, normals (NaN where missing), faces, per-face data.
    V: np.ndarray = None
    N: np.ndarray = None
    F: np.ndarray = None
    face_material: np.ndarray = None
    face_primitive: np.ndarray = None

    @property
    def triangle_count(self) -> int:
        return int(len(self.F))


def load_model(path) -> Model:
    """Read a GLB and return every drawn triangle in world space."""
    with open(path, "rb") as fh:
        data = fh.read()
    gltf, bin_chunk = parse_glb(data)
    model = Model(gltf=gltf, file_size=len(data))

    nodes = gltf.get("nodes", [])
    meshes = gltf.get("meshes", [])
    scenes = gltf.get("scenes", [])
    if scenes:
        scene_index = gltf.get("scene", 0)
        try:
            roots = scenes[scene_index].get("nodes", [])
        except (IndexError, TypeError):
            raise GLBError(f"default scene {scene_index} does not exist") from None
    else:
        children = {c for n in nodes for c in n.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]

    used_meshes: set[int] = set()
    stack = [(r, np.eye(4), 0) for r in reversed(roots)]
    while stack:
        ni, parent, depth = stack.pop()
        if depth > 64 or not (0 <= ni < len(nodes)):
            raise GLBError("node hierarchy is broken (missing node or a cycle)")
        node = nodes[ni]
        world = parent @ _node_matrix(node)
        if "mesh" in node:
            mi = node["mesh"]
            if not (0 <= mi < len(meshes)):
                raise GLBError(f"node {ni} points at missing mesh {mi}")
            used_meshes.add(mi)
            for pi, prim in enumerate(meshes[mi].get("primitives", [])):
                model.primitives.append(_read_primitive(gltf, bin_chunk, mi, pi, prim, world))
        for child in reversed(node.get("children", [])):
            stack.append((child, world, depth + 1))
    model.unused_meshes = [i for i in range(len(meshes)) if i not in used_meshes]

    tri_prims = [p for p in model.primitives if p.mode == 4]
    if tri_prims:
        offsets = np.cumsum([0] + [len(p.positions) for p in tri_prims])
        model.V = np.concatenate([p.positions for p in tri_prims])
        model.N = np.concatenate([
            p.normals if p.normals is not None else np.full_like(p.positions, np.nan)
            for p in tri_prims])
        model.F = np.concatenate([p.faces + offsets[k] for k, p in enumerate(tri_prims)])
        model.face_material = np.concatenate([
            np.full(len(p.faces), -1 if p.material is None else p.material) for p in tri_prims])
        model.face_primitive = np.concatenate([
            np.full(len(p.faces), k) for k, p in enumerate(tri_prims)])
    else:
        model.V = np.zeros((0, 3))
        model.N = np.zeros((0, 3))
        model.F = np.zeros((0, 3), dtype=np.int64)
        model.face_material = np.zeros(0, dtype=np.int64)
        model.face_primitive = np.zeros(0, dtype=np.int64)
    return model


def _read_primitive(gltf, bin_chunk, mi, pi, prim, world) -> Primitive:
    attrs = prim.get("attributes", {})
    if "POSITION" not in attrs:
        raise GLBError(f"mesh {mi} primitive {pi} has no POSITION")
    pos = read_accessor(gltf, bin_chunk, attrs["POSITION"]).astype(np.float64)
    if pos.shape[1] != 3:
        raise GLBError(f"mesh {mi} primitive {pi}: POSITION is not VEC3")
    if not np.all(np.isfinite(pos)):
        raise GLBError(f"mesh {mi} primitive {pi}: POSITION holds NaN or infinity")
    nrm = None
    if "NORMAL" in attrs:
        nrm = read_accessor(gltf, bin_chunk, attrs["NORMAL"]).astype(np.float64)
        if nrm.shape != pos.shape:
            raise GLBError(f"mesh {mi} primitive {pi}: NORMAL count differs from POSITION")
    mode = prim.get("mode", 4)
    if "indices" in prim:
        idx = read_accessor(gltf, bin_chunk, prim["indices"]).astype(np.int64).ravel()
    else:
        idx = np.arange(len(pos), dtype=np.int64)
    if len(idx) and (idx.min() < 0 or idx.max() >= len(pos)):
        raise GLBError(f"mesh {mi} primitive {pi}: an index points past the vertex list")
    if mode == 4:
        if len(idx) % 3:
            raise GLBError(f"mesh {mi} primitive {pi}: index count is not a multiple of 3")
        faces = idx.reshape(-1, 3)
    else:
        faces = np.zeros((0, 3), dtype=np.int64)

    lin = world[:3, :3]
    if abs(np.linalg.det(lin)) < 1e-12:
        raise GLBError(f"a node that draws mesh {mi} has a singular transform")
    pos = pos @ lin.T + world[:3, 3]
    normal_lengths = None
    if nrm is not None:
        normal_lengths = np.linalg.norm(nrm, axis=1)
        nrm = nrm @ np.linalg.inv(lin)  # inverse transpose, applied to row vectors
        n = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm = np.divide(nrm, n, out=np.zeros_like(nrm), where=n > 0)
    if np.linalg.det(lin) < 0:
        faces = faces[:, [0, 2, 1]]
    return Primitive(mesh=mi, index=pi, material=prim.get("material"), mode=mode,
                     attributes=sorted(attrs), positions=pos, normals=nrm, faces=faces,
                     normal_lengths=normal_lengths)


def weld(points: np.ndarray, tol: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Merge points closer than about `tol`. Returns (unique points, index per input point)."""
    if len(points) == 0:
        return points.copy(), np.zeros(0, dtype=np.int64)
    key = np.round(points / tol).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return points[first], inverse.ravel()


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. Returns hull vertices counter-clockwise."""
    pts = np.unique(np.round(points, 4), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def build(seq):
        out: list = []
        for p in seq:
            while len(out) >= 2:
                (ax, ay), (bx, by) = out[-2], out[-1]
                if (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax) <= 0:
                    out.pop()
                else:
                    break
            out.append((p[0], p[1]))
        return out

    seq = pts.tolist()
    lower = build(seq)
    upper = build(reversed(seq))
    return np.array(lower[:-1] + upper[:-1])


def footprint_rectangle(xz: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Smallest-area rectangle around points in the ground plane.

    Returns (length, width, unit vector of the long side in (x, z)).
    """
    hull = convex_hull_2d(xz)
    if len(hull) < 3:
        if len(hull) == 2:
            d = hull[1] - hull[0]
            n = np.linalg.norm(d)
            return float(n), 0.0, d / n if n else np.array([1.0, 0.0])
        return 0.0, 0.0, np.array([1.0, 0.0])
    edges = np.roll(hull, -1, axis=0) - hull
    lengths = np.linalg.norm(edges, axis=1)
    keep = lengths > 1e-9
    dirs = edges[keep] / lengths[keep, None]
    perp = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)
    a = hull @ dirs.T
    b = hull @ perp.T
    ext_a = a.max(axis=0) - a.min(axis=0)
    ext_b = b.max(axis=0) - b.min(axis=0)
    best = int(np.argmin(ext_a * ext_b))
    ea, eb = float(ext_a[best]), float(ext_b[best])
    if ea >= eb:
        return ea, eb, dirs[best]
    return eb, ea, perp[best]
