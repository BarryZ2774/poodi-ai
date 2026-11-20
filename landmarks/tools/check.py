#!/usr/bin/env python3
"""Check a landmark model folder against the rules in BRIEF.md.

Usage:  python tools/check.py models/<slug> [--rebuild]

Prints one line per test: PASS, FAIL, WARN (does not fail the check) or NOTE.
Exits 0 when nothing fails, 1 when something fails, 2 on a usage error.

--rebuild  also runs build.py in a scratch copy of the folder and checks that
           it writes exactly the committed .glb.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glbmesh import GLBError, footprint_rectangle, load_model, weld  # noqa: E402

# ---- Limits from BRIEF.md ----
TRIANGLE_LIMIT = {"building": 6000, "bridge": 25000}
FILE_LIMIT_BYTES = {"building": 400_000, "bridge": 1_500_000}   # 1 KB = 1,000 bytes
BELOW_ZERO_LIMIT = {"building": 0.0, "bridge": -3.0}
MAX_MESHES = 3
MATERIAL_NAMES = ("body", "accent")
BASE_COLOUR_SRGB = (0xF4, 0xF1, 0xEC)
COLOUR_TOLERANCE_8BIT = 1.5                # in sRGB steps of 1/255
MIN_THICKNESS = 0.6
MEASURE_TOLERANCE = 0.10                   # model against the real size (rule 10)
STATED_TOLERANCE = 0.02                    # the model's own size against the value stated in facts.json
MAX_HEIGHT = 1000.0
MAX_SPAN = 5000.0
PREVIEWS = ("front.png", "side.png", "three-quarter.png", "top.png")
PREVIEW_SIZE = (1600, 1000)            # what tools/render.py writes
# Extensions the reader can decode. The brief allows no extensions at all (rule 1), so a
# file that uses one fails; it is still decoded so the other tests can report their numbers.
READABLE_EXTENSIONS = ("KHR_mesh_quantization",)

WELD = 1e-3            # metres; corners closer than this are the same corner
DEGENERATE_AREA = 1e-6  # square metres
OVERLAP_DEPTH = 0.01    # coplanar faces overlapping by more than this, in metres


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c: float) -> float:
    c = min(max(c, 0.0), 1.0)
    return c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055


class Report:
    def __init__(self):
        self.counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "NOTE": 0}

    def line(self, status: str, rule: str, text: str):
        self.counts[status] += 1
        print(f"{status:<4}  [{rule}] {text}", flush=True)

    def ok(self, rule, text):
        self.line("PASS", rule, text)

    def fail(self, rule, text):
        self.line("FAIL", rule, text)

    def warn(self, rule, text):
        self.line("WARN", rule, text)

    def note(self, rule, text):
        self.line("NOTE", rule, text)

    def check(self, passed: bool, rule: str, text: str):
        (self.ok if passed else self.fail)(rule, text)


def fmt(x: float, digits: int = 2) -> str:
    return f"{x:,.{digits}f}"


# ---------------------------------------------------------------- facts.json
def check_facts(folder: Path, rep: Report) -> dict | None:
    path = folder / "facts.json"
    if not path.is_file():
        rep.fail("facts", "facts.json is missing")
        return None
    try:
        facts = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        rep.fail("facts", f"facts.json does not parse: {exc}")
        return None
    if not isinstance(facts, dict):
        rep.fail("facts", "facts.json is not an object")
        return None

    problems = []

    def need(key, kind, where=facts, label=None):
        label = label or key
        if key not in where:
            problems.append(f"'{label}' is missing")
            return False
        if not isinstance(where[key], kind) or isinstance(where[key], bool) and kind is not bool:
            problems.append(f"'{label}' has the wrong type")
            return False
        return True

    need("slug", str)
    if need("name", str) and not facts["name"].strip():
        problems.append("'name' is empty")
    if need("kind", str) and facts["kind"] not in TRIANGLE_LIMIT:
        problems.append(f"'kind' is '{facts['kind']}', expected 'building' or 'bridge'")
    if need("anchor", dict):
        a = facts["anchor"]
        if need("lat", (int, float), a, "anchor.lat") and not -90 <= a["lat"] <= 90:
            problems.append("anchor.lat is out of range")
        if need("lon", (int, float), a, "anchor.lon") and not -180 <= a["lon"] <= 180:
            problems.append("anchor.lon is out of range")
        need("what", str, a, "anchor.what")
    if need("measurements_m", list) and not facts["measurements_m"]:
        problems.append("'measurements_m' is empty")
    if need("checkpoints", list):
        for k, cp in enumerate(facts["checkpoints"]):
            if not isinstance(cp, dict):
                problems.append(f"checkpoints[{k}] is not an object")
                continue
            for key in ("what", "lat", "lon", "min_height_m", "within_m"):
                need(key, str if key == "what" else (int, float), cp, f"checkpoints[{k}].{key}")
    if need("simplified", list) and not all(isinstance(s, str) for s in facts["simplified"]):
        problems.append("'simplified' must be a list of strings")
    if need("origin_of_work", str) and not facts["origin_of_work"].strip():
        problems.append("'origin_of_work' is empty")

    if isinstance(facts.get("slug"), str):
        if facts["slug"] != folder.name:
            problems.append(f"slug '{facts['slug']}' does not match the folder name '{folder.name}'")
        elif not re.fullmatch(r"_?[a-z0-9]+(-[a-z0-9]+)*", facts["slug"]):
            problems.append(f"slug '{facts['slug']}' should be lowercase words joined by hyphens")

    if problems:
        rep.fail("facts", "; ".join(problems))
    else:
        rep.ok("facts", f"all required fields present, slug '{facts['slug']}' matches the folder")
    if isinstance(facts.get("checkpoints"), list) and not facts["checkpoints"]:
        rep.warn("checkpoints", "no checkpoints listed, so the position on the map is not tested")
    return facts


# ---------------------------------------------------------------- files
def check_files(folder: Path, slug: str, rep: Report) -> Path | None:
    glbs = sorted(folder.glob("*.glb"))
    others = sorted(p.name for p in folder.iterdir()
                    if p.suffix.lower() in {".gltf", ".bin", ".png", ".jpg", ".jpeg", ".ktx2", ".webp"}
                    and p.is_file())
    if not (folder / "build.py").is_file():
        rep.fail("files", "build.py is missing")
    else:
        rep.ok("files", "build.py present")
    if len(glbs) != 1:
        rep.fail("1 format", f"expected exactly one .glb in the folder, found {len(glbs)}")
        return glbs[0] if glbs else None
    if glbs[0].name != f"{slug}.glb":
        rep.fail("files", f"the model is named '{glbs[0].name}', expected '{slug}.glb'")
    if others:
        rep.fail("1 format", f"model-side files next to the .glb: {', '.join(others)} "
                             "(everything must be inside the one .glb)")
    missing = [p for p in PREVIEWS if not (folder / "preview" / p).is_file()]
    if missing:
        rep.warn("files", f"preview pictures missing: {', '.join(missing)} "
                          "(make them with tools/render.py)")
    wrong = []
    for name in PREVIEWS:
        path = folder / "preview" / name
        if name in missing:
            continue
        try:
            with Image.open(path) as im:
                if im.format != "PNG" or im.size != PREVIEW_SIZE:
                    wrong.append(f"{name} is {im.format} {im.size[0]} x {im.size[1]}")
        except (OSError, ValueError):
            wrong.append(f"{name} is not a picture")
    if wrong:
        rep.fail("files", "; ".join(wrong) + f" (want PNG {PREVIEW_SIZE[0]} x {PREVIEW_SIZE[1]}): "
                 "make the pictures with tools/render.py")
    elif not missing:
        rep.ok("files", f"four preview pictures present, PNG {PREVIEW_SIZE[0]} x {PREVIEW_SIZE[1]}")
    return glbs[0]


# ---------------------------------------------------------------- rule 1
def check_contents(model, rep: Report):
    g = model.gltf
    found = []
    for key in ("images", "textures", "samplers", "animations", "cameras", "skins"):
        if g.get(key):
            found.append(f"{len(g[key])} {key}")
    if any("camera" in n for n in g.get("nodes", [])):
        found.append("a node with a camera")
    used = set(g.get("extensionsUsed", [])) | set((g.get("extensions") or {}).keys())
    if any("light" in e.lower() for e in used):
        found.append("lights (" + ", ".join(sorted(e for e in used if "light" in e.lower())) + ")")
    required = [e for e in g.get("extensionsRequired", []) if e not in READABLE_EXTENSIONS]
    if required:
        found.append("required extensions this check cannot read: " + ", ".join(required))
    def has_texture(obj) -> bool:
        if isinstance(obj, dict):
            return any(k.endswith("Texture") or has_texture(v) for k, v in obj.items())
        if isinstance(obj, list):
            return any(has_texture(v) for v in obj)
        return False

    textured = [str(m.get("name", "?")) for m in g.get("materials", []) if has_texture(m)]
    if textured:
        found.append("materials with textures: " + ", ".join(textured))
    buffers = g.get("buffers", [])
    if len(buffers) != 1 or any("uri" in b for b in buffers):
        found.append(f"{len(buffers)} buffers (need exactly one, stored inside the .glb)")
    nontri = sorted({p.mode for p in model.primitives if p.mode != 4})
    if nontri:
        found.append(f"primitives that are not triangles (mode {nontri})")
    if found:
        rep.fail("1 contents", "not allowed: " + "; ".join(found))
    else:
        rep.ok("1 contents", "no textures, images, animations, cameras, lights or external data")
    extensions = sorted(set(g.get("extensionsUsed", [])) | set(g.get("extensionsRequired", [])))
    if extensions:
        rep.fail("1 extensions", "glTF extensions are not allowed: " + ", ".join(extensions)
                 + ". Write positions and normals as plain 32-bit floats")
    else:
        rep.ok("1 extensions", "no glTF extensions")
    if model.unused_meshes:
        rep.warn("1 contents", f"meshes {model.unused_meshes} are not used by any node and never drawn")


# ---------------------------------------------------------------- rules 2 and 3
def height_reference(facts: dict | None) -> float | None:
    """The model's overall height, only from a measurement marked "bbox": "height"."""
    if not facts:
        return None
    values = [float(m["real"]) for m in facts.get("measurements_m", []) or []
              if isinstance(m, dict) and m.get("bbox") == "height"
              and isinstance(m.get("real"), (int, float)) and not isinstance(m.get("real"), bool)]
    return max(values) if values else None


def checkpoint_heights(facts: dict | None) -> list[float]:
    cps = facts.get("checkpoints") if facts else None
    if not isinstance(cps, list):
        return []
    return [float(c["min_height_m"]) for c in cps if isinstance(c, dict)
            and isinstance(c.get("min_height_m"), (int, float)) and not isinstance(c["min_height_m"], bool)]


def check_extent(model, facts, kind, rep: Report):
    lo, hi = model.V.min(axis=0), model.V.max(axis=0)
    size = hi - lo
    floor = BELOW_ZERO_LIMIT.get(kind, 0.0)
    rep.check(lo[1] >= floor - 1e-3, "2 below-ground",
              f"lowest point y = {fmt(lo[1])} m (limit {fmt(floor)} m for a {kind or 'model'})")

    rep.check(size[1] <= MAX_HEIGHT and max(size[0], size[2]) <= MAX_SPAN and size.max() >= 2.0,
              "2 units",
              f"bounding box {fmt(size[0], 1)} x {fmt(size[1], 1)} x {fmt(size[2], 1)} m (x, y, z); "
              f"height limit {fmt(MAX_HEIGHT, 0)} m, width limit {fmt(MAX_SPAN, 0)} m, at least 2 m "
              "(1 unit = 1 m)")

    ref = height_reference(facts)
    top = hi[1]
    if ref:
        ok = 0.5 * ref <= top <= 1.25 * ref + 3.0
        rep.check(ok, "2 up-axis",
                  f"top of model y = {fmt(top, 1)} m against the measurement marked "
                  f"\"bbox\": \"height\" {fmt(ref, 1)} m (allowed {fmt(0.5 * ref, 1)} to "
                  f"{fmt(1.25 * ref + 3, 1)} m; Y must be up)")
        return
    need = checkpoint_heights(facts)
    if need:
        rep.check(top >= max(need), "2 up-axis",
                  f"top of model y = {fmt(top, 1)} m; the tallest checkpoint needs "
                  f"{fmt(max(need), 1)} m (Y must be up)")
        rep.note("2 up-axis", "no measurement marked \"bbox\": \"height\", so the up axis is "
                              "judged only from the checkpoint heights")
    else:
        rep.note("2 up-axis", "not tested: no measurement marked \"bbox\": \"height\" and no "
                              "checkpoints")


# ---------------------------------------------------------------- rule 3: nothing floats
FLOAT_ABOVE = 0.5        # metres; an underside lower than this counts as standing on zero
FLOAT_MIN_AREA = 2.0     # square metres; smaller undersides (rope ends) are ignored
FLOAT_JOIN = 2.0         # neighbouring cells join one underside if their heights differ less


def _underside_maps(V, F, max_cells=1_500_000):
    """Look at the model from straight below and from straight above on a grid.

    Returns (low, top, cell, x0, z0): for every grid cell the lowest and highest
    point of the model over it (inf / -inf where there is nothing).
    """
    from render import rasterize      # the same z-buffer the pictures use
    lo, hi = V[:, [0, 2]].min(axis=0), V[:, [0, 2]].max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    cell = max(0.25, math.sqrt(span[0] * span[1] / max_cells))
    nx = int(math.ceil(span[0] / cell)) + 2
    nz = int(math.ceil(span[1] / cell)) + 2
    x0, z0 = lo[0] - cell, lo[1] - cell
    gx, gz, y = (V[:, 0] - x0) / cell, (V[:, 2] - z0) / cell, V[:, 1]
    low, _ = rasterize(gx[F], gz[F], y[F], nx, nz)
    top, _ = rasterize(gx[F], gz[F], -y[F], nx, nz)
    top = -top
    # Walls standing straight up have no area seen from below: walk along every
    # edge so their bottom and top edges are seen too.
    P = V[F]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = P[:, i], P[:, j]
        steps = np.maximum(2, np.ceil(np.linalg.norm((b - a)[:, [0, 2]], axis=1) / (cell / 2))
                           .astype(np.int64) + 1)
        k = np.repeat(np.arange(len(a)), steps)
        start = np.repeat(np.cumsum(steps) - steps, steps)
        t = (np.arange(int(steps.sum())) - start) / np.repeat(steps - 1, steps)
        p = a[k] + (b - a)[k] * t[:, None]
        ci = np.clip(((p[:, 0] - x0) / cell).astype(np.int64), 0, nx - 1)
        cj = np.clip(((p[:, 2] - z0) / cell).astype(np.int64), 0, nz - 1)
        np.minimum.at(low, cj * nx + ci, p[:, 1])
        np.maximum.at(top, cj * nx + ci, p[:, 1])
    return low.reshape(nz, nx), top.reshape(nz, nx), cell, x0, z0


def _neighbours(a, fill):
    """The four shifted copies of a 2D array, padded with `fill`."""
    pad = np.pad(a, 1, constant_values=fill)
    h, w = a.shape
    return [pad[1 + dj:1 + dj + h, 1 + di:1 + di + w] for dj, di in ((0, 1), (0, -1), (1, 0), (-1, 0))]


def floating_parts(V, F):
    """Undersides that can see the ground straight below them, and whether each is held.

    Neighbouring cells of similar height form one underside. An underside is held
    when a neighbouring cell reaches lower: the ground itself, or another underside
    that is held in turn. A wall, pier or tower beside it is what holds it.
    """
    low, top, cell, x0, z0 = _underside_maps(V, F)
    up = np.isfinite(low) & (low > FLOAT_ABOVE)
    nz, nx = low.shape
    ids = -np.ones(low.shape, dtype=np.int64)
    cells = np.flatnonzero(up.ravel())
    ids.ravel()[cells] = np.arange(len(cells))
    parent = list(range(len(cells)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    with np.errstate(invalid="ignore"):
        for dj, di in ((0, 1), (1, 0)):
            both = up[:nz - dj, :nx - di] & up[dj:, di:]
            both &= np.abs(low[:nz - dj, :nx - di] - low[dj:, di:]) <= FLOAT_JOIN
            for a, b in zip(ids[:nz - dj, :nx - di][both].tolist(), ids[dj:, di:][both].tolist()):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(len(cells))], dtype=np.int64)
    _, label = np.unique(roots, return_inverse=True) if len(roots) else (None, roots)
    G = -np.ones(low.shape, dtype=np.int64)
    G.ravel()[cells] = label.ravel()
    n = int(G.max()) + 1

    to_ground = np.zeros(n, dtype=bool)
    edges = set()
    boundary = np.zeros(n)
    held_boundary = np.zeros(n)
    for nb_low, nb_g in zip(_neighbours(low, np.inf), _neighbours(G, -2)):
        edge = (G >= 0) & (nb_g != G)
        np.add.at(boundary, G[edge], 1)
        lower = edge & (nb_low < low - 0.25)
        np.add.at(held_boundary, G[lower], 1)
        g, h = G[lower], nb_g[lower]
        to_ground[g[h < 0]] = True
        edges.update(zip(g[h >= 0].tolist(), h[h >= 0].tolist()))
    holds: dict[int, list[int]] = {}
    for a, b in edges:
        holds.setdefault(b, []).append(a)
    held = to_ground.copy()
    queue = list(np.flatnonzero(held))
    while queue:
        b = queue.pop()
        for a in holds.get(b, []):
            if not held[a]:
                held[a] = True
                queue.append(a)

    parts = []
    for g in range(n):
        jj, ii = np.nonzero(G == g)
        area = len(jj) * cell * cell
        if area < FLOAT_MIN_AREA:
            continue
        xs, zs = x0 + (ii + 0.5) * cell, z0 + (jj + 0.5) * cell
        if len(xs) > 2:
            length, width, _ = footprint_rectangle(np.stack([xs, zs], axis=1))
        else:
            length = width = 0.0
        h = low[jj, ii]
        parts.append(dict(
            held=bool(held[g]), area=area, x=float(xs.mean()), z=float(zs.mean()),
            y_min=float(h.min()), y_med=float(np.median(h)),
            length=length + cell, width=width + cell,
            body=float(np.median(top[jj, ii] - h)),
            held_share=float(held_boundary[g] / max(boundary[g], 1))))
    return parts, cell


def _describe(p):
    return (f"underside y = {fmt(p['y_min'], 1)} m, {fmt(p['length'], 0)} x {fmt(p['width'], 0)} m, "
            f"at x={p['x']:.0f}, z={p['z']:.0f}")


def check_floating(model, rep: Report):
    parts, cell = floating_parts(model.V, model.F)
    loose = sorted((p for p in parts if not p["held"]), key=lambda p: (round(p["y_min"], 1), -p["area"]))
    if loose:
        listed = "; ".join(_describe(p) for p in loose[:8])
        more = f"; and {len(loose) - 8} more" if len(loose) > 8 else ""
        rep.fail("3 floating", f"{len(loose)} parts end in mid-air: nothing under them reaches "
                               f"y = 0 and nothing beside them reaches lower (lowest first). "
                               f"{listed}{more}")
    else:
        rep.ok("3 floating", "every underside more than 0.5 m up is held by something beside it "
                             f"that reaches the ground (seen from below on a {cell:.2f} m grid)")
    # A heavy, compact block whose underside hangs well above the ground and that
    # touches something lower on only a small part of its edge is very likely a
    # support that stops short. A span, a cantilever or a portal can look alike,
    # so this is a warning, not a failure.
    heavy = [p for p in parts if p["held"] and p["area"] >= 100 and p["width"] >= 5
             and p["length"] <= 4 * p["width"] and p["y_min"] >= 2
             and p["body"] >= p["y_med"] and p["held_share"] < 0.3]
    for p in sorted(heavy, key=lambda p: -p["area"]):
        rep.warn("3 floating", f"a block {fmt(p['length'], 0)} x {fmt(p['width'], 0)} m with "
                               f"{fmt(p['body'], 0)} m of model above it hangs with its underside "
                               f"{fmt(p['y_min'], 1)} m up at x={p['x']:.0f}, z={p['z']:.0f}, held only "
                               f"along {p['held_share'] * 100:.0f}% of its edge. If it is a support "
                               "or an anchorage, it must reach y = 0")
    rep.note("3 floating", "not caught: parts that float above other parts (only what can see the "
                           "ground straight below is tested), parts held only from above, and a "
                           "roadway that runs on past its last support without an abutment")


# ---------------------------------------------------------------- rule 4
def check_weight(model, kind, file_size, rep: Report):
    limit = TRIANGLE_LIMIT.get(kind)
    tris = model.triangle_count
    if limit is None:
        rep.fail("4 triangles", f"{tris:,} triangles; limit unknown because 'kind' is not "
                                "'building' or 'bridge'")
    else:
        rep.check(tris <= limit, "4 triangles", f"{tris:,} triangles (limit {limit:,} for a {kind})")
    size_limit = FILE_LIMIT_BYTES.get(kind, min(FILE_LIMIT_BYTES.values()))
    rep.check(file_size <= size_limit, "4 file-size",
              f"{file_size:,} bytes (limit {size_limit:,} bytes for a {kind})")


# ---------------------------------------------------------------- rule 5
def check_materials(model, rep: Report):
    g = model.gltf
    meshes = g.get("meshes", [])
    rep.check(1 <= len(meshes) <= MAX_MESHES, "5 meshes",
              f"{len(meshes)} meshes (limit {MAX_MESHES})")

    mats = g.get("materials", [])
    names = [m.get("name") for m in mats]
    bad = [repr(n) for n in names if n not in MATERIAL_NAMES]
    dupes = sorted({n for n in names if names.count(n) > 1})
    unassigned = any(p.material is None for p in model.primitives)
    problems = []
    if bad:
        problems.append("names not allowed: " + ", ".join(bad))
    if dupes:
        problems.append("names used twice: " + ", ".join(dupes))
    if "body" not in names:
        problems.append("no material named 'body'")
    if unassigned:
        problems.append("a primitive has no material")
    rep.check(not problems, "5 material-names",
              ("; ".join(problems) if problems else "materials: " + ", ".join(names))
              + " (allowed: 'body' and optionally 'accent')")

    want_lin = [srgb_to_linear(c / 255) for c in BASE_COLOUR_SRGB]
    colour_bad, metal_bad, rough_bad = [], [], []
    for m in mats:
        pbr = m.get("pbrMetallicRoughness", {}) or {}
        f = pbr.get("baseColorFactor", [1, 1, 1, 1])
        name = m.get("name", "?")
        try:
            srgb = [linear_to_srgb(float(v)) * 255 for v in f[:3]]
            off = max(abs(a - b) for a, b in zip(srgb, BASE_COLOUR_SRGB))
            alpha_ok = abs(float(f[3]) - 1.0) < 1e-3
        except (TypeError, ValueError, IndexError):
            off, alpha_ok = 999, False
        if off > COLOUR_TOLERANCE_8BIT or not alpha_ok:
            hint = ""
            if all(abs(float(v) - c / 255) < 0.01 for v, c in zip(f[:3], BASE_COLOUR_SRGB)):
                hint = " (these look like sRGB values; glTF stores linear values)"
            colour_bad.append(f"{name} = [{', '.join(f'{float(v):.4f}' for v in f)}]{hint}")
        metal = pbr.get("metallicFactor", 1.0)
        if metal != 0:
            metal_bad.append(f"{name} = {metal}" + (" (glTF default is 1 when left out)"
                                                    if "metallicFactor" not in pbr else ""))
        rough = pbr.get("roughnessFactor", 1.0)
        if rough != 1:
            rough_bad.append(f"{name} = {rough}")
    want = "[" + ", ".join(f"{v:.4f}" for v in want_lin) + ", 1.0000]"
    rep.check(not colour_bad, "5 base-colour",
              ("wrong: " + "; ".join(colour_bad) if colour_bad else "all materials")
              + f" (want #F4F1EC = linear {want}, tolerance {COLOUR_TOLERANCE_8BIT} sRGB steps)")
    rep.check(not metal_bad, "5 metallic",
              ("wrong: " + "; ".join(metal_bad) if metal_bad else "metallicFactor 0") + " (want 0)")
    rep.check(not rough_bad, "5 roughness",
              ("wrong: " + "; ".join(rough_bad) if rough_bad else "roughnessFactor 1") + " (want 1)")
    coloured = [f"mesh {p.mesh} primitive {p.index}" for p in model.primitives
                if any(a.startswith("COLOR_") for a in p.attributes)]
    rep.check(not coloured, "5 vertex-colours",
              ("vertex colours in " + ", ".join(coloured)) if coloured else "no vertex colours")
    extras = [m.get("name", "?") for m in mats
              if any(v != 0 for v in m.get("emissiveFactor", [0, 0, 0]))
              or m.get("alphaMode", "OPAQUE") != "OPAQUE"]
    if extras:
        rep.warn("5 materials", "emissive or transparent settings on: " + ", ".join(extras)
                 + " (the app ignores them)")


# ---------------------------------------------------------------- geometry helpers
class Geometry:
    """Welded triangles and the per-face numbers several rules share."""

    def __init__(self, model):
        self.V = model.V
        self.F = model.F
        self.P = model.V[model.F]                                  # (f, 3, 3)
        cross = np.cross(self.P[:, 1] - self.P[:, 0], self.P[:, 2] - self.P[:, 0])
        self.double_area = np.linalg.norm(cross, axis=1)
        self.area = self.double_area / 2
        with np.errstate(invalid="ignore", divide="ignore"):
            self.normal = cross / self.double_area[:, None]
        self.normal[~np.isfinite(self.normal)] = 0.0
        self.WV, wid = weld(model.V, WELD)
        self.WF = wid[model.F]
        self.degenerate = ((self.area < DEGENERATE_AREA)
                           | (self.WF[:, 0] == self.WF[:, 1])
                           | (self.WF[:, 1] == self.WF[:, 2])
                           | (self.WF[:, 0] == self.WF[:, 2]))
        self.good = ~self.degenerate
        # Faces whose corners collapse to fewer than three points are left out of the
        # edge count; zero-area faces with three distinct corners (slivers that close a
        # T-junction) stay in, so they do not leave their neighbours' edges open.
        self.collapsed = ((self.WF[:, 0] == self.WF[:, 1]) | (self.WF[:, 1] == self.WF[:, 2])
                          | (self.WF[:, 0] == self.WF[:, 2]))
        self._edges()
        self._components()

    def _edges(self):
        F = self.WF
        directed = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
        face_of = np.tile(np.arange(len(F)), 3)
        keep = np.repeat(~self.collapsed[None, :], 3, axis=0).ravel()
        directed, face_of = directed[keep], face_of[keep]
        undirected = np.sort(directed, axis=1)
        key, inverse, counts = np.unique(undirected, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.ravel()
        forward = (directed[:, 0] < directed[:, 1]).astype(np.int64)
        fwd_count = np.bincount(inverse, weights=forward, minlength=len(key))
        self.edge_count = counts
        self.boundary_edges = int(np.sum(counts == 1))
        self.nonmanifold_edges = int(np.sum(counts > 2))
        self.flipped_edges = int(np.sum((counts == 2) & (fwd_count != 1)))
        bad_edge = (counts != 2) | (fwd_count != 1)
        self.face_on_bad_edge = np.zeros(len(F), dtype=bool)
        self.face_on_bad_edge[face_of[bad_edge[inverse]]] = True
        self.edge_key, self.edge_inverse, self.edge_face = key, inverse, face_of

    def _components(self):
        """Connected pieces: faces that share a corner belong together."""
        n = len(self.WV)
        parent = list(range(n))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b, c in self.WF[self.good].tolist():
            ra, rb, rc = find(a), find(b), find(c)
            r = min(ra, rb, rc)
            parent[ra] = parent[rb] = parent[rc] = r
        roots = np.array([find(i) for i in range(n)], dtype=np.int64)
        _, vlabel = np.unique(roots, return_inverse=True)
        self.vertex_piece = vlabel.ravel()
        self.face_piece = self.vertex_piece[self.WF[:, 0]]
        self.piece_count = int(self.face_piece[self.good].max() + 1) if self.good.any() else 0


# ---------------------------------------------------------------- rule 6
def check_normals(model, geo: Geometry, rep: Report):
    missing = [f"mesh {p.mesh} primitive {p.index}" for p in model.primitives
               if p.mode == 4 and p.normals is None]
    if missing:
        rep.fail("6 normals", "no NORMAL attribute in " + ", ".join(missing))
        return
    lengths = np.concatenate([p.normal_lengths for p in model.primitives if p.mode == 4])
    not_unit = int(np.sum(np.abs(lengths - 1.0) > 0.02))
    # Each face's stored normals should agree with the way its corners wind.
    vn = model.N[model.F].sum(axis=1)
    against = int(np.sum((np.einsum("ij,ij->i", vn, geo.normal) <= 0) & geo.good))
    problems = []
    if not_unit:
        problems.append(f"{not_unit:,} normals are not unit length")
    if against:
        problems.append(f"{against:,} faces have normals pointing against their winding")
    rep.check(not problems, "6 normals",
              "; ".join(problems) if problems else
              "normals present on every primitive, unit length, agreeing with face winding")

    # Hard edges: a corner shared (same vertex index) by two faces meeting at a sharp angle
    # means the edge is shaded smooth.
    F = model.F
    directed = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    faces = np.tile(np.arange(len(F)), 3)
    und = np.sort(directed, axis=1)
    order = np.lexsort((und[:, 1], und[:, 0]))
    und, faces = und[order], faces[order]
    same = np.all(und[1:] == und[:-1], axis=1)
    fa, fb = faces[:-1][same], faces[1:][same]
    big = (geo.area[fa] >= 0.01) & (geo.area[fb] >= 0.01)       # slivers have no meaningful normal
    fa, fb = fa[big], fb[big]
    cosang = np.einsum("ij,ij->i", geo.normal[fa], geo.normal[fb])
    # 80 degrees: smooth round parts with as few as five sides (72 degrees) are fine.
    sharp = int(np.sum(cosang < math.cos(math.radians(80))))
    if sharp:
        rep.warn("6 hard-edges", f"{sharp:,} edges sharper than 80 degrees share their normals and "
                                 "will look rounded; give each face its own corners there")


# ---------------------------------------------------------------- rule 7
def _plane_groups(geo: Geometry):
    """Yield arrays of face indices that lie in the same plane (either facing)."""
    good = np.flatnonzero(geo.good)
    if len(good) == 0:
        return
    n = geo.normal[good]
    g = np.array([0.5257, 0.7071, 0.4729])          # any fixed direction no axis lines up with
    sign = np.where(n @ g >= 0, 1.0, -1.0)
    nc = n * sign[:, None]
    d = np.einsum("ij,ij->i", nc, geo.P[good, 0])
    key = np.round(nc * 400).astype(np.int64)
    order = np.lexsort((d, key[:, 2], key[:, 1], key[:, 0]))
    key, d, faces = key[order], d[order], good[order]
    split = np.any(key[1:] != key[:-1], axis=1) | (np.diff(d) > 0.005)
    bounds = np.concatenate([[0], np.flatnonzero(split) + 1, [len(faces)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a > 1:
            yield faces[a:b]


def _overlapping_pairs(geo: Geometry, faces: np.ndarray):
    """Pairs of faces in one plane whose insides overlap by more than OVERLAP_DEPTH."""
    fn = geo.normal[faces]
    fn = fn * np.where(fn @ fn[0] >= 0, 1.0, -1.0)[:, None]    # faces may face either way
    n = fn.mean(axis=0)
    n /= np.linalg.norm(n)
    helper = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(n, helper)
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    tri = np.stack([geo.P[faces] @ u, geo.P[faces] @ v], axis=-1)  # (m, 3, 2)
    lo, hi = tri.min(axis=1), tri.max(axis=1)
    edges = np.roll(tri, -1, axis=1) - tri
    axes = np.stack([-edges[..., 1], edges[..., 0]], axis=-1)
    axes /= np.maximum(np.linalg.norm(axes, axis=-1, keepdims=True), 1e-12)
    found = []
    m = len(faces)
    chunk = max(1, 2_000_000 // m)
    for start in range(0, m, chunk):
        i = np.arange(start, min(m, start + chunk))
        cand = ((lo[i, None, 0] < hi[None, :, 0] - OVERLAP_DEPTH)
                & (lo[None, :, 0] < hi[i, None, 0] - OVERLAP_DEPTH)
                & (lo[i, None, 1] < hi[None, :, 1] - OVERLAP_DEPTH)
                & (lo[None, :, 1] < hi[i, None, 1] - OVERLAP_DEPTH))
        cand &= np.arange(m)[None, :] > i[:, None]
        a, b = np.nonzero(cand)
        if len(a) == 0:
            continue
        a = i[a]
        ax = np.concatenate([axes[a], axes[b]], axis=1)              # (p, 6, 2)
        pa = np.einsum("pkd,pjd->pkj", ax, tri[a])                    # (p, 6, 3)
        pb = np.einsum("pkd,pjd->pkj", ax, tri[b])
        depth = np.minimum(pa.max(-1), pb.max(-1)) - np.maximum(pa.min(-1), pb.min(-1))
        hit = depth.min(axis=1) > OVERLAP_DEPTH
        found.extend(zip(faces[a[hit]].tolist(), faces[b[hit]].tolist()))
    return found


def check_clean(geo: Geometry, rep: Report):
    deg = int(np.sum(geo.degenerate))
    detail = ""
    if deg:
        k = int(np.flatnonzero(geo.degenerate)[0])
        c = geo.P[k].mean(axis=0)
        detail = (f"; first near x={c[0]:.1f}, y={c[1]:.1f}, z={c[2]:.1f}, area "
                  f"{geo.area[k]:.4f} m2, shortest edge {np.min(np.linalg.norm(geo.P[k] - np.roll(geo.P[k], 1, axis=0), axis=1)) * 1000:.2f} mm")
    rep.check(deg == 0, "7 degenerate",
              f"{deg:,} degenerate triangles (area under {DEGENERATE_AREA:g} m2, or two corners "
              f"closer than {WELD * 1000:g} mm; limit 0){detail}")

    WF = geo.WF[geo.good]
    srt = np.sort(WF, axis=1)
    # Orientation of each face relative to its sorted corner order.
    rot = np.argmin(WF, axis=1)
    cyc = np.stack([WF[np.arange(len(WF)), (rot + k) % 3] for k in range(3)], axis=1)
    winding = (cyc[:, 1] == srt[:, 1]).astype(np.int64)
    _, inv, counts = np.unique(srt, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    fwd = np.bincount(inv, weights=winding, minlength=len(counts)).astype(np.int64)
    bwd = counts - fwd
    duplicate = int(np.sum(np.maximum(fwd - 1, 0) + np.maximum(bwd - 1, 0)))
    back_to_back = int(np.sum((fwd > 0) & (bwd > 0)))
    rep.check(duplicate == 0 and back_to_back == 0, "7 duplicates",
              f"{duplicate:,} duplicate faces, {back_to_back:,} back-to-back face pairs "
              "(same corners facing opposite ways); limit 0")

    same_pairs = opposite_pairs = 0
    example = None
    for group in _plane_groups(geo):
        for a, b in _overlapping_pairs(geo, group):
            if np.array_equal(np.sort(geo.WF[a]), np.sort(geo.WF[b])):
                continue                      # already counted above
            if np.dot(geo.normal[a], geo.normal[b]) > 0:
                same_pairs += 1
            else:
                opposite_pairs += 1
            if example is None:
                example = geo.P[a].mean(axis=0)
    where = ""
    if example is not None:
        where = f"; first one near x={example[0]:.1f}, y={example[1]:.1f}, z={example[2]:.1f}"
    rep.check(same_pairs == 0 and opposite_pairs == 0, "7 overlaps",
              f"{opposite_pairs:,} pairs of faces lying on each other facing opposite ways "
              f"(hidden faces between touching parts), {same_pairs:,} pairs overlapping in the "
              f"same plane facing the same way (flicker); limit 0{where}"
              + (". Leave out the face that is hidden where two parts meet"
                 if opposite_pairs else ""))

    # Inside-out pieces: a closed piece must enclose positive volume.
    # Pieces with a few open edges are still tested: their volume sign is still clear.
    inside_out, tested = 0, 0
    P = geo.P
    vol = np.einsum("ij,ij->i", P[:, 0], np.cross(P[:, 1], P[:, 2])) / 6.0
    for piece in range(geo.piece_count):
        faces = (geo.face_piece == piece) & ~geo.collapsed
        if not faces.any() or np.mean(geo.face_on_bad_edge[faces]) > 0.05:
            continue
        tested += 1
        if vol[faces].sum() < 0:
            inside_out += 1
    untested = geo.piece_count - tested
    rep.check(inside_out == 0, "7 inside-out",
              f"{inside_out} of {tested} closed or nearly closed pieces have their faces pointing "
              "inwards (limit 0)" + (f"; {untested} open pieces not tested" if untested else ""))

    if geo.boundary_edges or geo.nonmanifold_edges or geo.flipped_edges:
        rep.warn("7 watertight",
                 f"not watertight: {geo.boundary_edges:,} open edges, {geo.nonmanifold_edges:,} "
                 f"edges shared by more than two faces, {geo.flipped_edges:,} edges whose two faces "
                 "wind the same way. Fine for parts that are hidden or open by design; "
                 "otherwise look for holes")
    else:
        rep.ok("7 watertight", f"every edge is shared by exactly two faces ({geo.piece_count} pieces)")
    rep.note("7 overlaps", "not tested: faces of one part buried inside another part "
                           "(parts that pass through each other)")


# ---------------------------------------------------------------- rule 8
def piece_widths(geo: Geometry):
    """Smallest width of each connected piece.

    The width is measured across many directions: the piece's own face normals
    (exact for boxes and prisms), its principal axes, and the three world axes.
    This is the smallest side of an oriented bounding box, or tighter.
    """
    out = []
    for piece in range(geo.piece_count):
        faces = np.flatnonzero((geo.face_piece == piece) & geo.good)
        verts = np.unique(geo.WF[faces])
        pts = geo.WV[verts]
        dirs = [np.eye(3)]
        centred = pts - pts.mean(axis=0)
        if len(pts) >= 3:
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            dirs.append(vt)
        fn = np.unique(np.round(geo.normal[faces], 4), axis=0)
        fn = fn[np.linalg.norm(fn, axis=1) > 0.5]
        dirs.append(fn / np.linalg.norm(fn, axis=1, keepdims=True))
        dirs = np.concatenate(dirs)
        best, best_dir = np.inf, None
        for s in range(0, len(dirs), 256):
            d = dirs[s:s + 256]
            proj = centred @ d.T
            ext = proj.max(axis=0) - proj.min(axis=0)
            k = int(np.argmin(ext))
            if ext[k] < best:
                best, best_dir = float(ext[k]), d[k]
        out.append((best, piece, pts.mean(axis=0), best_dir, len(faces)))
    return out


def thin_walls(geo: Geometry, limit: float):
    """Probe inwards from the middle of every face to find walls thinner than `limit`.

    A ray leaves each face's centre against its normal. If the first face it meets
    within `limit` belongs to the same piece and faces away (the far side of the
    same solid), the wall is that thin there. Parts that only touch or pass
    through each other are ignored.
    """
    good = np.flatnonzero(geo.good & (geo.area > 1e-4))
    if len(good) == 0:
        return np.inf, None, 0
    P = geo.P
    reach = limit * 1.05
    origin = P[good].mean(axis=1)
    direction = -geo.normal[good]
    seg_lo = np.minimum(origin, origin + direction * reach)
    seg_hi = np.maximum(origin, origin + direction * reach)
    all_faces = np.flatnonzero(geo.good)
    tlo, thi = P[all_faces].min(axis=1), P[all_faces].max(axis=1)
    axis = int(np.argmax(geo.V.max(axis=0) - geo.V.min(axis=0)))
    t_order = np.argsort(tlo[:, axis], kind="stable")
    all_faces, tlo, thi = all_faces[t_order], tlo[t_order], thi[t_order]
    r_order = np.argsort(seg_lo[:, axis], kind="stable")

    nearest = np.full(len(good), np.inf)
    nearest_exit = np.zeros(len(good), dtype=bool)
    for s in range(0, len(good), 256):
        rays = r_order[s:s + 256]
        lo_a, hi_a = seg_lo[rays, axis].min(), seg_hi[rays, axis].max()
        upto = np.searchsorted(tlo[:, axis], hi_a, side="right")
        cands = np.flatnonzero(thi[:upto, axis] >= lo_a)
        if len(cands) == 0:
            continue
        ok = np.all((tlo[None, cands] <= seg_hi[rays, None]) & (thi[None, cands] >= seg_lo[rays, None]),
                    axis=2)
        ok &= geo.face_piece[all_faces[cands]][None, :] == geo.face_piece[good[rays]][:, None]
        ok &= all_faces[cands][None, :] != good[rays][:, None]
        ri, ci = np.nonzero(ok)
        if len(ri) == 0:
            continue
        ray = rays[ri]
        tri = all_faces[cands[ci]]
        o, d = origin[ray], direction[ray]
        v0, v1, v2 = P[tri, 0], P[tri, 1], P[tri, 2]
        e1, e2 = v1 - v0, v2 - v0
        pvec = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, pvec)
        valid = np.abs(det) > 1e-12
        inv = np.where(valid, 1.0 / np.where(valid, det, 1.0), 0.0)
        tvec = o - v0
        u = np.einsum("ij,ij->i", tvec, pvec) * inv
        qvec = np.cross(tvec, e1)
        v = np.einsum("ij,ij->i", d, qvec) * inv
        t = np.einsum("ij,ij->i", e2, qvec) * inv
        eps = 1e-7
        hit = valid & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (t > 1e-4) & (t <= reach)
        if not hit.any():
            continue
        ray, tri, t = ray[hit], tri[hit], t[hit]
        exits = np.einsum("ij,ij->i", geo.normal[tri], direction[ray]) > 0
        order = np.lexsort((t, ray))
        ray, t, exits = ray[order], t[order], exits[order]
        first = np.ones(len(ray), dtype=bool)
        first[1:] = ray[1:] != ray[:-1]
        ray, t, exits = ray[first], t[first], exits[first]
        better = t < nearest[ray]
        nearest[ray[better]] = t[better]
        nearest_exit[ray[better]] = exits[better]
    thickness = np.where(nearest_exit, nearest, np.inf)
    k = int(np.argmin(thickness))
    return float(thickness[k]), origin[k], len(good)


def check_thin(geo: Geometry, rep: Report):
    widths = piece_widths(geo)
    limit = MIN_THICKNESS * 0.995
    thin = [w for w in widths if w[0] < limit]
    if not widths:
        rep.fail("8 thin-pieces", "no pieces to measure")
        return
    worst = min(widths, key=lambda w: w[0])
    if thin:
        w, _, c, d, nf = worst
        rep.fail("8 thin-pieces",
                 f"{len(thin)} of {len(widths)} separate pieces are thinner than "
                 f"{fmt(MIN_THICKNESS)} m; thinnest {fmt(w, 3)} m ({nf} faces) near "
                 f"x={c[0]:.1f}, y={c[1]:.1f}, z={c[2]:.1f}, measured along "
                 f"({d[0]:.2f}, {d[1]:.2f}, {d[2]:.2f}). A round part is measured across its flats")
    elif len(widths) == 1:
        rep.note("8 thin-pieces", f"the model is one joined piece ({fmt(worst[0], 1)} m at its "
                                  "thinnest overall), so this says nothing about its parts; "
                                  "see thin-walls")
    else:
        rep.ok("8 thin-pieces", f"thinnest of {len(widths)} separate pieces is {fmt(worst[0], 3)} m "
                                f"(limit {fmt(MIN_THICKNESS)} m)")
    t, where, probes = thin_walls(geo, MIN_THICKNESS)
    if t < limit:
        rep.fail("8 thin-walls", f"a wall only {fmt(t, 3)} m thick near x={where[0]:.1f}, "
                                 f"y={where[1]:.1f}, z={where[2]:.1f} (limit {fmt(MIN_THICKNESS)} m; "
                                 f"{probes:,} inward probes, one per face)")
    else:
        rep.ok("8 thin-walls", f"no wall thinner than {fmt(MIN_THICKNESS)} m found by {probes:,} "
                               "inward probes, one from the middle of each face")
    rep.note("8 thin-walls", "not caught: points and knife edges (spire tips, sharp wedges), "
                             "thin spots that no face centre looks across, open single-sided "
                             "surfaces joined to a solid, and parts thin where they pass "
                             "through other parts")


# ---------------------------------------------------------------- rule 10
def check_measurements(facts, geo: Geometry, rep: Report):
    entries = facts.get("measurements_m") if facts else None
    if not isinstance(entries, list) or not entries:
        rep.fail("10 measurements", "no measurements_m in facts.json")
        return
    xz = geo.V[:, [0, 2]]
    length, width, _ = footprint_rectangle(xz) if len(xz) else (0.0, 0.0, None)
    bbox = {"length": length, "width": width, "height": float(geo.V[:, 1].max())}
    url = re.compile(r"https?://[^\s/$.?#][^\s]*\.[^\s]+", re.I)
    for k, m in enumerate(entries):
        label = f"#{k + 1}"
        if not isinstance(m, dict):
            rep.fail("10 measurements", f"{label}: not an object")
            continue
        what = m.get("what")
        label = f"#{k + 1} '{what}'" if isinstance(what, str) and what.strip() else label
        real, mod = m.get("real"), m.get("model")
        src = m.get("source")
        missing = [key for key in ("what", "real", "model") if key not in m]
        if not isinstance(src, str) or not url.fullmatch(src.strip()):
            rep.fail("10 sources", f"{label}: no source URL" if not src else
                     f"{label}: source is not a web address: {src!r}")
        else:
            rep.ok("10 sources", f"{label}: {src}")
        if missing or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                              for x in (real, mod)):
            rep.fail("10 measurements", f"{label}: needs numbers 'real' and 'model'"
                     + (f"; missing {', '.join(missing)}" if missing else ""))
            continue
        if real <= 0:
            rep.fail("10 measurements", f"{label}: 'real' must be above zero")
            continue
        diff = (mod - real) / real
        thick = m.get("thickened", False)
        if not isinstance(thick, bool):
            rep.fail("10 measurements", f"{label}: 'thickened' must be true or false, not {thick!r}")
        elif thick:
            rep.check(mod >= real, "10 measurements",
                      f"{label}: model {fmt(mod)} m vs real {fmt(real)} m ({diff * 100:+.2f}%); "
                      f"marked thickened, drawn thicker on purpose (rule 8), so the {MEASURE_TOLERANCE * 100:.0f}% rule is "
                      "skipped and the model must not be smaller than the real thing")
        else:
            rep.check(abs(diff) <= MEASURE_TOLERANCE + 1e-9, "10 measurements",
                      f"{label}: model {fmt(mod)} m vs real {fmt(real)} m ({diff * 100:+.2f}%, "
                      f"limit {MEASURE_TOLERANCE * 100:.0f}%)")
        hint = m.get("bbox")
        if hint is not None:
            if hint not in bbox:
                rep.fail("10 bbox", f"{label}: 'bbox' is {hint!r}, expected length, width or height")
                continue
            got = bbox[hint]
            d = (got - mod) / mod if mod else math.inf
            rep.check(abs(d) <= STATED_TOLERANCE + 1e-9, "10 bbox",
                      f"{label}: model's own {hint} measures {fmt(got)} m vs stated {fmt(mod)} m "
                      f"({d * 100:+.2f}%, limit {STATED_TOLERANCE * 100:.0f}%)")
    rep.note("10 bbox", "height = top of the model above the origin; length and width = sides of "
                        "the smallest rectangle around the footprint")


# ---------------------------------------------------------------- checkpoints
def latlon_to_model(lat, lon, lat0, lon0):
    """Local flat approximation around the anchor: +X east, -Z north, metres."""
    phi = math.radians(lat0)
    m_lat = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
    m_lon = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi)
    return (lon - lon0) * m_lon, -(lat - lat0) * m_lat


def height_near(geo: Geometry, cx: float, cz: float, radius: float) -> float:
    """Highest point of the model's surface whose horizontal distance to (cx, cz) <= radius."""
    P = geo.P[geo.good]
    if len(P) == 0:
        return -math.inf
    x, y, z = P[..., 0] - cx, P[..., 1], P[..., 2] - cz
    near = ((x.min(1) <= radius) & (x.max(1) >= -radius)
            & (z.min(1) <= radius) & (z.max(1) >= -radius))
    x, y, z = x[near], y[near], z[near]
    best = -math.inf
    if len(x) == 0:
        return best
    # 1) corners inside the circle
    inside = x * x + z * z <= radius * radius
    if inside.any():
        best = max(best, float(y[inside].max()))
    # 2) edges clipped to the circle: the height along an edge peaks at an end of the clip
    for i, j in ((0, 1), (1, 2), (2, 0)):
        ax, az, ay = x[:, i], z[:, i], y[:, i]
        dx, dz, dy = x[:, j] - ax, z[:, j] - az, y[:, j] - ay
        a = dx * dx + dz * dz
        b = 2 * (ax * dx + az * dz)
        c = ax * ax + az * az - radius * radius
        disc = b * b - 4 * a * c
        ok = (a > 1e-12) & (disc >= 0)
        sq = np.sqrt(np.where(ok, disc, 0))
        for sgn in (-1, 1):
            t = np.where(ok, (-b + sgn * sq) / (2 * np.where(ok, a, 1)), -1)
            t = np.clip(t, 0, 1)
            px, pz = ax + t * dx, az + t * dz
            hit = ok & (px * px + pz * pz <= radius * radius * (1 + 1e-9) + 1e-9)
            if hit.any():
                best = max(best, float((ay + t * dy)[hit].max()))
    # 3) points on the circle and its centre that fall inside a face seen from above
    ang = np.linspace(0, 2 * math.pi, 72, endpoint=False)
    sx = np.concatenate([[0.0], radius * np.cos(ang)])
    sz = np.concatenate([[0.0], radius * np.sin(ang)])
    x0, z0 = x[:, 0:1], z[:, 0:1]
    e1x, e1z, e2x, e2z = x[:, 1:2] - x0, z[:, 1:2] - z0, x[:, 2:3] - x0, z[:, 2:3] - z0
    det = e1x * e2z - e2x * e1z
    okd = np.abs(det) > 1e-9
    det = np.where(okd, det, 1.0)
    px, pz = sx[None, :] - x0, sz[None, :] - z0
    u = (px * e2z - e2x * pz) / det
    v = (e1x * pz - px * e1z) / det
    inside = okd & (u >= 0) & (v >= 0) & (u + v <= 1)
    if inside.any():
        h = y[:, 0:1] + u * (y[:, 1:2] - y[:, 0:1]) + v * (y[:, 2:3] - y[:, 0:1])
        best = max(best, float(h[inside].max()))
    return best


def check_checkpoints(facts, geo: Geometry, rep: Report):
    if not facts or not isinstance(facts.get("checkpoints"), list):
        return
    anchor = facts.get("anchor") or {}
    lat0, lon0 = anchor.get("lat"), anchor.get("lon")
    if not all(isinstance(v, (int, float)) for v in (lat0, lon0)):
        rep.fail("checkpoints", "anchor lat/lon missing, so checkpoints cannot be placed")
        return
    placed = []
    for k, cp in enumerate(facts["checkpoints"]):
        if isinstance(cp, dict) and all(isinstance(cp.get(key), (int, float)) for key in ("lat", "lon")):
            placed.append(latlon_to_model(cp["lat"], cp["lon"], lat0, lon0))
    if facts.get("kind") == "bridge":
        far = max((math.dist(a, b) for i, a in enumerate(placed) for b in placed[i + 1:]), default=0.0)
        rep.check(len(placed) >= 2 and far >= 100, "checkpoints",
                  f"a bridge needs at least two checkpoints at least 100 m apart, so that both its "
                  f"position and its direction are tested; this one has {len(placed)}"
                  + (f", at most {fmt(far, 0)} m apart" if len(placed) >= 2 else ""))
    for k, cp in enumerate(facts["checkpoints"]):
        if not isinstance(cp, dict) or not all(isinstance(cp.get(key), (int, float))
                                               for key in ("lat", "lon", "min_height_m", "within_m")):
            rep.fail("checkpoints", f"#{k + 1}: needs lat, lon, min_height_m and within_m")
            continue
        label = f"#{k + 1} '{cp.get('what', '')}'"
        cx, cz = latlon_to_model(cp["lat"], cp["lon"], lat0, lon0)
        cx, cz = cx + 0.0, cz + 0.0          # no '-0.0' in the output
        h = height_near(geo, cx, cz, float(cp["within_m"]))
        need = float(cp["min_height_m"])
        if h >= need:
            rep.ok("checkpoints", f"{label} at x={cx:.1f}, z={cz:.1f}: geometry {fmt(h, 1)} m tall "
                                  f"within {fmt(cp['within_m'], 1)} m (needs {fmt(need, 1)} m)")
            continue
        tall = geo.V[geo.V[:, 1] >= need]
        hint = ""
        if len(tall):
            dist = np.hypot(tall[:, 0] - cx, tall[:, 2] - cz)
            j = int(np.argmin(dist))
            hint = (f"; the nearest point at least that tall is {fmt(dist[j], 1)} m away at "
                    f"x={tall[j, 0]:.1f}, z={tall[j, 2]:.1f}")
        got = "nothing" if h == -math.inf else f"only {fmt(h, 1)} m"
        rep.fail("checkpoints", f"{label} at x={cx:.1f}, z={cz:.1f}: {got} within "
                                f"{fmt(cp['within_m'], 1)} m (needs {fmt(need, 1)} m){hint}")


# ---------------------------------------------------------------- rebuild
def check_rebuild(folder: Path, glb: Path, rep: Report):
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / folder.name
        shutil.copytree(folder, copy, ignore=shutil.ignore_patterns("preview", "*.glb", "__pycache__"))
        try:
            run = subprocess.run([sys.executable, "-I", "build.py"], cwd=copy, capture_output=True,
                                 text=True, timeout=600)
        except subprocess.TimeoutExpired:
            rep.fail("rebuild", "build.py ran for more than 10 minutes")
            return
        if run.returncode != 0:
            tail = (run.stderr or run.stdout).strip().splitlines()[-3:]
            rep.fail("rebuild", "build.py failed: " + " | ".join(tail))
            return
        out = copy / glb.name
        if not out.is_file():
            rep.fail("rebuild", f"build.py did not write {glb.name} next to itself")
            return
        same = out.read_bytes() == glb.read_bytes()
        rep.check(same, "rebuild", f"build.py rewrites {glb.name} "
                  + ("byte for byte" if same else "differently from the committed file"))


# ---------------------------------------------------------------- main
def run(folder: Path, rebuild: bool = False) -> int:
    rep = Report()
    print(f"Checking {folder}")
    facts = check_facts(folder, rep)
    slug = folder.name
    kind = facts.get("kind") if facts else None
    glb = check_files(folder, slug, rep)

    model = None
    if glb is not None:
        try:
            model = load_model(glb)
        except GLBError as exc:
            rep.fail("1 format", f"{glb.name} is not a valid glTF 2.0 binary: {exc}")
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            rep.fail("1 format", f"{glb.name} could not be read: {exc}")
    if model is not None:
        rep.ok("1 format", f"{glb.name} is a glTF 2.0 binary with its data inside")
        check_contents(model, rep)
        if model.triangle_count == 0:
            rep.fail("1 format", "the model draws no triangles")
            model = None
    if model is None:
        rep.fail("geometry", "rules 2 to 10 not tested because the model could not be read")
    else:
        check_extent(model, facts, kind, rep)
        check_weight(model, kind, model.file_size, rep)
        check_materials(model, rep)
        geo = Geometry(model)
        check_normals(model, geo, rep)
        check_clean(geo, rep)
        check_thin(geo, rep)
        check_floating(model, rep)
        check_measurements(facts, geo, rep)
        check_checkpoints(facts, geo, rep)
        rep.note("9 outline", "not tested by machine: the pictures are judged by eye")
        if rebuild and (folder / "build.py").is_file():
            check_rebuild(folder, glb, rep)

    c = rep.counts
    verdict = "FAIL" if c["FAIL"] else "PASS"
    print(f"\nResult: {verdict}  ({c['PASS']} passed, {c['FAIL']} failed, {c['WARN']} warnings)")
    return 1 if c["FAIL"] else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check a landmark model folder against BRIEF.md.")
    ap.add_argument("folder", type=Path, help="the model folder, for example models/golden-gate-bridge")
    ap.add_argument("--rebuild", action="store_true",
                    help="also run build.py in a scratch copy and compare the .glb it writes")
    args = ap.parse_args(argv)
    folder = args.folder.resolve()
    if not folder.is_dir():
        print(f"not a folder: {args.folder}", file=sys.stderr)
        return 2
    return run(folder, args.rebuild)


if __name__ == "__main__":
    sys.exit(main())
