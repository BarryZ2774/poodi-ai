#!/usr/bin/env python3
"""Prove that check.py passes the sample and fails broken models for the right reason.

Usage:  python tools/tests/test_check.py

Each case copies the sample tower into a scratch folder, breaks one thing,
runs check.py on it and compares the rules that FAIL with the rules expected.
"""
from __future__ import annotations

import json
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1]
ROOT = TOOLS.parent
SAMPLE = ROOT / "models" / "_sample-tower"
SLUG = SAMPLE.name
sys.path.insert(0, str(TOOLS))
from glbmesh import load_model  # noqa: E402


# ---------------------------------------------------------------- reading the sample
def sample_parts():
    """The sample as editable arrays: [(positions, normals, faces, material index)], materials."""
    model = load_model(SAMPLE / f"{SLUG}.glb")
    parts = [dict(pos=p.positions.copy(), nrm=p.normals.copy(), faces=p.faces.copy(),
                  material=p.material) for p in model.primitives]
    return parts, json.loads(json.dumps(model.gltf["materials"]))


# ---------------------------------------------------------------- writing a .glb
def write_glb(path: Path, parts, materials, normals=True, extra=None):
    blob = bytearray()
    views, accessors, prims = [], [], []

    def view(data: bytes, target=None):
        while len(blob) % 4:
            blob.append(0)
        v = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        blob.extend(data)
        return len(views) - 1

    for part in parts:
        pos = part["pos"].astype(np.float32)
        attrs = {"POSITION": len(accessors)}
        accessors.append({"bufferView": view(pos.tobytes(), 34962), "componentType": 5126,
                          "count": len(pos), "type": "VEC3",
                          "min": pos.min(axis=0).tolist(), "max": pos.max(axis=0).tolist()})
        if normals:
            attrs["NORMAL"] = len(accessors)
            accessors.append({"bufferView": view(part["nrm"].astype(np.float32).tobytes(), 34962),
                              "componentType": 5126, "count": len(pos), "type": "VEC3"})
        idx = part["faces"].astype(np.uint32).ravel()
        accessors.append({"bufferView": view(idx.tobytes(), 34963), "componentType": 5125,
                          "count": len(idx), "type": "SCALAR"})
        prims.append({"attributes": attrs, "indices": len(accessors) - 1,
                      "material": part["material"], "mode": 4})
    gltf = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}], "meshes": [{"primitives": prims}], "materials": materials,
            "accessors": accessors, "bufferViews": views}
    if extra:
        extra(gltf, view)
    while len(blob) % 4:
        blob.append(0)
    gltf["buffers"] = [{"byteLength": len(blob)}]
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    out = struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob))
    out += struct.pack("<II", len(js), 0x4E4F534A) + js
    out += struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    path.write_bytes(out)


# ---------------------------------------------------------------- shapes for the cases
def box_part(lo, hi, material=0):
    """A closed box with hard edges."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = (lo + hi) / 2
    pos, nrm, faces = [], [], []
    for axis in range(3):
        for side in (0, 1):
            n = np.zeros(3)
            n[axis] = 1 if side else -1
            u, v = [a for a in range(3) if a != axis]
            quad = []
            for du, dv in ((0, 0), (1, 0), (1, 1), (0, 1)):
                p = c.copy()
                p[axis] = hi[axis] if side else lo[axis]
                p[u] = hi[u] if du else lo[u]
                p[v] = hi[v] if dv else lo[v]
                quad.append(p)
            k = len(pos)
            pos += quad
            nrm += [n] * 4
            tri = [(k, k + 1, k + 2), (k, k + 2, k + 3)]
            for t in tri:
                a, b, d = (pos[i] for i in t)
                if np.dot(np.cross(b - a, d - a), n) < 0:
                    t = (t[0], t[2], t[1])
                faces.append(t)
    return dict(pos=np.array(pos), nrm=np.array(nrm), faces=np.array(faces), material=material)


def extrude_part(profile_xy, z0, z1, cap_tris, offset, material=0):
    """Extrude a polygon in the XY plane along Z. cap_tris index into profile_xy."""
    prof = np.asarray(profile_xy, float)
    off = np.asarray(offset, float)
    pos, nrm, faces = [], [], []

    def add(tri_pts, n):
        k = len(pos)
        a, b, c = tri_pts
        if np.dot(np.cross(b - a, c - a), n) < 0:
            b, c = c, b
        pos.extend([a + off, b + off, c + off])
        nrm.extend([n] * 3)
        faces.append((k, k + 1, k + 2))

    for z, n in ((z0, np.array([0, 0, -1.0])), (z1, np.array([0, 0, 1.0]))):
        for t in cap_tris:
            add([np.array([*prof[i], z]) for i in t], n)
    for i in range(len(prof)):
        a, b = prof[i], prof[(i + 1) % len(prof)]
        d = b - a
        n = np.array([d[1], -d[0], 0.0])
        n /= np.linalg.norm(n)
        centre = prof.mean(axis=0)
        if np.dot(n[:2], (a + b) / 2 - centre) < 0:
            n = -n
        p = [np.array([*a, z0]), np.array([*b, z0]), np.array([*b, z1]), np.array([*a, z1])]
        add([p[0], p[1], p[2]], n)
        add([p[0], p[2], p[3]], n)
    return dict(pos=np.array(pos), nrm=np.array(nrm), faces=np.array(faces), material=material)


def subdivide(part):
    """Split every triangle into four, sharing the new middle corners."""
    pos, nrm = list(part["pos"]), list(part["nrm"])
    mid = {}

    def m(a, b):
        key = (min(a, b), max(a, b))
        if key not in mid:
            pos.append((pos[a] + pos[b]) / 2)
            n = nrm[a] + nrm[b]
            nrm.append(n / np.linalg.norm(n))
            mid[key] = len(pos) - 1
        return mid[key]

    faces = []
    for a, b, c in part["faces"].tolist():
        ab, bc, ca = m(a, b), m(b, c), m(c, a)
        faces += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
    return dict(pos=np.array(pos), nrm=np.array(nrm), faces=np.array(faces), material=part["material"])


# ---------------------------------------------------------------- the cases
def case_good(folder, parts, mats, facts):
    pass


def case_too_many_triangles(folder, parts, mats, facts):
    parts[:] = [subdivide(subdivide(p)) for p in parts]


def case_texture(folder, parts, mats, facts):
    png = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000"
                        "1f15c4890000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082")

    def extra(gltf, view):
        gltf["images"] = [{"bufferView": view(png), "mimeType": "image/png"}]
        gltf["samplers"] = [{}]
        gltf["textures"] = [{"source": 0, "sampler": 0}]
        gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"] = {"index": 0}
    return extra


def case_extension(folder, parts, mats, facts):
    def extra(gltf, view):
        gltf["extensionsUsed"] = ["KHR_materials_unlit"]
    return extra


def case_wrong_colour(folder, parts, mats, facts):
    for m in mats:   # the sRGB numbers written straight in, without converting to linear
        m["pbrMetallicRoughness"]["baseColorFactor"] = [0xF4 / 255, 0xF1 / 255, 0xEC / 255, 1.0]


def case_wrong_material_name(folder, parts, mats, facts):
    mats[1]["name"] = "Material.001"


def case_thin_separate_part(folder, parts, mats, facts):
    parts.append(box_part([11, 11, -0.1], [14, 14, 0.1]))          # a 0.2 m slab


def case_thin_fin_joined(folder, parts, mats, facts):
    # One solid: a 2 m block with a 0.2 m fin on top. Its overall size is fine;
    # only the fin is thin, so only the inward probes can find it.
    profile = [(0, 0), (2, 0), (2, 2), (1.1, 2), (1.1, 4), (0.9, 4), (0.9, 2), (0, 2)]
    caps = [(0, 1, 2), (0, 2, 3), (0, 3, 6), (0, 6, 7), (6, 3, 4), (6, 4, 5)]
    parts.append(extrude_part(profile, 0, 2, caps, offset=(11, 11, -1)))


def case_below_ground(folder, parts, mats, facts):
    for p in parts:
        p["pos"][:, 1] -= 0.4


def case_wrong_unit(folder, parts, mats, facts):
    for p in parts:
        p["pos"] *= 100.0                                             # centimetres


def case_z_up(folder, parts, mats, facts):
    for p in parts:                                                   # height along -Z, not +Y
        for key in ("pos", "nrm"):
            x, y, z = p[key].T.copy()
            p[key] = np.stack([x, z, -y], axis=1)


def case_back_to_back(folder, parts, mats, facts):
    # A second copy of one face, with its own corners, turned to face the other way.
    p = parts[0]
    a, b, c = p["faces"][0]
    k = len(p["pos"])
    p["pos"] = np.vstack([p["pos"], p["pos"][[a, c, b]]])
    p["nrm"] = np.vstack([p["nrm"], -p["nrm"][[a, c, b]]])
    p["faces"] = np.vstack([p["faces"], [[k, k + 1, k + 2]]])


def case_touching_parts(folder, parts, mats, facts):
    parts.append(box_part([11, 10, -1], [14, 12, 1]))                 # sits on the podium roof


def case_no_normals(folder, parts, mats, facts):
    return "no-normals"


def case_measurement_off(folder, parts, mats, facts):
    facts["measurements_m"][3]["model"] = facts["measurements_m"][3]["real"] * 1.05


def case_checkpoint_missed(folder, parts, mats, facts):
    facts["checkpoints"][1]["lon"] += 30 / 88100                      # 30 m east


def case_missing_source(folder, parts, mats, facts):
    del facts["measurements_m"][0]["source"]


def case_thickened_thicker(folder, parts, mats, facts):
    facts["measurements_m"].append({"what": "lightning rod diameter", "real": 0.3, "model": 0.7,
                                    "thickened": True, "source": "https://example.com/sample-tower/rod"})


def case_thickened_thinner(folder, parts, mats, facts):
    facts["measurements_m"].append({"what": "spire collar diameter", "real": 0.9, "model": 0.7,
                                    "thickened": True, "source": "https://example.com/sample-tower/rod"})


def case_small_height_named(folder, parts, mats, facts):
    # A part's height in the list must not be taken for the model's height.
    facts["measurements_m"].append({"what": "podium height", "real": 10.0, "model": 10.0,
                                    "source": "https://example.com/sample-tower/drawings"})


def case_z_up_unmarked(folder, parts, mats, facts):
    case_z_up(folder, parts, mats, facts)
    for m in facts["measurements_m"]:
        m.pop("bbox", None)


def case_slug_mismatch(folder, parts, mats, facts):
    facts["slug"] = "sample-tower"


def _drop_footprint_marks(facts):
    for m in facts["measurements_m"]:
        if m.get("bbox") in ("length", "width"):
            m.pop("bbox")


def case_floating_block(folder, parts, mats, facts):
    parts.append(box_part([17, 6, -1.5], [20, 9, 1.5]))               # 6 m up, nothing under it
    _drop_footprint_marks(facts)


def case_heavy_block_at_side(folder, parts, mats, facts):
    # A 12 m block stuck to the side of the podium 4 m up: held at one side only.
    parts.append(box_part([14, 4, -6], [26, 16, 6]))
    _drop_footprint_marks(facts)


def case_sliver_closing_seam(folder, parts, mats, facts):
    # A box on the ground whose top has an extra corner in the middle of one edge,
    # with a zero-area sliver closing the seam: degenerate, but still closed.
    box = box_part([20, 0, 20], [23, 3, 23])
    top = next(i for i, f in enumerate(box["faces"]) if np.all(box["pos"][f][:, 1] == 3))
    a, b, c = box["faces"][top]
    k = len(box["pos"])
    box["pos"] = np.vstack([box["pos"], (box["pos"][a] + box["pos"][b]) / 2])
    box["nrm"] = np.vstack([box["nrm"], [[0, 1, 0]]])
    faces = [f for i, f in enumerate(box["faces"].tolist()) if i != top]
    faces += [[a, k, c], [k, b, c], [a, b, k]]                          # the last one is the sliver
    box["faces"] = np.array(faces)
    parts.append(box)
    _drop_footprint_marks(facts)


def case_wrong_preview_size(folder, parts, mats, facts):
    from PIL import Image
    path = folder / "preview" / "side.png"
    with Image.open(path) as im:
        im.resize((800, 500)).save(path)


def case_bridge_checkpoints_close(folder, parts, mats, facts):
    facts["kind"] = "bridge"                         # its two checkpoints are only 16 m apart


CASES = [
    # name, function, rules that must fail, whether those must be the only failures
    ("good sample", case_good, set(), True),
    ("too many triangles", case_too_many_triangles, {"4 triangles"}, True),
    ("a texture", case_texture, {"1 contents"}, True),
    ("a glTF extension", case_extension, {"1 extensions"}, True),
    ("sRGB colour not converted to linear", case_wrong_colour, {"5 base-colour"}, True),
    ("wrong material name", case_wrong_material_name, {"5 material-names"}, True),
    ("separate part 0.2 m thin", case_thin_separate_part, {"8 thin-pieces", "8 thin-walls"}, True),
    ("0.2 m fin joined to a block", case_thin_fin_joined, {"8 thin-walls"}, True),
    ("geometry below ground", case_below_ground, {"2 below-ground"}, True),
    ("centimetres instead of metres", case_wrong_unit, {"2 units"}, False),
    ("Z up instead of Y up", case_z_up, {"2 up-axis"}, False),
    ("Z up, no bbox height mark", case_z_up_unmarked, {"2 up-axis"}, False),
    ("a part's height listed, not marked bbox", case_small_height_named, set(), True),
    ("thickened part drawn thicker", case_thickened_thicker, set(), True),
    ("thickened part drawn thinner", case_thickened_thinner, {"10 measurements"}, True),
    ("back-to-back faces", case_back_to_back, {"7 duplicates"}, True),
    ("parts touching face to face", case_touching_parts, {"7 overlaps"}, True),
    ("normals missing", case_no_normals, {"6 normals"}, True),
    ("measurement 5% off", case_measurement_off, {"10 measurements"}, True),
    ("checkpoint missed", case_checkpoint_missed, {"checkpoints"}, True),
    ("source URL missing", case_missing_source, {"10 sources"}, True),
    ("slug does not match folder", case_slug_mismatch, {"facts"}, True),
    ("block floating beside the model", case_floating_block, {"3 floating"}, True),
    ("heavy block held at one side only", case_heavy_block_at_side, set(), True),
    ("zero-area sliver closing a seam", case_sliver_closing_seam, {"7 degenerate"}, True),
    ("preview picture the wrong size", case_wrong_preview_size, {"files"}, True),
    ("bridge checkpoints only 16 m apart", case_bridge_checkpoints_close, {"checkpoints"}, True),
]

# Rules that must WARN in a case (on top of the failures above).
WARNS = {
    "heavy block held at one side only": {"3 floating"},
}


def failing_rules(output: str, word="FAIL") -> set[str]:
    rules = set()
    for line in output.splitlines():
        if line.startswith(word) and "[" in line:
            rules.add(line[line.index("[") + 1: line.index("]")])
    return rules


def run_case(tmp: Path, name, fn, expected, exact):
    folder = tmp / name.replace(" ", "_").replace("%", "pct").replace("/", "_") / SLUG
    folder.mkdir(parents=True)
    shutil.copy(SAMPLE / "build.py", folder / "build.py")
    if (SAMPLE / "preview").is_dir():
        shutil.copytree(SAMPLE / "preview", folder / "preview")
    facts = json.loads((SAMPLE / "facts.json").read_text())
    parts, mats = sample_parts()
    result = fn(folder, parts, mats, facts)
    write_glb(folder / f"{SLUG}.glb", parts, mats, normals=result != "no-normals",
              extra=result if callable(result) else None)
    (folder / "facts.json").write_text(json.dumps(facts, indent=2))
    run = subprocess.run([sys.executable, str(TOOLS / "check.py"), str(folder)],
                         capture_output=True, text=True)
    got = failing_rules(run.stdout)
    ok_exit = (run.returncode == 0) == (not expected)
    ok_rules = got == expected if exact else expected <= got
    ok_rules &= WARNS.get(name, set()) <= failing_rules(run.stdout, "WARN")
    if not (ok_exit and ok_rules):
        print(run.stdout)
        print(run.stderr)
    return ok_exit and ok_rules, got


def main() -> int:
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, fn, expected, exact in CASES:
            ok, got = run_case(Path(tmp), name, fn, expected, exact)
            want = ", ".join(sorted(expected)) or "nothing"
            seen = ", ".join(sorted(got)) or "nothing"
            print(f"{'ok  ' if ok else 'BAD '} {name:<42} expected to fail: {want:<30} failed: {seen}")
            failures += not ok
    print(f"\n{len(CASES) - failures} of {len(CASES)} cases behave as expected")
    render_failures = render_tests()
    return 1 if failures or render_failures else 0


# ---------------------------------------------------------------- render.py
def render_tests() -> int:
    """Small tests of render.py: the close-up clipping and the caption characters."""
    import render
    bad = 0

    def report(ok, name):
        nonlocal bad
        print(f"{'ok  ' if ok else 'BAD '} render: {name}")
        bad += not ok

    # A triangle from 10 m in front of the camera to 10 m behind it: what is left is
    # the part in front of the cutting plane, and nothing behind it.
    V = np.array([[0, 0, -10], [4, 0, 10], [-4, 0, 10]], dtype=float)
    N = np.tile([0.0, 1.0, 0.0], (3, 1))
    F = np.array([[0, 1, 2]])
    pos, fwd = np.zeros(3), np.array([0.0, 0.0, -1.0])
    V2, N2, F2 = render.clip_near(V, N, F, pos, fwd)
    depth = (V2 - pos) @ fwd
    P = V2[F2]
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1).sum()
    d = 10 - render.NEAR                       # length of the part in front of the plane
    want = 0.5 * (8 * d / 20) * d
    report(len(F2) == 1 and depth.min() >= render.NEAR - 1e-9 and abs(area - want) < 1e-6,
           "a triangle reaching behind the camera is cut at the near plane")
    V3 = np.array([[0, 0, 10], [4, 0, -10], [-4, 0, -10]], dtype=float)
    V2, N2, F2 = render.clip_near(V3, N, F, pos, fwd)
    P = V2[F2]
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    report(len(F2) == 2 and ((V2 - pos) @ fwd).min() >= render.NEAR - 1e-9
           and np.all(nrm @ np.cross(V3[1] - V3[0], V3[2] - V3[0]) > 0),
           "a triangle with one corner behind becomes two, facing the same way")
    font = render._font(24)
    report(render.plain_text("Bay Bridge – West Span", font) == "Bay Bridge - West Span",
           "an en dash in a caption becomes a plain dash")
    ascii_text = "Sample Tower (invented, for testing the tools)"
    report(render.plain_text(ascii_text, font) == ascii_text, "plain text is left alone")
    return bad


if __name__ == "__main__":
    sys.exit(main())
