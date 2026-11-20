#!/usr/bin/env python3
"""Render the four preview pictures of a landmark model.

Usage:  python tools/render.py models/<slug>

Writes preview/front.png, side.png, three-quarter.png and top.png, each
1600 x 1000 pixels. Pure numpy software rendering with a z-buffer: no GPU, no
display, and the same pixels on every run.

Views. The front view looks across the model's long side (for a square
footprint: from the south), from whichever side faces the light. The side view
looks along the long side. Both are orthographic with a ground line at y = 0.
The top view is orthographic with north up. The three-quarter view has a mild
perspective and looks from the front side and the unlit side, 28 degrees above
the ground, so two walls read with different shades.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glbmesh import GLBError, footprint_rectangle, load_model  # noqa: E402

WIDTH, HEIGHT = 1600, 1000
SS = 2                                    # supersampling for smooth edges
BACKGROUND = np.array([0xE9, 0xE5, 0xDE], dtype=np.float64)
MODEL = np.array([0xF4, 0xF1, 0xEC], dtype=np.float64)
GROUND_LINE = np.array([0xC9, 0xC3, 0xB8], dtype=np.float64)
GROUND_TINT = np.array([0xE1, 0xDC, 0xD3], dtype=np.float64)
SHADOW = np.array([0xCF, 0xCA, 0xC1], dtype=np.float64)
OUTLINE = np.array([0x9C, 0x96, 0x8C], dtype=np.float64)
TEXT = (0x8A, 0x84, 0x7A)

# Light comes from the south-west (-X, +Z) and 35 degrees above the horizon.
_EL = math.radians(35)
LIGHT = np.array([-math.cos(_EL) / math.sqrt(2), math.sin(_EL), math.cos(_EL) / math.sqrt(2)])
FILL = np.array([0.5, 0.35, -0.5]) / np.linalg.norm([0.5, 0.35, -0.5])
# The ground shadow uses the same direction but a higher sun, so it stays compact.
_SEL = math.radians(62)
SHADOW_DIR = np.array([-math.cos(_SEL) / math.sqrt(2), math.sin(_SEL), math.cos(_SEL) / math.sqrt(2)])

MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 90, 100, 120   # final-image pixels


# ---------------------------------------------------------------- cameras
class Camera:
    def __init__(self, forward, up, ortho: bool, position=None):
        f = np.asarray(forward, dtype=np.float64)
        f /= np.linalg.norm(f)
        u = np.asarray(up, dtype=np.float64)
        u = u - f * np.dot(u, f)
        u /= np.linalg.norm(u)
        self.f, self.u, self.r = f, u, np.cross(f, u)
        self.ortho = ortho
        self.position = None if position is None else np.asarray(position, dtype=np.float64)
        self.scale = 1.0      # pixels per metre (ortho) or focal length in pixels (perspective)
        self.cx = self.cy = 0.0

    def view(self, p):
        q = p if self.ortho else p - self.position
        return q @ self.r, q @ self.u, q @ self.f

    def project(self, p):
        """Screen x, y (supersampled pixels) and a depth value that is affine on screen."""
        x, y, z = self.view(p)
        if self.ortho:
            return self.cx + self.scale * x, self.cy - self.scale * y, z
        return self.cx + self.scale * x / z, self.cy - self.scale * y / z, -1.0 / z

    def fit(self, points, width, height, left, right, top, bottom):
        x, y, z = self.view(points)
        if not self.ortho:
            x, y = x / z, y / z
        w = max(x.max() - x.min(), 1e-9)
        h = max(y.max() - y.min(), 1e-9)
        self.scale = min((width - left - right) / w, (height - top - bottom) / h)
        self.cx = left + (width - left - right) / 2 - self.scale * (x.max() + x.min()) / 2
        self.cy = top + (height - top - bottom) / 2 + self.scale * (y.max() + y.min()) / 2

    def metres_per_pixel(self, depth_metric):
        return (1.0 / self.scale) if self.ortho else depth_metric / self.scale


# ---------------------------------------------------------------- rasteriser
def rasterize(sx, sy, depth, width, height):
    """Z-buffer triangles given per corner as (m, 3) arrays. Returns (zbuf, idbuf) flat."""
    zbuf = np.full(width * height, np.inf)
    idbuf = np.full(width * height, -1, dtype=np.int64)
    m = len(sx)
    if m == 0:
        return zbuf, idbuf
    x0 = np.clip(np.ceil(sx.min(1) - 0.5), 0, width - 1).astype(np.int64)
    x1 = np.clip(np.floor(sx.max(1) - 0.5), -1, width - 1).astype(np.int64)
    y0 = np.clip(np.ceil(sy.min(1) - 0.5), 0, height - 1).astype(np.int64)
    y1 = np.clip(np.floor(sy.max(1) - 0.5), -1, height - 1).astype(np.int64)
    area = ((sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0])
            - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0]))
    on_screen = (x1 >= x0) & (y1 >= y0) & (np.abs(area) > 1e-12) & np.all(np.isfinite(depth), axis=1)
    on_screen &= (sx.max(1) >= 0) & (sx.min(1) <= width) & (sy.max(1) >= 0) & (sy.min(1) <= height)

    def merge(idx, d, tid):
        order = np.lexsort((tid, d, idx))
        idx, d, tid = idx[order], d[order], tid[order]
        first = np.ones(len(idx), dtype=bool)
        first[1:] = idx[1:] != idx[:-1]
        idx, d, tid = idx[first], d[first], tid[first]
        better = d < zbuf[idx]
        zbuf[idx[better]] = d[better]
        idbuf[idx[better]] = tid[better]

    def cover(tids, px, py):
        """px, py broadcast against (n, 1, 1) triangle arrays."""
        X, Y, D, A = (sx[tids][:, :, None, None], sy[tids][:, :, None, None],
                      depth[tids][:, :, None, None], area[tids][:, None, None])
        cx, cy = px + 0.5, py + 0.5
        w0 = ((X[:, 2] - X[:, 1]) * (cy - Y[:, 1]) - (Y[:, 2] - Y[:, 1]) * (cx - X[:, 1])) / A
        w1 = ((X[:, 0] - X[:, 2]) * (cy - Y[:, 2]) - (Y[:, 0] - Y[:, 2]) * (cx - X[:, 2])) / A
        w2 = 1.0 - w0 - w1
        eps = -1e-9
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        d = w0 * D[:, 0] + w1 * D[:, 1] + w2 * D[:, 2]
        return inside, d

    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    small = on_screen & (bw <= 64) & (bh <= 64)
    cls_w = np.where(small, 1 << np.ceil(np.log2(np.maximum(bw, 1))).astype(np.int64), 0)
    cls_h = np.where(small, 1 << np.ceil(np.log2(np.maximum(bh, 1))).astype(np.int64), 0)
    for cw in (1, 2, 4, 8, 16, 32, 64):
        for ch in (1, 2, 4, 8, 16, 32, 64):
            tids_all = np.flatnonzero(small & (cls_w == cw) & (cls_h == ch))
            step = max(1, 3_000_000 // (cw * ch))
            for s in range(0, len(tids_all), step):
                tids = tids_all[s:s + step]
                ox = np.arange(cw)[None, None, :]
                oy = np.arange(ch)[None, :, None]
                px = x0[tids][:, None, None] + ox
                py = y0[tids][:, None, None] + oy
                inside, d = cover(tids, px, py)
                inside &= (px <= x1[tids][:, None, None]) & (py <= y1[tids][:, None, None])
                n, j, i = np.nonzero(inside)
                merge(py[n, j, 0] * width + px[n, 0, i], d[n, j, i], tids[n])
    for t in np.flatnonzero(on_screen & ~small):
        # Big triangles: walk their bounding box in bands to bound memory.
        for ys in range(y0[t], y1[t] + 1, max(1, 2_000_000 // int(bw[t]))):
            ye = min(y1[t], ys + max(1, 2_000_000 // int(bw[t])) - 1)
            px = np.arange(x0[t], x1[t] + 1)[None, None, :]
            py = np.arange(ys, ye + 1)[None, :, None]
            inside, d = cover(np.array([t]), px, py)
            _, j, i = np.nonzero(inside)
            pyv, pxv = py[0, j, 0], px[0, 0, i]
            merge(pyv * width + pxv, d[0, j, i], np.full(len(j), t))
    return zbuf, idbuf


# ---------------------------------------------------------------- shading
def shade_view(model, cam: Camera, W, H, name, kind_label, ground_line, shadow, scale_bar):
    V, F = model.V, model.F
    sxv, syv, dv = cam.project(V)
    sx, sy, dz = sxv[F], syv[F], dv[F]

    P = V[F]
    gn = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    gl = np.linalg.norm(gn, axis=1, keepdims=True)
    gn = np.divide(gn, gl, out=np.zeros_like(gn), where=gl > 0)
    if cam.ortho:
        back = gn @ cam.f > 0
    else:
        back = np.einsum("ij,ij->i", gn, P.mean(axis=1) - cam.position) > 0
    # Push back faces a hair further away so front faces win along shared edges.
    span = np.ptp(dz) if dz.size else 1.0
    dz_biased = dz + np.where(back, 1e-7 * max(span, 1e-9), 0.0)[:, None]

    zbuf, idbuf = rasterize(sx, sy, dz_biased, W, H)
    zbuf = zbuf.reshape(H, W)
    idbuf = idbuf.reshape(H, W)
    covered = idbuf >= 0

    # Background, ground and shadow.
    img = np.empty((H, W, 3))
    img[:] = BACKGROUND
    if ground_line:
        gy = cam.project(np.array([[0.0, 0.0, 0.0]]))[1][0]
        rows = np.arange(H)[:, None] + 0.5
        below = np.clip((rows - gy) / (90 * SS), 0, 1)
        tint = np.where(rows >= gy, 1.0 - below, 0.0)[..., None]
        img = img * (1 - tint) + GROUND_TINT * tint
        line = np.abs(rows - gy) <= 1.2 * SS
        img[np.broadcast_to(line, (H, W))] = GROUND_LINE
    if shadow:
        Q = V - SHADOW_DIR[None, :] * (V[:, 1:2] / SHADOW_DIR[1])
        qx, qy, qd = cam.project(Q)
        _, sid = rasterize(qx[F], qy[F], qd[F], W, H)
        mask = (sid.reshape(H, W) >= 0).astype(np.float64)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        soft = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(5 * SS)), dtype=np.float64) / 255
        a = 0.45 * soft[..., None]
        img = img * (1 - a) + SHADOW * a

    if covered.any():
        rows_any = np.flatnonzero(covered.any(axis=1))
        cols_any = np.flatnonzero(covered.any(axis=0))
        pad = 64 * SS
        # Crop to the model, on whole blocks of SS x SS pixels.
        r0 = max(rows_any[0] - pad, 0) // SS * SS
        c0 = max(cols_any[0] - pad, 0) // SS * SS
        r1 = min(-(-(rows_any[-1] + pad + 1) // SS) * SS, H)
        c1 = min(-(-(cols_any[-1] + pad + 1) // SS) * SS, W)
        sub = (slice(r0, r1), slice(c0, c1))
        colour, edge = _surface(model, cam, gn, back, sx, sy, dz, zbuf[sub], idbuf[sub], r0, c0)
        cov = covered[sub]
        region = img[sub]
        region[cov] = colour[cov]
        e = edge[..., None] * 0.55
        region[:] = region * (1 - e) + OUTLINE * e
        img[sub] = region

    # Downsample and finish in Pillow.
    small = img.reshape(H // SS, SS, W // SS, SS, 3).mean(axis=(1, 3))
    out = Image.fromarray(np.clip(np.round(small), 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(out)
    font = _font(24)
    small_font = _font(20)
    draw.text((40, 32), name, fill=TEXT, font=font)
    draw.text((40, 64), kind_label, fill=TEXT, font=small_font)
    if scale_bar:
        _scale_bar(draw, cam.scale / SS, small_font)
    return out


def _surface(model, cam, gn, back, sx, sy, dz, zbuf, idbuf, r0, c0):
    """Colour and outline strength for a cropped block of the image."""
    h, w = idbuf.shape
    colour = np.zeros((h, w, 3))
    cov = idbuf >= 0
    t = idbuf[cov]
    py, px = np.nonzero(cov)
    cx, cy = px + c0 + 0.5, py + r0 + 0.5
    X, Y = sx[t], sy[t]
    area = (X[:, 1] - X[:, 0]) * (Y[:, 2] - Y[:, 0]) - (X[:, 2] - X[:, 0]) * (Y[:, 1] - Y[:, 0])
    b0 = ((X[:, 2] - X[:, 1]) * (cy - Y[:, 1]) - (Y[:, 2] - Y[:, 1]) * (cx - X[:, 1])) / area
    b1 = ((X[:, 0] - X[:, 2]) * (cy - Y[:, 2]) - (Y[:, 0] - Y[:, 2]) * (cx - X[:, 2])) / area
    b = np.clip(np.stack([b0, b1, 1 - b0 - b1], axis=1), 0, 1)
    if not cam.ortho:
        zc = -1.0 / dz[t]                       # metric depth per corner
        b = b / zc
    b /= np.maximum(b.sum(axis=1, keepdims=True), 1e-12)
    corner_n = model.N[model.F[t]]              # (k, 3, 3)
    n = np.einsum("kc,kcd->kd", b, np.nan_to_num(corner_n))
    nl = np.linalg.norm(n, axis=1, keepdims=True)
    bad = (nl[:, 0] < 1e-6) | np.isnan(corner_n).any(axis=(1, 2))
    n = np.where(bad[:, None], gn[t], n / np.maximum(nl, 1e-12))
    n = np.where(back[t][:, None], -n, n)

    key = np.clip(n @ LIGHT, 0, 1)
    fill = np.clip(n @ FILL, 0, 1)
    sky = 0.5 + 0.5 * n[:, 1]
    shade = 0.71 + 0.30 * key + 0.05 * fill + 0.05 * sky

    # Depth gradient of each face's plane on screen, for crease shading and outlines.
    D = dz[t]
    det = area
    ga = ((D[:, 1] - D[:, 0]) * (Y[:, 2] - Y[:, 0]) - (D[:, 2] - D[:, 0]) * (Y[:, 1] - Y[:, 0])) / det
    gb = ((X[:, 1] - X[:, 0]) * (D[:, 2] - D[:, 0]) - (X[:, 2] - X[:, 0]) * (D[:, 1] - D[:, 0])) / det
    gx = np.zeros((h, w))
    gy = np.zeros((h, w))
    gx[cov], gy[cov] = ga, gb

    def metric(dval):
        return dval if cam.ortho else -1.0 / dval

    zpad = np.pad(zbuf, 64 * SS, constant_values=np.inf)

    def neighbour(ox, oy):
        p = 64 * SS
        return zpad[p + oy: p + oy + h, p + ox: p + ox + w]

    zm = np.where(cov, zbuf, 0.0)
    z_here = metric(np.where(cov, zbuf, -1.0 if not cam.ortho else 0.0))
    z_here = np.where(cov, z_here, 0.0)
    mpp = cam.metres_per_pixel(np.abs(z_here))

    # Crease darkening: compare the average depth of the model around each pixel
    # with what the pixel's own face would give over the same pixels. On a flat or
    # evenly sloping face the two agree; in an inside corner the surroundings come
    # forward and the pixel gets darker; outside corners never darken. Background
    # pixels are left out of both averages.
    diag = float(np.linalg.norm(np.ptp(model.V, axis=0)))
    if cam.ortho:
        gxm, gym = gx, gy
    else:
        dsq = np.where(cov, zbuf, -1.0) ** 2
        gxm, gym = gx / dsq, gy / dsq          # depth slope in metres per pixel
    cols = np.broadcast_to(np.arange(w, dtype=np.float64)[None, :], (h, w))
    rows = np.broadcast_to(np.arange(h, dtype=np.float64)[:, None], (h, w))
    covf = cov.astype(np.float64)
    # The averages are smooth, so they are taken on the final pixel grid and
    # spread back over the supersampled pixels.
    def down(a):
        return a.reshape(h // SS, SS, w // SS, SS).mean(axis=(1, 3))

    def up(a):
        return np.repeat(np.repeat(a, SS, axis=0), SS, axis=1)

    maps = np.stack([down(covf), down(covf * z_here), down(covf * cols), down(covf * rows)])
    ao = np.ones((h, w))
    for sigma, strength in ((5 * SS, 0.16), (18 * SS, 0.10)):
        weight, sz, sx_, sy_ = _blur(maps, sigma / SS)
        safe = np.where(weight > 1e-9, weight, 1.0)
        around = up(sz / safe)
        off_x = up(sx_ / safe) - cols
        off_y = up(sy_ / safe) - rows
        own_plane = z_here + gxm * off_x + gym * off_y
        reach = np.maximum(sigma * mpp, 0.015 * diag)
        delta = np.where(cov, (own_plane - around) / reach, 0.0)
        ao -= strength * np.clip(delta, 0, 1)

    shade_img = np.ones((h, w))
    shade_img[cov] = shade
    lit = MODEL[None, None, :] * (shade_img * ao)[..., None]
    colour[:] = np.clip(lit, 0, 255)

    # Outlines: the silhouette against the background, and jumps in depth.
    edge = np.zeros((h, w))
    for ox, oy in ((1, 0), (-1, 0), (0, 1), (0, -1), (SS, 0), (-SS, 0), (0, SS), (0, -SS)):
        nb = neighbour(ox, oy)
        sil = cov & ~np.isfinite(nb)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            pred = zm + gx * ox + gy * oy
            mpp = cam.metres_per_pixel(np.abs(z_here))
            jump = cov & np.isfinite(nb) & ((metric(nb) - metric(pred)) > np.maximum(
                12 * mpp * max(abs(ox), abs(oy)), 0.5))
        edge = np.maximum(edge, np.where(sil, 1.0, np.where(jump, 0.7, 0.0)))
    return colour, edge


def _box(a, k, axis):
    """Mean over a window of k pixels along one axis, zero outside the array."""
    pad = [(0, 0)] * a.ndim
    pad[axis] = (k // 2 + 1, k - k // 2)
    c = np.cumsum(np.pad(a, pad), axis=axis)
    n = a.shape[axis]
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(k, k + n)
    lo[axis] = slice(0, n)
    return (c[tuple(hi)] - c[tuple(lo)]) / k


def _blur(a, sigma):
    """Three box passes along each axis: close to a Gaussian, exact and repeatable."""
    k = max(1, int(round(sigma * 2)) | 1)
    for axis in (-2, -1):
        for _ in range(3):
            a = _box(a, k, axis)
    return a


def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                     # Pillow older than 10.1
        return ImageFont.load_default()


def _scale_bar(draw, px_per_m, font):
    target = 220 / px_per_m
    exp = 10 ** math.floor(math.log10(target))
    length = max(m * exp for m in (1, 2, 5) if m * exp <= target)
    px = length * px_per_m
    x0, y0 = 40, HEIGHT - 44
    draw.line([(x0, y0), (x0 + px, y0)], fill=TEXT, width=2)
    draw.line([(x0, y0 - 7), (x0, y0 + 7)], fill=TEXT, width=2)
    draw.line([(x0 + px, y0 - 7), (x0 + px, y0 + 7)], fill=TEXT, width=2)
    label = f"{length:g} m"
    draw.text((x0 + px + 12, y0 - 12), label, fill=TEXT, font=font)


def _north_arrow(draw, cam):
    """A small arrow in the lower right corner pointing north on the picture."""
    north = np.array([0.0, 0.0, -1.0])
    dx, dy = float(north @ cam.r), -float(north @ cam.u)
    n = math.hypot(dx, dy) or 1.0
    dx, dy = dx / n, dy / n
    cx, cy, half = WIDTH - 70, HEIGHT - 70, 30
    tip = (cx + dx * half, cy + dy * half)
    tail = (cx - dx * half, cy - dy * half)
    draw.line([tail, tip], fill=TEXT, width=3)
    left = (tip[0] - dx * 12 - dy * 8, tip[1] - dy * 12 + dx * 8)
    right = (tip[0] - dx * 12 + dy * 8, tip[1] - dy * 12 - dx * 8)
    draw.polygon([tip, left, right], fill=TEXT)
    font = _font(20)
    lx, ly = cx + dx * (half + 20), cy + dy * (half + 20)
    draw.text((lx, ly), "N", fill=TEXT, font=font, anchor="mm")


COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"]


def compass(direction_xz):
    """Name of the compass direction of a horizontal vector (+X east, -Z north)."""
    bearing = math.degrees(math.atan2(direction_xz[0], -direction_xz[1])) % 360
    return COMPASS[int((bearing + 22.5) // 45) % 8]


def principal_axis(V):
    """Main direction of the footprint as a unit (x, z) vector, from the spread of the corners.

    Used to turn the views of long models. Unlike the smallest rectangle, it follows
    the length of a bridge rather than a bulge such as a pier fender.
    """
    xz = V[:, [0, 2]] - V[:, [0, 2]].mean(axis=0)
    _, _, vt = np.linalg.svd(xz, full_matrices=False)
    a = vt[0]
    return a / np.linalg.norm(a)


def view_directions(V):
    """Horizontal unit vectors (x, z) from the model towards the front and side cameras."""
    light_h = np.array([LIGHT[0], LIGHT[2]])
    length, width, _ = footprint_rectangle(V[:, [0, 2]])
    axis = principal_axis(V)
    if width > 0 and length / width >= 1.3:
        side = axis.copy()
        front = np.array([-axis[1], axis[0]])
    else:
        side = np.array([1.0, 0.0])
        front = np.array([0.0, 1.0])
    if front @ light_h < 0:
        front = -front
    if side @ light_h < 0:
        side = -side
    return front, side


def render_folder(folder: Path, verbose=True):
    facts = {}
    if (folder / "facts.json").is_file():
        try:
            facts = json.loads((folder / "facts.json").read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            facts = {}
    glbs = sorted(folder.glob("*.glb"))
    if len(glbs) != 1:
        raise SystemExit(f"expected one .glb in {folder}, found {len(glbs)}")
    model = load_model(glbs[0])
    if model.triangle_count == 0:
        raise SystemExit("the model draws no triangles")
    name = str(facts.get("name") or folder.name) if isinstance(facts, dict) else folder.name
    V = model.V
    W, H = WIDTH * SS, HEIGHT * SS
    out_dir = folder / "preview"
    out_dir.mkdir(exist_ok=True)

    front, side = view_directions(V)
    up = np.array([0.0, 1.0, 0.0])
    lo, hi = V.min(axis=0), V.max(axis=0)
    centre = (lo + hi) / 2
    radius = float(np.linalg.norm(hi - lo) / 2) or 1.0
    fit_pts = np.vstack([V, [[centre[0], 0.0, centre[2]]]])

    def h3(d2):
        return np.array([d2[0], 0.0, d2[1]])

    tq = front - side
    tq /= np.linalg.norm(tq)
    el = math.radians(28)
    tq_dir = np.array([tq[0] * math.cos(el), math.sin(el), tq[1] * math.cos(el)])

    # Top view: north up, unless the footprint is long and thin (a bridge), when it
    # is turned so the long side runs across the picture; an arrow then shows north.
    length, width, _ = footprint_rectangle(V[:, [0, 2]])
    axis = principal_axis(V)
    if width > 0 and length / width >= 3 and abs(axis[1]) > abs(axis[0]) * 0.5:
        if axis[0] < 0 or (axis[0] == 0 and axis[1] > 0):
            axis = -axis
        top_camera = Camera([0, -1, 0], [axis[1], 0, -axis[0]], True)
        top_caption = "top, turned so the long side runs across; the arrow points north"
    else:
        top_camera = Camera([0, -1, 0], [0, 0, -1], True)
        top_caption = "top, north up"

    views = [
        ("front", Camera(-h3(front), up, True), f"front, seen from the {compass(front)}", True, False, True),
        ("side", Camera(-h3(side), up, True), f"side, seen from the {compass(side)}", True, False, True),
        ("three-quarter", Camera(-tq_dir, up, False, centre + tq_dir * radius * 3.4),
         f"three-quarter, seen from the {compass(tq)}", False, True, False),
        ("top", top_camera, top_caption, False, True, True),
    ]
    for label, cam, caption, ground, shadow, bar in views:
        start = time.perf_counter()
        cam.fit(fit_pts, W, H, MARGIN_X * SS, MARGIN_X * SS, MARGIN_TOP * SS, MARGIN_BOTTOM * SS)
        img = shade_view(model, cam, W, H, name, caption, ground, shadow, bar)
        if label == "top":
            _north_arrow(ImageDraw.Draw(img), cam)
        path = out_dir / f"{label}.png"
        img.save(path, optimize=False)
        if verbose:
            print(f"wrote {path.relative_to(folder.parent.parent) if folder.parent.parent in path.parents else path}"
                  f"  ({time.perf_counter() - start:.1f} s)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render the four preview pictures of a model folder.")
    ap.add_argument("folder", type=Path)
    args = ap.parse_args(argv)
    folder = args.folder.resolve()
    if not folder.is_dir():
        print(f"not a folder: {args.folder}", file=sys.stderr)
        return 2
    try:
        render_folder(folder)
    except GLBError as exc:
        print(f"cannot read the model: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
