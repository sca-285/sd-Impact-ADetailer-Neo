"""Capsule renderer: the hand a set of landmarks implies, as a mask or a depth map.

Pure numpy and deterministic: the same landmarks always give the same pixels.
Each capsule only touches its own bounding box, so a 1024 px crop renders in a
few milliseconds.

Hand landmark order (DWPose and MediaPipe alike):
  0 wrist, 1-4 thumb (CMC, MCP, IP, tip), 5-8 index (MCP, PIP, DIP, tip),
  9-12 middle, 13-16 ring, 17-20 pinky.
"""

from __future__ import annotations

import numpy as np

FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
FINGER_ORDER = ("thumb", "index", "middle", "ring", "pinky")

# The 20 bones: wrist to each finger base, then along each finger.
HAND_BONES = [(0, chain[0]) for chain in FINGERS.values()] + [
    (chain[i], chain[i + 1]) for chain in FINGERS.values() for i in range(3)
]
PALM = (0, 1, 5, 9, 13, 17)

# Finger radius at the base, as a fraction of the palm reference width.
FINGER_RADIUS = {"thumb": 0.17, "index": 0.135, "middle": 0.14, "ring": 0.13, "pinky": 0.11}
TIP_TAPER = 0.85

# Adult bone lengths over the index-to-pinky knuckle span (about 5.8 cm), used
# to guess how far a bone is tilted out of the picture plane.
BONE_LENGTH = {
    (0, 1): 0.45, (1, 2): 0.78, (2, 3): 0.55, (3, 4): 0.43,
    (0, 5): 1.12, (5, 6): 0.69, (6, 7): 0.41, (7, 8): 0.31,
    (0, 9): 1.09, (9, 10): 0.76, (10, 11): 0.48, (11, 12): 0.33,
    (0, 13): 1.0, (13, 14): 0.71, (14, 15): 0.47, (15, 16): 0.33,
    (0, 17): 0.91, (17, 18): 0.57, (18, 19): 0.33, (19, 20): 0.29,
}

# COCO-17 limbs for the skeleton overlay.
BODY_LIMBS = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4),
]


def palm_reference(pts: np.ndarray) -> float:
    """A palm width that survives a hand seen side-on.

    The knuckle span (index MCP to pinky MCP) collapses when the hand turns
    edge-on; wrist to middle knuckle does not, and is about 1.1 spans long.
    The larger of the two is used."""
    p = np.asarray(pts, dtype=np.float64)[:, :2]
    span = float(np.linalg.norm(p[5] - p[17]))
    length = float(np.linalg.norm(p[9] - p[0])) / 1.1
    return max(span, length, 1.0)


def _finger_of(index: int) -> str | None:
    for name, chain in FINGERS.items():
        if index in chain:
            return name
    return None


def _bone_radii(pts: np.ndarray, scale: float = 1.0) -> list[tuple[int, int, float, float]]:
    ref = palm_reference(pts) * float(scale)
    out = []
    for a, b in HAND_BONES:
        finger = _finger_of(b)
        base = FINGER_RADIUS.get(finger, 0.13) * ref
        chain = FINGERS[finger]
        # Taper from the knuckle to the tip; the wrist end of a metacarpal is
        # covered by the palm polygon anyway.
        pos_a = chain.index(a) if a in chain else 0
        pos_b = chain.index(b)
        ra = base * (1.0 - (1.0 - TIP_TAPER) * pos_a / 3.0)
        rb = base * (1.0 - (1.0 - TIP_TAPER) * pos_b / 3.0)
        out.append((a, b, ra, rb))
    return out


def _window(shape, xs, ys, pad):
    h, w = shape
    x1 = max(0, int(np.floor(min(xs) - pad)))
    y1 = max(0, int(np.floor(min(ys) - pad)))
    x2 = min(w, int(np.ceil(max(xs) + pad)) + 1)
    y2 = min(h, int(np.ceil(max(ys) + pad)) + 1)
    return x1, y1, x2, y2


def capsule(shape, p0, p1, r0, r1):
    """(window, t, dist, radius) for one tapered capsule, or None if it is off
    canvas. `t` is the position along the bone (0 at p0), `dist` the distance
    from the bone axis, both over the window only."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    pad = max(r0, r1) + 1.0
    win = _window(shape, (x0, x1), (y0, y1), pad)
    wx1, wy1, wx2, wy2 = win
    if wx2 <= wx1 or wy2 <= wy1:
        return None
    ys = np.arange(wy1, wy2, dtype=np.float32)[:, None] + 0.5
    xs = np.arange(wx1, wx2, dtype=np.float32)[None, :] + 0.5
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    if length2 < 1e-6:
        t = np.zeros((wy2 - wy1, wx2 - wx1), dtype=np.float32)
    else:
        t = np.clip(((xs - x0) * dx + (ys - y0) * dy) / length2, 0.0, 1.0)
    px = x0 + t * dx
    py = y0 + t * dy
    dist = np.sqrt((xs - px) ** 2 + (ys - py) ** 2)
    radius = r0 + t * (r1 - r0)
    return win, t, dist, radius


def _hull(points: np.ndarray) -> np.ndarray:
    """Convex hull, counter-clockwise (monotone chain)."""
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 2:
        return np.asarray(pts, dtype=np.float64)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def convex_fill(shape, points) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    """Inside test for the convex hull of `points`, over its bounding window."""
    hull = _hull(np.asarray(points, dtype=np.float64)[:, :2])
    if len(hull) < 3:
        return None
    win = _window(shape, hull[:, 0], hull[:, 1], 1.0)
    x1, y1, x2, y2 = win
    if x2 <= x1 or y2 <= y1:
        return None
    ys = np.arange(y1, y2, dtype=np.float64)[:, None] + 0.5
    xs = np.arange(x1, x2, dtype=np.float64)[None, :] + 0.5
    inside = np.ones((y2 - y1, x2 - x1), dtype=bool)
    n = len(hull)
    # Which side is "inside" depends on the winding, and image y grows down.
    area = sum(hull[i, 0] * hull[(i + 1) % n, 1] - hull[(i + 1) % n, 0] * hull[i, 1] for i in range(n))
    sign = 1.0 if area >= 0 else -1.0
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        inside &= sign * ((bx - ax) * (ys - ay) - (by - ay) * (xs - ax)) >= 0
    return win, inside


def _palm_points(p: np.ndarray, ref: float) -> np.ndarray:
    """The palm outline: the knuckles, the thumb base, and the wrist widened
    sideways (a wrist is a joint, not a point, and a palm drawn to a point
    reads as a triangle)."""
    axis = p[9] - p[0]
    norm = float(np.linalg.norm(axis))
    pts = [p[i] for i in PALM]
    if norm > 1e-6:
        side = np.array([-axis[1], axis[0]]) / norm * (0.38 * ref)
        pts += [p[0] + side, p[0] - side]
    return np.asarray(pts, dtype=np.float64)


def render_hand_mask(
    pts: np.ndarray,
    shape: tuple[int, int],
    scale: float = 1.0,
    skip: tuple[str, ...] = (),
    palm: bool = True,
) -> np.ndarray:
    """Boolean (H, W) of the five-finger hand these landmarks imply.

    `scale` grows or shrinks every radius (0.6 gives the core of each finger),
    `skip` leaves fingers out by name."""
    out = np.zeros(shape, dtype=bool)
    p = np.asarray(pts, dtype=np.float64)[:, :2]
    for a, b, ra, rb in _bone_radii(pts, scale):
        if _finger_of(b) in skip:
            continue
        got = capsule(shape, p[a], p[b], ra, rb)
        if got is None:
            continue
        (x1, y1, x2, y2), _, dist, radius = got
        out[y1:y2, x1:x2] |= dist <= radius
    if palm:
        poly = convex_fill(shape, _palm_points(p, palm_reference(p) * scale))
        if poly is not None:
            (x1, y1, x2, y2), inside = poly
            out[y1:y2, x1:x2] |= inside
    return out


def render_finger_masks(pts: np.ndarray, shape: tuple[int, int], scale: float = 1.0) -> dict[str, np.ndarray]:
    """One boolean mask per finger, metacarpal excluded (the palm owns it),
    except for the thumb whose whole chain is visible."""
    p = np.asarray(pts, dtype=np.float64)[:, :2]
    out = {name: np.zeros(shape, dtype=bool) for name in FINGER_ORDER}
    for a, b, ra, rb in _bone_radii(pts, scale):
        name = _finger_of(b)
        if a == 0 and name != "thumb":
            continue
        got = capsule(shape, p[a], p[b], ra, rb)
        if got is None:
            continue
        (x1, y1, x2, y2), _, dist, radius = got
        out[name][y1:y2, x1:x2] |= dist <= radius
    return out


def estimate_hand_z(pts: np.ndarray) -> np.ndarray:
    """Depth per landmark (positive = away from the camera), from how short
    each bone looks next to its adult length.

    2D landmarks cannot say which way a bone tilts, only how much. Every
    shortened bone is taken as leaning away, half as far as the shortening
    allows: a gentle, consistent guess that keeps the depth map smooth rather
    than one that pretends to know."""
    p = np.asarray(pts, dtype=np.float64)[:, :2]
    span = max(float(np.linalg.norm(p[5] - p[17])), 1.0)
    # Same robust width as the radii, expressed in knuckle spans.
    unit = max(span, float(np.linalg.norm(p[9] - p[0])) / 1.09)
    z = np.zeros(21, dtype=np.float64)
    for a, b in HAND_BONES:
        expected = BONE_LENGTH[(a, b)] * unit
        seen = float(np.linalg.norm(p[b] - p[a]))
        dz = np.sqrt(max(0.0, expected * expected - seen * seen)) * 0.5
        z[b] = z[a] + dz
    return z


def _box_blur(img: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur with edge padding."""
    if radius <= 0:
        return img
    k = 2 * radius + 1
    out = img
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (radius + 1, radius)
        padded = np.pad(out, pad, mode="edge")
        csum = np.cumsum(padded, axis=axis, dtype=np.float32)
        if axis == 0:
            out = (csum[k:, :] - csum[:-k, :]) / k
        else:
            out = (csum[:, k:] - csum[:, :-k]) / k
    return out.astype(np.float32)


def render_hand_depth(
    hands: list[tuple[np.ndarray, np.ndarray | None]],
    shape: tuple[int, int],
    smooth: int | None = None,
) -> np.ndarray:
    """Float32 (H, W) depth map in 0..1, near = bright, background 0.

    `hands` is a list of (landmarks (21, 2+), z (21,) or None). z is positive
    away from the camera; None means estimate it from the bone lengths. Every
    hand is drawn with five fingers whatever the image shows: that is the shape
    ControlNet is meant to pull the repaint towards.

    Large maps are drawn at half size and scaled up: the result is smoothed
    anyway, and it keeps a 1024 px crop well under 20 ms."""
    h, w = shape
    factor = 2 if min(h, w) > 384 else 1
    small = (max(1, (h + factor - 1) // factor), max(1, (w + factor - 1) // factor))
    depth = np.zeros(small, dtype=np.float32)
    refs = []
    for pts, z in hands:
        full = np.asarray(pts, dtype=np.float64)[:, :2]
        zz = estimate_hand_z(full) if z is None else np.asarray(z, dtype=np.float64)
        span = float(np.ptp(zz)) if len(zz) else 0.0
        # Nearness 1 at the closest joint, 0.45 at the farthest.
        near = 1.0 - 0.55 * ((zz - zz.min()) / span if span > 1e-6 else np.zeros_like(zz))
        p = full / factor
        refs.append(palm_reference(p))
        for a, b, ra, rb in _bone_radii(p):
            got = capsule(small, p[a], p[b], ra, rb)
            if got is None:
                continue
            (x1, y1, x2, y2), t, dist, radius = got
            inside = dist <= radius
            if not inside.any():
                continue
            # A rounded cross-section: brightest on the bone axis.
            dome = np.sqrt(np.clip(1.0 - (dist / np.maximum(radius, 1e-3)) ** 2, 0.0, 1.0))
            value = (near[a] + t * (near[b] - near[a])) * (0.85 + 0.15 * dome)
            region = depth[y1:y2, x1:x2]
            np.maximum(region, np.where(inside, value, 0.0).astype(np.float32), out=region)
        poly = convex_fill(small, _palm_points(p, refs[-1]))
        if poly is not None:
            (x1, y1, x2, y2), inside = poly
            value = float(np.mean(near[list(PALM)])) * 0.92
            region = depth[y1:y2, x1:x2]
            np.maximum(region, np.where(inside, value, 0.0).astype(np.float32), out=region)
    if smooth is None:
        smooth = max(1, int(round(min(refs) * 0.04))) if refs else 1
    else:
        smooth = int(round(smooth / factor))
    if smooth > 0:
        depth = _blur_used(depth, smooth)
    if factor > 1:
        from PIL import Image

        u8 = Image.fromarray((np.clip(depth, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), mode="L")
        depth = np.asarray(u8.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    return np.clip(depth, 0.0, 1.0)


def _blur_used(depth: np.ndarray, radius: int) -> np.ndarray:
    """Blur only the window that holds something, plus the blur's reach."""
    rows = np.flatnonzero(depth.any(axis=1))
    cols = np.flatnonzero(depth.any(axis=0))
    if not len(rows):
        return depth
    reach = 2 * radius + 1
    h, w = depth.shape
    y1, y2 = max(0, rows[0] - reach), min(h, rows[-1] + reach + 1)
    x1, x2 = max(0, cols[0] - reach), min(w, cols[-1] + reach + 1)
    out = depth.copy()
    out[y1:y2, x1:x2] = _box_blur(_box_blur(depth[y1:y2, x1:x2], radius), radius)
    return out


def depth_to_image(depth: np.ndarray):
    """The depth map as an RGB PIL image, the form ControlNet expects."""
    from PIL import Image

    u8 = (np.clip(depth, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(np.stack([u8] * 3, axis=-1), mode="RGB")


# ---------------------------------------------------------------- overlay

KIND_COLOURS = {
    "hand_shape": (255, 72, 72),
    "extra_hand": (255, 150, 40),
    "missing_hand": (255, 150, 40),
    "fused_bodies": (200, 90, 255),
    "head_ratio": (255, 220, 60),
    "limb_asymmetry": (255, 220, 60),
}
BONE_COLOUR = (80, 220, 255)
HAND_COLOUR = (120, 255, 160)


def _plain(text: str) -> str:
    # The WebUI's font has no maths operators; a missing glyph draws as a box.
    return text.replace("≈", "~")


def draw_overlay(report, size: tuple[int, int], min_score: float = 0.3):
    """Transparent RGBA image of `size` with the skeletons, the hand capsules
    and the findings. `report.size` is the image the points are in; the
    overlay is scaled to `size` (the comparer's preview size)."""
    from PIL import Image, ImageDraw

    w, h = size
    src_w, src_h = report.size if report.size and report.size[0] else size
    sx, sy = w / float(max(1, src_w)), h / float(max(1, src_h))
    k = (sx + sy) / 2.0
    over = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(over, "RGBA")
    line = max(2, int(round(min(w, h) / 360.0)))

    def pt(p):
        return (float(p[0]) * sx, float(p[1]) * sy)

    for person in report.people:
        b = person.body
        for a, c in BODY_LIMBS:
            if b[a, 2] >= min_score and b[c, 2] >= min_score:
                draw.line([pt(b[a]), pt(b[c])], fill=BONE_COLOUR + (210,), width=line)
        for j in range(len(b)):
            if b[j, 2] >= min_score:
                x, y = pt(b[j])
                r = line + 1
                draw.ellipse([x - r, y - r, x + r, y + r], fill=BONE_COLOUR + (230,))
        for pts in person.hands:
            if pts is None or float(np.clip(pts[:, 2], 0, 1).mean()) < min_score:
                continue
            for a, c, ra, rc in _bone_radii(pts):
                width = max(line, int(round((ra + rc) * k)))
                draw.line([pt(pts[a]), pt(pts[c])], fill=HAND_COLOUR + (70,), width=width)
                draw.line([pt(pts[a]), pt(pts[c])], fill=HAND_COLOUR + (220,), width=max(1, line - 1))

    try:
        from ..preview import _font, _label, _metrics

        _, font_size = _metrics((w, h))
        font = _font(font_size)
    except Exception:
        font, _label = None, None
    placed: list = []
    for f in report.findings:
        colour = KIND_COLOURS.get(f.kind, (255, 255, 255))
        x1, y1, x2, y2 = f.bbox
        box = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        draw.rectangle(box, outline=colour + (255,), width=line + (1 if f.fixable else 0))
        text = _plain(f"{f.kind} p{f.person}" + (f" {f.side}" if f.side else "") + (f": {f.detail}" if f.detail else ""))
        if font is not None and _label is not None:
            _label(draw, (box[0], box[1]), text, font, colour, (w, h), placed)
        else:
            draw.text((box[0] + 2, box[1] + 2), text, fill=colour + (255,))
    if font is not None and _label is not None:
        _label(draw, (6, 6), f"Anatomy {int(round(report.score))}", font, (240, 240, 240), (w, h), placed)
    return over
