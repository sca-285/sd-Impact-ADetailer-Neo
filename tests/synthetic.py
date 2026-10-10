"""Synthetic hands and bodies for the Inspector tests. No models, no GPU."""

from __future__ import annotations

import numpy as np


def open_hand(cx: float = 200.0, cy: float = 230.0, size: float = 60.0, score: float = 0.9) -> np.ndarray:
    """21 landmarks of an open right hand, fingers up, palm to the camera.
    `size` is about the knuckle span in pixels."""
    s = size / 58.0
    wrist = np.array([cx, cy])
    pts = [wrist]
    # thumb: out to the side and up
    thumb_dir = np.array([-0.8, -0.6])
    thumb_dir /= np.linalg.norm(thumb_dir)
    base = wrist + np.array([-14, -12]) * s
    pts.append(base)
    for length in (30, 22, 18):
        base = base + thumb_dir * length * s
        pts.append(base)
    knuckles_x = (-22, -7, 8, 22)
    lengths = ((40, 24, 18), (44, 28, 19), (41, 27, 19), (33, 19, 17))
    spread = (-0.12, -0.03, 0.05, 0.14)
    for kx, ls, ang in zip(knuckles_x, lengths, spread):
        k = wrist + np.array([kx * 1.0 + 4, -62]) * s
        pts.append(k)
        d = np.array([np.sin(ang), -np.cos(ang)])
        cur = k
        for length in ls:
            cur = cur + d * length * s
            pts.append(cur)
    arr = np.array(pts, dtype=np.float64)
    return np.concatenate([arr, np.full((21, 1), score)], axis=1)


def extra_finger(pts: np.ndarray, shape) -> np.ndarray:
    """A mask capsule for a sixth finger beside the pinky."""
    from iadetailer.anatomy import render

    p = pts[:, :2]
    ref = render.palm_reference(pts)
    out = np.zeros(shape, dtype=bool)
    base = p[17] + (p[17] - p[13]) * 0.95
    d = (p[20] - p[17])
    d = d / np.linalg.norm(d)
    tip = base + (d + np.array([0.35, 0.0])) / np.linalg.norm(d + np.array([0.35, 0.0])) * ref * 1.05
    got = render.capsule(shape, base, tip, ref * 0.11, ref * 0.095)
    (x1, y1, x2, y2), _, dist, radius = got
    out[y1:y2, x1:x2] |= dist <= radius
    # join it to the palm
    got = render.capsule(shape, p[17], base, ref * 0.12, ref * 0.11)
    (x1, y1, x2, y2), _, dist, radius = got
    out[y1:y2, x1:x2] |= dist <= radius
    return out


def jitter(pts: np.ndarray, amount: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = pts.copy()
    out[:, :2] += rng.normal(0.0, amount, size=(21, 2))
    return out
