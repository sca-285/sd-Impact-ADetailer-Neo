"""Per-hand check: the hand pixels against the hand the landmarks imply.

A landmark model always returns 21 points, so it never sees a sixth finger. The
pixels do. Rendering the five-finger hand those 21 points describe and laying
it over the real hand mask leaves two kinds of disagreement:

  unexplained  hand pixels no finger accounts for (an extra finger, a blob,
               fingers fused into a paddle);
  missing      parts of the rendered hand with no hand pixels under them (a
               finger the model never drew).

The mask's own outline gives a second, independent opinion: fingertips stick
out of it. A finger count is only claimed when both opinions agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import render
from .findings import HAND_SHAPE, Finding, make

try:
    import cv2
except Exception:  # pragma: no cover - the WebUI always ships opencv
    cv2 = None


@dataclass
class HandCheck:
    checked: bool
    reason: str = ""                 # why not checked, when it was not
    unexplained: float = 0.0         # fraction of the expected hand area
    missing: float = 0.0
    missing_fingers: list[str] = field(default_factory=list)
    extra_blobs: int = 0
    landmark_count: int = 5          # fingers, as the landmark comparison sees it
    contour_count: int | None = None # fingers, as the outline sees it
    severity: float = 0.0
    confidence: float = 0.0
    detail: str = ""

    @property
    def agreed(self) -> bool:
        return self.contour_count is not None and self.contour_count == self.landmark_count


def hand_box(pts: np.ndarray, pad: float = 0.35, min_score: float = 0.0) -> tuple[int, int, int, int]:
    """Box around the 21 points, padded by `pad` of its larger side."""
    p = np.asarray(pts, dtype=np.float64)
    seen = p[p[:, 2] >= min_score] if p.shape[1] > 2 and min_score > 0 else p
    if len(seen) < 2:
        seen = p
    x1, y1 = seen[:, 0].min(), seen[:, 1].min()
    x2, y2 = seen[:, 0].max(), seen[:, 1].max()
    side = max(x2 - x1, y2 - y1, render.palm_reference(p) * 1.5)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = side * (0.5 + pad)
    return int(np.floor(cx - half)), int(np.floor(cy - half)), int(np.ceil(cx + half)), int(np.ceil(cy + half))


def usable(pts: np.ndarray | None, profile: dict) -> bool:
    """Enough landmark confidence to judge this hand at all."""
    if pts is None:
        return False
    p = np.asarray(pts, dtype=np.float64)
    if p.shape[0] != 21:
        return False
    scores = np.clip(p[:, 2], 0.0, 1.0) if p.shape[1] > 2 else np.ones(21)
    if float(scores.mean()) < float(profile["hand_min_score"]):
        return False
    return int((scores >= float(profile["point_min_score"])).sum()) >= 15


def _components(mask: np.ndarray) -> tuple[int, np.ndarray]:
    if cv2 is not None:
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return int(n), labels
    return _label_numpy(mask)


def _label_numpy(mask: np.ndarray) -> tuple[int, np.ndarray]:
    """4-connected labelling without opencv (slow, only a fallback)."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    h, w = mask.shape
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or labels[y, x]:
                continue
            current += 1
            stack = [(y, x)]
            labels[y, x] = current
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
    return current + 1, labels


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    if cv2 is not None:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return cv2.dilate(mask.astype(np.uint8), k) > 0
    out = mask.copy()
    for _ in range(radius):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _forearm(shape, pts: np.ndarray, ref: float) -> np.ndarray:
    """Pixels on the arm side of the wrist. Neither check counts them: a hand
    mask usually runs on up the arm, and nothing there is a finger."""
    h, w = shape
    p = np.asarray(pts, dtype=np.float64)[:, :2]
    axis = p[9] - p[0]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return np.zeros(shape, dtype=bool)
    u = axis / norm
    ys = np.arange(h, dtype=np.float64)[:, None] + 0.5
    xs = np.arange(w, dtype=np.float64)[None, :] + 0.5
    along = (xs - p[0, 0]) * u[0] + (ys - p[0, 1]) * u[1]
    return along < -0.15 * ref


def count_protrusions(mask: np.ndarray, pts: np.ndarray, ref: float, exclude: np.ndarray | None = None) -> int | None:
    """Fingertips on the mask outline: peaks of the distance from the palm
    centre along the contour. None when there is no usable outline."""
    if cv2 is None:
        return None
    m = mask.copy()
    if exclude is not None:
        m &= ~exclude
    if m.sum() < 16:
        return None
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    n = len(contour)
    if n < 24:
        return None
    p = np.asarray(pts, dtype=np.float64)[:, :2]
    centre = p[[0, 5, 9, 13, 17]].mean(axis=0)
    radial = np.linalg.norm(contour - centre, axis=1)
    win = max(3, n // 60)
    kernel = np.ones(2 * win + 1) / (2 * win + 1)
    padded = np.concatenate([radial[-win:], radial, radial[:win]])
    smooth = np.convolve(padded, kernel, mode="valid")

    reach = max(4, n // 10)
    peaks: list[np.ndarray] = []
    for i in range(n):
        r = smooth[i]
        if r < 0.6 * ref:
            continue
        lo, hi = i - reach, i + reach
        idx = np.arange(lo, hi + 1) % n
        seg = smooth[idx]
        if r < seg.max() - 1e-9:
            continue
        left = smooth[np.arange(lo, i) % n].min()
        right = smooth[np.arange(i + 1, hi + 1) % n].min()
        if r - max(left, right) < 0.22 * ref:
            continue
        pt = contour[i]
        if any(np.linalg.norm(pt - q) < 0.25 * ref for q in peaks):
            continue
        peaks.append(pt)
    return len(peaks)


def check_hand(
    pts: np.ndarray,
    mask: np.ndarray | None,
    profile: dict,
    mask_quality: float = 1.0,
) -> HandCheck:
    """Compare one hand (landmarks in mask coordinates) with its mask.

    `mask` is a boolean array covering the hand box, in the same coordinates
    as `pts`. `mask_quality` (0..1) is how much the mask source can be trusted,
    e.g. a SAM score."""
    if mask is None:
        return HandCheck(checked=False, reason="no hand mask")
    p = np.asarray(pts, dtype=np.float64)
    shape = mask.shape[:2]
    mask = np.asarray(mask) > 0.5 if mask.dtype != bool else mask
    ref = render.palm_reference(p)

    expected = render.render_hand_mask(p, shape)
    area = float(expected.sum())
    if area < 32:
        return HandCheck(checked=False, reason="hand too small")
    tol = max(1, int(round(float(profile["tolerance"]) * ref)))
    expected_tol = _dilate(expected, tol)
    arm = _forearm(shape, p, ref)

    # Keep only the parts of the mask that touch the hand: a mask from a
    # whole-image segmenter also holds the other hand, and the other person's.
    n, labels = _components(mask)
    if n > 2:
        touching = np.unique(labels[mask & expected_tol])
        touching = touching[touching > 0]
        mask = np.isin(labels, touching)
    if float(mask.sum()) < 0.15 * area:
        return HandCheck(checked=False, reason="mask does not cover the hand")

    # --- unexplained: blobs of hand with no finger to explain them ------
    stray = mask & ~expected_tol & ~arm
    finger_area = np.pi * (render.FINGER_RADIUS["middle"] * ref) ** 2 * 4.0
    blobs = 0
    stray_area = 0.0
    if stray.any():
        sn, slabels = _components(stray)
        sizes = np.bincount(slabels.ravel(), minlength=sn)[1:]
        significant = sizes[sizes >= 0.12 * finger_area]
        stray_area = float(significant.sum())
        # Finger-sized ones count as candidate extra fingers.
        blobs = int((sizes >= 0.35 * finger_area).sum())
    unexplained = stray_area / area

    # --- missing: rendered hand with no pixels under it -----------------
    core = render.render_hand_mask(p, shape, scale=0.6) & ~arm
    core_area = max(1.0, float(core.sum()))
    missing = float((core & ~mask).sum()) / core_area
    missing_fingers = []
    for name, finger in render.render_finger_masks(p, shape, scale=0.6).items():
        f_area = float(finger.sum())
        if f_area < 8:
            continue
        if float((finger & ~mask).sum()) / f_area >= float(profile["finger_missing"]):
            missing_fingers.append(name)

    landmark_count = 5 + blobs - len(missing_fingers)
    contour_count = count_protrusions(mask, p, ref, exclude=arm)

    out = HandCheck(
        checked=True,
        unexplained=unexplained,
        missing=missing,
        missing_fingers=missing_fingers,
        extra_blobs=blobs,
        landmark_count=landmark_count,
        contour_count=contour_count,
    )

    excess = max(unexplained / float(profile["unexplained_max"]), missing / float(profile["missing_max"]))
    if landmark_count != 5:
        excess = max(excess, 1.0 + 0.5 * abs(landmark_count - 5))
    if excess < 1.0:
        out.confidence = _confidence(p, mask, area, mask_quality, profile, out.agreed)
        return out

    severity = 0.35 + 0.35 * (excess - 1.0)
    if out.agreed and landmark_count != 5:
        severity = max(severity, 0.75)
    out.severity = float(np.clip(severity, 0.0, 1.0))
    out.confidence = _confidence(p, mask, area, mask_quality, profile, out.agreed and landmark_count != 5)
    out.detail = _detail(out)
    return out


def _confidence(p, mask, area, mask_quality, profile, agreed) -> float:
    scores = np.clip(p[:, 2], 0.0, 1.0) if p.shape[1] > 2 else np.ones(21)
    landmarks = float(scores.mean())
    spill = float(mask.sum()) / max(1.0, area)
    # A mask three times the size of the hand is probably the whole arm, or a
    # sleeve: it says less about the fingers.
    quality = float(np.clip(mask_quality, 0.0, 1.0)) * (0.5 if spill > 3.0 else 1.0)
    conf = landmarks * (0.6 + 0.4 * quality)
    if agreed:
        conf += 0.15
    return float(np.clip(conf, 0.0, float(profile["hand_confidence_cap"])))


def _detail(c: HandCheck) -> str:
    if c.agreed and c.landmark_count != 5:
        return f"≈{c.landmark_count} fingers"
    if c.extra_blobs:
        return f"extra finger-like blob, {int(round(c.unexplained * 100))}% unexplained"
    if c.missing_fingers:
        return "missing " + ", ".join(c.missing_fingers)
    if c.unexplained >= c.missing:
        return f"{int(round(c.unexplained * 100))}% unexplained"
    return f"{int(round(c.missing * 100))}% missing"


def to_finding(check: HandCheck, person: int, side: str, bbox, profile: dict) -> Finding | None:
    """The check as a finding, or None when there is nothing to report."""
    if not check.checked or check.severity <= 0.0:
        return None
    if check.confidence < float(profile["min_confidence"]):
        return None
    return make(HAND_SHAPE, person, bbox, check.severity, check.confidence, check.detail, side)
