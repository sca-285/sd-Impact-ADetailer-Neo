"""inspect(image, profile) -> Report. The Inspector's one public entry point."""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

from . import body, hands, models, pose, profiles
from .findings import SIDE_NAMES, Finding, Report, iou, score_of

# Landmarks handed to SAM as positive points: wrist, knuckles and fingertips.
SAM_POINTS = (0, 5, 9, 13, 17, 4, 8, 12, 16, 20)


def _interrupted() -> bool:
    try:
        from modules import shared

        st = getattr(shared, "state", None)
        return bool(st is not None and (getattr(st, "interrupted", False) or getattr(st, "skipped", False)))
    except Exception:
        return False


def _clip_box(box, size) -> tuple[int, int, int, int]:
    w, h = size
    x1, y1, x2, y2 = box
    return max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))


def _run(image, name, confidence, cache):
    if not name:
        return []
    try:
        from ..detect import run_detector

        return run_detector(image, name, confidence, cache=cache)
    except Exception as exc:
        print(f"[Impact ADetailer] Inspector: {name} failed: {exc}")
        return []


def _refine_with_mediapipe(image: Image.Image, people, profile: dict) -> int:
    """Swap DWPose hand points for MediaPipe's where both find the same hand.
    MediaPipe is sharper on realistic fingers and gives a z per point."""
    solver = models.mediapipe_hands()
    if solver is None:
        return 0
    rgb = np.asarray(image.convert("RGB"))
    refined = 0
    for person in people:
        for si, pts in enumerate(person.hands):
            if not hands.usable(pts, profile):
                continue
            x1, y1, x2, y2 = _clip_box(hands.hand_box(pts, pad=0.6), image.size)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            try:
                res = solver.process(np.ascontiguousarray(rgb[y1:y2, x1:x2]))
            except Exception:
                continue
            if not getattr(res, "multi_hand_landmarks", None):
                continue
            cw, ch = x2 - x1, y2 - y1
            lm = res.multi_hand_landmarks[0].landmark
            xy = np.array([[p.x * cw + x1, p.y * ch + y1] for p in lm], dtype=np.float32)
            from .render import palm_reference

            if float(np.linalg.norm(xy - pts[:, :2], axis=1).mean()) > 0.5 * palm_reference(pts):
                continue                # a different hand in the same crop
            try:
                conf = float(res.multi_handedness[0].classification[0].score)
            except Exception:
                conf = 0.8
            person.hands[si] = np.concatenate([xy, np.full((21, 1), conf, np.float32)], axis=1)
            person.hand_z[si] = np.array([p.z * cw for p in lm], dtype=np.float32)
            refined += 1
    return refined


def _mask_from_seg(dets, box) -> tuple[np.ndarray, float] | None:
    best, best_iou = None, 0.0
    for det in dets:
        if det.mask is None:
            continue
        overlap = iou(det.bbox, box)
        if overlap > best_iou:
            best, best_iou = det, overlap
    if best is None or best_iou < 0.1:
        return None
    return best.mask, float(best.score)


def inspect(image: Image.Image, profile: str = profiles.REALISTIC, cache=None) -> Report:
    """Find what is anatomically wrong in `image`.

    Never raises for a missing model: the report comes back with ok=False and
    the reason in `notes`."""
    prof = profiles.get(profile)
    image = image.convert("RGB")
    size = image.size
    report = Report(people=[], findings=[], profile=prof["name"], size=size)
    started = time.perf_counter()
    try:
        det_net, pose_net = models.sessions()
    except Exception as exc:
        report.ok = False
        report.notes.append(str(exc))
        return report

    people = pose.run(image, det_net, pose_net)
    report.people = people
    if _interrupted():
        report.notes.append("interrupted")
        return report
    if prof.get("prefer_mediapipe"):
        _refine_with_mediapipe(image, people, prof)

    if cache is None:
        from ..detect import DetectorCache

        cache = DetectorCache()
    seg_name = models.pick_detector("hand_seg")
    hand_name = models.pick_detector("hand")
    face_name = models.pick_detector("face")
    seg_dets = _run(image, seg_name, 0.25, cache)
    hand_dets = seg_dets if hand_name and hand_name == seg_name else _run(image, hand_name, 0.3, cache)
    face_dets = _run(image, face_name, 0.4, cache)
    sam_name = None if seg_name else models.pick_sam()

    findings: list[Finding] = []
    for pi, person in enumerate(people):
        for si, pts in enumerate(person.hands):
            if _interrupted():
                report.notes.append("interrupted")
                break
            if not hands.usable(pts, prof):
                continue
            side = SIDE_NAMES[si]
            box = _clip_box(hands.hand_box(pts), size)
            x1, y1, x2, y2 = box
            if x2 - x1 < 12 or y2 - y1 < 12:
                continue
            full_mask, quality = None, 0.0
            got = _mask_from_seg(seg_dets, box)
            if got is not None:
                full_mask, quality = got
            elif sam_name:
                use = [i for i in SAM_POINTS if pts[i, 2] >= prof["point_min_score"]]
                from ..sam import segment_with_points

                full_mask = segment_with_points(image, box, pts[use, :2], sam_name)
                quality = 0.8
            if full_mask is None:
                report.notes.append(f"p{pi} {side}: not checked (no hand mask)")
                continue
            crop = np.asarray(full_mask[y1:y2, x1:x2]) > 0.5
            local = pts.copy()
            local[:, 0] -= x1
            local[:, 1] -= y1
            check = hands.check_hand(local, crop, prof, quality)
            if not check.checked:
                report.notes.append(f"p{pi} {side}: not checked ({check.reason})")
                continue
            tight = _clip_box(hands.hand_box(pts, pad=0.1), size)
            finding = hands.to_finding(check, pi, side, tight, prof)
            if finding is not None:
                findings.append(finding)

    hand_boxes = [(d.bbox, d.score) for d in hand_dets]
    face_boxes = [(d.bbox, d.score) for d in face_dets]
    findings += body.run(people, size, hand_boxes, face_boxes, prof)
    findings.sort(key=lambda f: f.severity * f.confidence, reverse=True)
    report.findings = findings
    report.score = score_of(findings)
    try:
        from ..settings import log

        log(f"Inspector: {len(people)} people, {time.perf_counter() - started:.2f}s")
    except Exception:
        pass
    return report


def hands_in_crop(report: Report, crop_region, profile: str | None = None):
    """Every usable hand whose landmarks lie mostly inside `crop_region`, as
    (person, side, landmarks, z) with landmarks in image pixels."""
    prof = profiles.get(profile or report.profile)
    x1, y1, x2, y2 = crop_region
    out = []
    for pi, person in enumerate(report.people):
        for si, pts in enumerate(person.hands):
            if not hands.usable(pts, prof):
                continue
            inside = (pts[:, 0] >= x1) & (pts[:, 0] <= x2) & (pts[:, 1] >= y1) & (pts[:, 1] <= y2)
            if float(inside.mean()) < 0.7:
                continue
            z = person.hand_z[si] if si < len(person.hand_z) else None
            out.append((pi, SIDE_NAMES[si], pts, z))
    return out
