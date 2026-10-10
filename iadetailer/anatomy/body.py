"""Per-person checks. All warnings: a hand repaint cannot fix any of them.

A skeleton always has two arms and one head by construction, so what it cannot
explain is what gives a body away: a hand box no wrist reaches, a face with no
skeleton under it, limbs two skeletons share.

2D points cannot tell a foreshortened bone from a short one, so the length
checks only run where foreshortening is unlikely (upright, both sides at the
same angle), and never claim much confidence.
"""

from __future__ import annotations

import numpy as np

from .findings import (
    EXTRA_HAND,
    FUSED_BODIES,
    HEAD_RATIO,
    LIMB_ASYMMETRY,
    Finding,
    Person,
    centre_inside,
    iou,
    make,
)

L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16

LIMB_PAIRS = (
    ("upper arm", (L_SHO, L_ELB), (R_SHO, R_ELB)),
    ("forearm", (L_ELB, L_WRI), (R_ELB, R_WRI)),
    ("thigh", (L_HIP, L_KNE), (R_HIP, R_KNE)),
    ("shin", (L_KNE, L_ANK), (R_KNE, R_ANK)),
)

# Pairs of joints that make a limb, for "two skeletons share a limb".
SHARED_LIMBS = ((L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB), (R_ELB, R_WRI),
                (L_HIP, L_KNE), (L_KNE, L_ANK), (R_HIP, R_KNE), (R_KNE, R_ANK))


def _seen(person: Person, i: int, thr: float) -> bool:
    return float(person.body[i, 2]) >= thr


def _height(person: Person) -> float:
    x1, y1, x2, y2 = person.bbox
    return float(max(1, y2 - y1))


def _box_distance(point, box) -> float:
    x, y = point
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return float(np.hypot(dx, dy))


# -------------------------------------------------------------- extra_hand


def check_extra_hands(people: list[Person], hand_boxes: list[tuple[tuple, float]], profile: dict) -> list[Finding]:
    """More than two hands inside one person that no wrist of anybody explains.

    A hand box is explained when a wrist (any person's) sits in it or close to
    it. Boxes inside a person that nobody's wrist reaches are the extra ones,
    reported only when that person then has more than two hands."""
    thr = float(profile["point_min_score"])
    boxes = [(tuple(b), float(s)) for b, s in hand_boxes if float(s) >= float(profile["hand_box_min"])]
    if not boxes or not people:
        return []

    def explained(box) -> bool:
        diag = float(np.hypot(box[2] - box[0], box[3] - box[1]))
        for person in people:
            for w in (L_WRI, R_WRI):
                if _seen(person, w, thr) and _box_distance(person.body[w, :2], box) <= 0.6 * diag:
                    return True
        return False

    findings = []
    for pi, person in enumerate(people):
        inside = [(b, s) for b, s in boxes if centre_inside(b, person.bbox)]
        # A box inside two overlapping people belongs to the smaller one.
        inside = [
            (b, s) for b, s in inside
            if not any(
                qi != pi and centre_inside(b, q.bbox)
                and (q.bbox[2] - q.bbox[0]) * (q.bbox[3] - q.bbox[1])
                < (person.bbox[2] - person.bbox[0]) * (person.bbox[3] - person.bbox[1])
                for qi, q in enumerate(people)
            )
        ]
        if len(inside) <= 2:
            continue
        stray = [(b, s) for b, s in inside if not explained(b)]
        for b, s in stray:
            confidence = min(0.9, s) * min(1.0, person.score + 0.3)
            findings.append(make(EXTRA_HAND, pi, b, 0.8, confidence, f"{len(inside)} hands, no wrist reaches this one"))
    return findings


# ------------------------------------------------------------ fused_bodies


def check_fused(people: list[Person], face_boxes: list[tuple[tuple, float]], profile: dict) -> list[Finding]:
    thr = float(profile["point_min_score"])
    findings: list[Finding] = []
    reported: set[tuple[int, int]] = set()

    # Two skeletons that share a whole limb but are otherwise different people.
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            a, b = people[i], people[j]
            if iou(a.bbox, b.bbox) < 0.2:
                continue
            near = 0.03 * max(_height(a), _height(b))
            both = [k for k in range(17) if _seen(a, k, thr) and _seen(b, k, thr)]
            if len(both) < 4:
                continue
            close = {k for k in both if np.linalg.norm(a.body[k, :2] - b.body[k, :2]) <= near}
            # Nearly every point shared = the same person found twice, not two fused.
            if len(close) >= 0.6 * len(both):
                continue
            shared = [limb for limb in SHARED_LIMBS if limb[0] in close and limb[1] in close]
            if not shared:
                continue
            box = (min(a.bbox[0], b.bbox[0]), min(a.bbox[1], b.bbox[1]),
                   max(a.bbox[2], b.bbox[2]), max(a.bbox[3], b.bbox[3]))
            findings.append(make(FUSED_BODIES, i, box, 0.7, 0.55, f"p{i} and p{j} share a limb"))
            reported.add((i, j))

    # More faces than skeletons with a head, inside a cluster of overlapping people.
    faces = [(tuple(f), s) for f, s in face_boxes if s >= 0.4]
    if faces and people:
        clusters = _clusters(people)
        for members in clusters:
            box = (min(people[k].bbox[0] for k in members), min(people[k].bbox[1] for k in members),
                   max(people[k].bbox[2] for k in members), max(people[k].bbox[3] for k in members))
            heads = sum(1 for k in members if _seen(people[k], 0, thr) or _seen(people[k], 1, thr) or _seen(people[k], 2, thr))
            inside = [f for f, _ in faces if centre_inside(f, box)]
            if len(inside) > heads and heads >= 1:
                pair = (min(members), max(members))
                if pair in reported:
                    continue
                findings.append(make(FUSED_BODIES, min(members), box, 0.6, 0.45,
                                     f"{len(inside)} faces, {heads} skeleton{'s' if heads != 1 else ''}"))
    return findings


def _clusters(people: list[Person]) -> list[list[int]]:
    parent = list(range(len(people)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            if iou(people[i].bbox, people[j].bbox) >= 0.1:
                parent[root(i)] = root(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(people)):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


# --------------------------------------------------------------- head_ratio


def head_measure(person: Person, size: tuple[int, int], thr: float) -> tuple[float, float, float] | None:
    """(head top y, head height, feet y) when the person is upright and whole
    in frame, else None."""
    w, h = size
    need = (L_SHO, R_SHO, L_HIP, R_HIP, L_ANK, R_ANK)
    if not all(_seen(person, k, thr) for k in need):
        return None
    if person.face is None:
        return None
    face = person.face
    # 68-point face: 8 = chin, 36-47 = eyes.
    eyes = face[36:48]
    eyes = eyes[eyes[:, 2] >= thr]
    if float(face[8, 2]) < thr or len(eyes) < 4:
        return None
    b = person.body
    shoulders = (b[L_SHO, :2] + b[R_SHO, :2]) / 2.0
    hips = (b[L_HIP, :2] + b[R_HIP, :2]) / 2.0
    torso = hips - shoulders
    if torso[1] <= 0:
        return None
    if np.degrees(np.arctan2(abs(torso[0]), torso[1])) > 20.0:
        return None                     # leaning or lying: not upright
    eye_y = float(eyes[:, 1].mean())
    chin_y = float(face[8, 1])
    if chin_y <= eye_y:
        return None
    head_h = 2.0 * (chin_y - eye_y)     # eyes sit about halfway down the head
    top = eye_y - (chin_y - eye_y)
    feet_pts = [b[L_ANK, 1], b[R_ANK, 1]]
    if person.feet is not None:
        feet_pts += [float(p[1]) for p in person.feet if p[2] >= thr]
    feet = float(max(feet_pts))
    # Whole in frame: head top and feet inside, with a margin.
    if top < 2 or feet > h - 2:
        return None
    return top, head_h, feet


def check_head_ratio(people: list[Person], size: tuple[int, int], profile: dict) -> list[Finding]:
    span = profile.get("head_ratio")
    if not span:
        return []
    lo, hi = span
    thr = float(profile["point_min_score"])
    findings = []
    for pi, person in enumerate(people):
        got = head_measure(person, size, thr)
        if got is None:
            continue
        top, head_h, feet = got
        ratio = (feet - top) / max(1.0, head_h)
        if lo <= ratio <= hi:
            continue
        off = (lo - ratio) / lo if ratio < lo else (ratio - hi) / hi
        severity = float(np.clip(0.3 + off * 2.0, 0.0, 1.0))
        x1, _, x2, _ = person.bbox
        findings.append(make(HEAD_RATIO, pi, (x1, top, x2, feet), severity, 0.5,
                             f"{ratio:.1f} heads tall (expected {lo:g}-{hi:g})"))
    return findings


# ---------------------------------------------------------- limb_asymmetry


def _angle(v) -> float:
    return float(np.degrees(np.arctan2(v[1], v[0])))


def check_limb_asymmetry(people: list[Person], profile: dict) -> list[Finding]:
    thr = max(0.4, float(profile["point_min_score"]))
    max_angle = float(profile["asym_angle"])
    max_ratio = float(profile["asym_ratio"])
    findings = []
    for pi, person in enumerate(people):
        b = person.body
        for name, (la, lb), (ra, rb) in LIMB_PAIRS:
            if not all(_seen(person, k, thr) for k in (la, lb, ra, rb)):
                continue
            lv = b[lb, :2] - b[la, :2]
            rv = b[rb, :2] - b[ra, :2]
            ll, rl = float(np.linalg.norm(lv)), float(np.linalg.norm(rv))
            if min(ll, rl) < 0.03 * _height(person):
                continue
            # Mirror the right limb so a symmetric pose reads as the same angle.
            diff = abs(_angle(lv) - _angle((-rv[0], rv[1])))
            diff = min(diff, 360.0 - diff)
            if diff > max_angle:
                continue
            ratio = max(ll, rl) / max(1e-6, min(ll, rl))
            if ratio <= max_ratio:
                continue
            longer = "left" if ll > rl else "right"
            pts = b[[la, lb, ra, rb], :2]
            box = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
            severity = float(np.clip((ratio - max_ratio) / max_ratio + 0.3, 0.0, 1.0))
            findings.append(make(LIMB_ASYMMETRY, pi, box, severity, 0.4,
                                 f"{longer} {name} {ratio:.2f}x the other"))
    return findings


def run(people: list[Person], size: tuple[int, int], hand_boxes, face_boxes, profile: dict) -> list[Finding]:
    out: list[Finding] = []
    out += check_extra_hands(people, hand_boxes, profile)
    out += check_fused(people, face_boxes, profile)
    out += check_head_ratio(people, size, profile)
    out += check_limb_asymmetry(people, profile)
    return [f for f in out if f.confidence >= float(profile["min_confidence"])]
