"""What the Inspector found, in plain data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

HAND_SHAPE = "hand_shape"
EXTRA_HAND = "extra_hand"
MISSING_HAND = "missing_hand"      # reserved, not produced in v1
FUSED_BODIES = "fused_bodies"
HEAD_RATIO = "head_ratio"
LIMB_ASYMMETRY = "limb_asymmetry"

KINDS = (HAND_SHAPE, EXTRA_HAND, MISSING_HAND, FUSED_BODIES, HEAD_RATIO, LIMB_ASYMMETRY)
# Only these may let a tab repaint a region in "Only flagged" mode.
FIXABLE = frozenset({HAND_SHAPE})

# How much one finding at full severity and confidence takes off the score.
WEIGHTS = {
    HAND_SHAPE: 35.0,
    EXTRA_HAND: 40.0,
    MISSING_HAND: 20.0,
    FUSED_BODIES: 40.0,
    HEAD_RATIO: 15.0,
    LIMB_ASYMMETRY: 10.0,
}

# COCO-17 body order, the one DWPose returns.
BODY_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
LEFT, RIGHT = 0, 1
SIDE_NAMES = ("left", "right")


@dataclass
class Person:
    """One skeleton, in image pixels. Arrays are (N, 3): x, y, score."""

    bbox: tuple[int, int, int, int]
    body: np.ndarray                       # (17, 3)
    feet: np.ndarray | None = None         # (6, 3) big toe, small toe, heel (left, then right)
    face: np.ndarray | None = None         # (68, 3)
    hands: list = field(default_factory=lambda: [None, None])   # [left (21,3), right (21,3)]
    score: float = 0.0
    # Optional per-hand z (21,), from MediaPipe when it ran on that hand.
    hand_z: list = field(default_factory=lambda: [None, None])

    def to_dict(self) -> dict:
        def arr(a):
            return None if a is None else np.round(np.asarray(a, dtype=np.float32), 2).tolist()

        return {
            "bbox": list(self.bbox),
            "body": arr(self.body),
            "feet": arr(self.feet),
            "face": arr(self.face),
            "hands": [arr(h) for h in self.hands],
            "score": round(float(self.score), 3),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Person":
        def arr(a):
            return None if a is None else np.asarray(a, dtype=np.float32)

        return cls(
            bbox=tuple(int(v) for v in data.get("bbox", (0, 0, 0, 0))),
            body=arr(data.get("body")) if data.get("body") is not None else np.zeros((17, 3), np.float32),
            feet=arr(data.get("feet")),
            face=arr(data.get("face")),
            hands=[arr(h) for h in (data.get("hands") or [None, None])],
            score=float(data.get("score", 0.0)),
        )


@dataclass
class Finding:
    kind: str           # one of KINDS
    person: int         # index into Report.people, -1 when not tied to one person
    bbox: tuple[int, int, int, int]
    severity: float     # 0..1, how wrong
    confidence: float   # 0..1, how sure the Inspector is
    detail: str         # short English text for console / overlay, e.g. "≈6 fingers"
    fixable: bool       # True only for kinds the gate may act on
    side: str = ""      # "left" / "right" for hand findings

    def label(self) -> str:
        who = f"p{self.person}" if self.person >= 0 else "?"
        side = f" {self.side}" if self.side else ""
        detail = f" {self.detail}" if self.detail else ""
        return f"{self.kind} {who}{side}{detail}"

    def to_dict(self) -> dict:
        out = asdict(self)
        out["bbox"] = list(self.bbox)
        out["severity"] = round(float(self.severity), 3)
        out["confidence"] = round(float(self.confidence), 3)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "Finding":
        return cls(
            kind=str(data["kind"]),
            person=int(data.get("person", -1)),
            bbox=tuple(int(v) for v in data.get("bbox", (0, 0, 0, 0))),
            severity=float(data.get("severity", 0.0)),
            confidence=float(data.get("confidence", 0.0)),
            detail=str(data.get("detail", "")),
            fixable=bool(data.get("fixable", False)),
            side=str(data.get("side", "")),
        )


def make(kind: str, person: int, bbox, severity: float, confidence: float,
         detail: str = "", side: str = "") -> Finding:
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    return Finding(
        kind=kind,
        person=int(person),
        bbox=(x1, y1, x2, y2),
        severity=float(np.clip(severity, 0.0, 1.0)),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        detail=detail,
        fixable=kind in FIXABLE,
        side=side,
    )


@dataclass
class Report:
    people: list[Person]
    findings: list[Finding]
    score: float = 100.0            # 0..100, 100 = nothing found
    profile: str = ""
    notes: list[str] = field(default_factory=list)   # hands not checked, skipped checks
    size: tuple[int, int] = (0, 0)  # image (w, h) the report is for
    ok: bool = True                 # False when the Inspector could not run

    def fixable(self) -> list[Finding]:
        return [f for f in self.findings if f.fixable]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.kind] = out.get(f.kind, 0) + 1
        return out

    def infotext(self) -> str:
        """`87 (hand_shape x1)`, or just `100`."""
        counts = self.counts()
        head = f"{int(round(self.score))}"
        if not counts:
            return head
        return head + " (" + ", ".join(f"{k} x{n}" for k, n in counts.items()) + ")"

    def console(self) -> str:
        n = len(self.findings)
        line = f"Inspector: score {int(round(self.score))}, {n} finding{'s' if n != 1 else ''}"
        if n:
            line += " (" + ", ".join(f.label() for f in self.findings) + ")"
        return line

    def to_dict(self) -> dict:
        return {
            "people": [p.to_dict() for p in self.people],
            "findings": [f.to_dict() for f in self.findings],
            "score": round(float(self.score), 1),
            "profile": self.profile,
            "notes": list(self.notes),
            "size": list(self.size),
            "ok": self.ok,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Report":
        return cls(
            people=[Person.from_dict(p) for p in data.get("people", [])],
            findings=[Finding.from_dict(f) for f in data.get("findings", [])],
            score=float(data.get("score", 100.0)),
            profile=str(data.get("profile", "")),
            notes=[str(n) for n in data.get("notes", [])],
            size=tuple(int(v) for v in data.get("size", (0, 0))),
            ok=bool(data.get("ok", True)),
        )


def score_of(findings: list[Finding]) -> float:
    lost = sum(WEIGHTS.get(f.kind, 10.0) * f.severity * f.confidence for f in findings)
    return float(np.clip(100.0 - lost, 0.0, 100.0))


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area = max(1e-6, (ax2 - ax1) * (ay2 - ay1)) + max(1e-6, (bx2 - bx1) * (by2 - by1)) - inter
    return float(inter / area)


def centre_inside(point_box, box) -> bool:
    cx = (point_box[0] + point_box[2]) / 2.0
    cy = (point_box[1] + point_box[3]) / 2.0
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def overlaps(det_box, finding_box, min_iou: float = 0.1) -> bool:
    """Tolerant match between a detection and a finding: some IoU, or either
    centre inside the other box. A hand detector's box and a box built from
    21 landmarks rarely agree on the edges, only on where the hand is."""
    return (
        iou(det_box, finding_box) >= min_iou
        or centre_inside(finding_box, det_box)
        or centre_inside(det_box, finding_box)
    )
