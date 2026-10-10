"""Style profiles: every threshold the Inspector uses, in one table.

Tune numbers here, not in the checks. A profile only changes how strict the
checks are; it never changes what they look at.
"""

from __future__ import annotations

REALISTIC = "Realistic"
ANIME = "Anime"
STYLISED = "Stylised"
NAMES = [REALISTIC, ANIME, STYLISED]

_BASE = {
    # --- hands ---------------------------------------------------------
    # A hand is checked only when its landmarks are this sure of themselves.
    "hand_min_score": 0.35,
    # Points under this score count as "not seen".
    "point_min_score": 0.3,
    # Fraction of the expected hand area that may be unexplained hand pixels
    # (an extra finger, a blob) before it is a finding.
    "unexplained_max": 0.22,
    # Fraction of the expected hand area that may have no hand pixels under it.
    "missing_max": 0.28,
    # A finger whose capsule is this empty counts as missing.
    "finger_missing": 0.6,
    # Slack around the rendered hand, as a fraction of the palm width, so
    # landmark jitter does not read as an extra finger.
    "tolerance": 0.18,
    # Highest confidence a hand finding may claim.
    "hand_confidence_cap": 1.0,
    # Findings under this confidence are not reported.
    "min_confidence": 0.3,
    # --- body ----------------------------------------------------------
    "head_ratio": (6.8, 8.2),
    # Two limbs are compared only when their angles differ less than this.
    "asym_angle": 15.0,
    # Longer / shorter above this is a finding.
    "asym_ratio": 1.35,
    # Hand boxes from a detector below this confidence are ignored.
    "hand_box_min": 0.45,
    "prefer_mediapipe": False,
}

PROFILES: dict[str, dict] = {
    REALISTIC: {**_BASE, "prefer_mediapipe": True},
    ANIME: {
        **_BASE,
        "hand_min_score": 0.3,
        "unexplained_max": 0.32,
        "missing_max": 0.4,
        "finger_missing": 0.75,
        "tolerance": 0.25,
        "hand_confidence_cap": 0.6,
        "head_ratio": (6.0, 8.5),
        "asym_ratio": 1.5,
    },
    STYLISED: {
        **_BASE,
        "hand_min_score": 0.3,
        "unexplained_max": 0.36,
        "missing_max": 0.45,
        "finger_missing": 0.8,
        "tolerance": 0.28,
        "hand_confidence_cap": 0.6,
        "head_ratio": None,          # not checked
        "asym_ratio": 1.6,
    },
}


def get(name: str | None) -> dict:
    """The profile by name, Realistic when the name is unknown."""
    key = name if name in PROFILES else REALISTIC
    return {**PROFILES[key], "name": key}
