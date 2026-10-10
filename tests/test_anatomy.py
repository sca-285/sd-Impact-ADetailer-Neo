"""Anatomy Inspector unit tests. No models, no GPU, no WebUI.

    python -m pytest tests          (or: python -m unittest discover tests)
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from synthetic import extra_finger, jitter, open_hand  # noqa: E402

from iadetailer.anatomy import body, findings, hands, profiles, render  # noqa: E402
from iadetailer.anatomy.findings import Person, Report  # noqa: E402

try:
    import cv2  # noqa: F401

    HAVE_CV2 = True
except Exception:
    HAVE_CV2 = False

SHAPE = (400, 400)
REALISTIC = profiles.get(profiles.REALISTIC)


def _with_forearm(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    got = render.capsule(mask.shape, pts[0, :2], pts[0, :2] + np.array([0.0, 150.0]), 30.0, 34.0)
    (x1, y1, x2, y2), _, dist, radius = got
    out = mask.copy()
    out[y1:y2, x1:x2] |= dist <= radius
    return out


def _clean_mask(pts: np.ndarray) -> np.ndarray:
    # A real hand is never exactly the rendered one: slightly fatter fingers,
    # landmarks a pixel or two off, and the arm attached.
    return _with_forearm(render.render_hand_mask(jitter(pts, 1.5), SHAPE, scale=1.08), pts)


@unittest.skipUnless(HAVE_CV2, "opencv is needed for the contour opinion")
class HandChecks(unittest.TestCase):
    def setUp(self):
        self.pts = open_hand()

    def test_clean_hand_has_no_finding(self):
        check = hands.check_hand(self.pts, _clean_mask(self.pts), REALISTIC)
        self.assertTrue(check.checked)
        self.assertEqual(check.landmark_count, 5)
        self.assertEqual(check.contour_count, 5)
        self.assertEqual(check.severity, 0.0)
        self.assertIsNone(hands.to_finding(check, 0, "right", (0, 0, 10, 10), REALISTIC))

    def test_six_fingers(self):
        mask = _clean_mask(self.pts) | extra_finger(self.pts, SHAPE)
        check = hands.check_hand(self.pts, mask, REALISTIC)
        finding = hands.to_finding(check, 0, "right", (0, 0, 10, 10), REALISTIC)
        self.assertIsNotNone(finding)
        self.assertEqual(finding.kind, findings.HAND_SHAPE)
        self.assertTrue(finding.fixable)
        self.assertEqual(check.landmark_count, 6)
        self.assertEqual(finding.detail, "≈6 fingers")
        self.assertGreaterEqual(finding.severity, 0.7)

    def test_missing_finger(self):
        mask = _with_forearm(render.render_hand_mask(self.pts, SHAPE, scale=1.05, skip=("ring",)), self.pts)
        check = hands.check_hand(self.pts, mask, REALISTIC)
        finding = hands.to_finding(check, 0, "left", (0, 0, 10, 10), REALISTIC)
        self.assertIsNotNone(finding)
        self.assertIn("ring", check.missing_fingers)
        self.assertEqual(check.landmark_count, 4)
        self.assertEqual(finding.side, "left")

    def test_no_mask_is_not_checked(self):
        check = hands.check_hand(self.pts, None, REALISTIC)
        self.assertFalse(check.checked)
        self.assertIsNone(hands.to_finding(check, 0, "left", (0, 0, 1, 1), REALISTIC))

    def test_low_landmark_scores_are_not_usable(self):
        self.assertTrue(hands.usable(self.pts, REALISTIC))
        self.assertFalse(hands.usable(open_hand(score=0.1), REALISTIC))
        self.assertFalse(hands.usable(None, REALISTIC))

    def test_anime_profile_caps_confidence(self):
        anime = profiles.get(profiles.ANIME)
        mask = _clean_mask(self.pts) | extra_finger(self.pts, SHAPE)
        check = hands.check_hand(self.pts, mask, anime)
        self.assertLessEqual(check.confidence, 0.6)


def _standing_person(cx: float, top: float = 40.0, height: float = 500.0) -> Person:
    """An upright skeleton, about 7.5 heads tall, arms down."""
    u = height / 7.5            # one head
    b = np.zeros((17, 3), np.float32)
    b[:, 2] = 0.9
    pts = {
        0: (cx, top + 0.55 * u), 1: (cx + 0.15 * u, top + 0.45 * u), 2: (cx - 0.15 * u, top + 0.45 * u),
        3: (cx + 0.35 * u, top + 0.5 * u), 4: (cx - 0.35 * u, top + 0.5 * u),
        5: (cx + 0.9 * u, top + 1.5 * u), 6: (cx - 0.9 * u, top + 1.5 * u),
        7: (cx + 1.0 * u, top + 2.8 * u), 8: (cx - 1.0 * u, top + 2.8 * u),
        9: (cx + 1.05 * u, top + 3.9 * u), 10: (cx - 1.05 * u, top + 3.9 * u),
        11: (cx + 0.5 * u, top + 3.8 * u), 12: (cx - 0.5 * u, top + 3.8 * u),
        13: (cx + 0.5 * u, top + 5.6 * u), 14: (cx - 0.5 * u, top + 5.6 * u),
        15: (cx + 0.5 * u, top + 7.3 * u), 16: (cx - 0.5 * u, top + 7.3 * u),
    }
    for k, (x, y) in pts.items():
        b[k, :2] = (x, y)
    face = np.zeros((68, 3), np.float32)
    face[:, 2] = 0.9
    face[:, :2] = (cx, top + 0.5 * u)
    face[36:48, 1] = top + 0.45 * u
    face[8, 1] = top + 0.95 * u
    bbox = (int(cx - 1.4 * u), int(top), int(cx + 1.4 * u), int(top + 7.5 * u))
    return Person(bbox=bbox, body=b, face=face, score=0.9)


class BodyChecks(unittest.TestCase):
    def test_third_hand_box_is_extra_hand(self):
        person = _standing_person(300.0)
        b = person.body
        boxes = [
            ((int(b[9, 0] - 20), int(b[9, 1] - 5), int(b[9, 0] + 20), int(b[9, 1] + 45)), 0.9),
            ((int(b[10, 0] - 20), int(b[10, 1] - 5), int(b[10, 0] + 20), int(b[10, 1] + 45)), 0.9),
            # a hand on the chest that no wrist reaches
            ((280, 200, 320, 240), 0.85),
        ]
        out = body.check_extra_hands([person], boxes, REALISTIC)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].kind, findings.EXTRA_HAND)
        self.assertFalse(out[0].fixable)
        self.assertEqual(out[0].bbox, (280, 200, 320, 240))

    def test_two_hand_boxes_are_fine(self):
        person = _standing_person(300.0)
        b = person.body
        boxes = [
            ((int(b[9, 0] - 20), int(b[9, 1] - 5), int(b[9, 0] + 20), int(b[9, 1] + 45)), 0.9),
            ((int(b[10, 0] - 20), int(b[10, 1] - 5), int(b[10, 0] + 20), int(b[10, 1] + 45)), 0.9),
        ]
        self.assertEqual(body.check_extra_hands([person], boxes, REALISTIC), [])

    def test_head_ratio_in_range(self):
        person = _standing_person(300.0)
        self.assertEqual(body.check_head_ratio([person], (600, 700), REALISTIC), [])

    def test_head_ratio_big_head(self):
        person = _standing_person(300.0, top=100.0)
        person.face[8, 1] += 30          # a chin far below the eyes: a huge head
        out = body.check_head_ratio([person], (600, 700), REALISTIC)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].kind, findings.HEAD_RATIO)
        # the stylised profile does not check it at all
        self.assertEqual(body.check_head_ratio([person], (600, 700), profiles.get(profiles.STYLISED)), [])
        self.assertLess(out[0].confidence, 0.6)

    def test_head_ratio_skipped_when_cut_off(self):
        person = _standing_person(300.0, top=40.0, height=700.0)
        self.assertEqual(body.check_head_ratio([person], (600, 700), REALISTIC), [])

    def test_limb_asymmetry(self):
        person = _standing_person(300.0)
        self.assertEqual(body.check_limb_asymmetry([person], REALISTIC), [])
        person.body[10, 1] += 70         # right forearm much longer, same direction
        out = body.check_limb_asymmetry([person], REALISTIC)
        self.assertTrue(any(f.kind == findings.LIMB_ASYMMETRY and "forearm" in f.detail for f in out))

    def test_fused_bodies_share_a_limb(self):
        a = _standing_person(300.0)
        b = _standing_person(360.0)
        # b's left arm lies exactly on a's left arm
        b.body[[5, 7, 9], :2] = a.body[[5, 7, 9], :2]
        out = body.check_fused([a, b], [], REALISTIC)
        self.assertTrue(any(f.kind == findings.FUSED_BODIES for f in out))

    def test_duplicate_skeleton_is_not_fused(self):
        a = _standing_person(300.0)
        b = _standing_person(301.0)
        self.assertEqual(body.check_fused([a, b], [], REALISTIC), [])


class Render(unittest.TestCase):
    def test_depth_shape_range_and_determinism(self):
        pts = open_hand(cx=500, cy=700, size=200)
        a = render.render_hand_depth([(pts, None)], (1024, 768))
        b = render.render_hand_depth([(pts, None)], (1024, 768))
        self.assertEqual(a.shape, (1024, 768))
        self.assertEqual(a.dtype, np.float32)
        self.assertGreaterEqual(float(a.min()), 0.0)
        self.assertLessEqual(float(a.max()), 1.0)
        self.assertGreater(float(a.max()), 0.8)
        self.assertEqual(float(a[0, 0]), 0.0)              # background is far
        self.assertTrue(np.array_equal(a, b))
        image = render.depth_to_image(a)
        self.assertEqual(image.size, (768, 1024))
        self.assertEqual(image.mode, "RGB")

    def test_depth_speed(self):
        pts = open_hand(cx=512, cy=600, size=180)
        render.render_hand_depth([(pts, None)], (1024, 1024))       # warm up
        times = []
        for _ in range(5):
            t = time.perf_counter()
            render.render_hand_depth([(pts, None)], (1024, 1024))
            times.append(time.perf_counter() - t)
        best = min(times) * 1000.0
        print(f"\nhand depth 1024x1024: {best:.1f} ms")
        # The target is 20 ms; the bound leaves room for a slow CI machine.
        self.assertLess(best, 60.0)

    def test_five_fingers_even_when_z_given(self):
        pts = open_hand()
        z = np.linspace(0, 20, 21)
        depth = render.render_hand_depth([(pts, z)], SHAPE, smooth=0)
        mask = render.render_hand_mask(pts, SHAPE)
        # Everything drawn is inside the five-finger silhouette (give or take
        # the edge pixels the half-size render and the upscale soften).
        spill = (depth > 0.05) & ~render.render_hand_mask(pts, SHAPE, scale=1.2)
        self.assertLess(float(spill.sum()) / float(mask.sum()), 0.01)
        self.assertGreater(float((depth > 0).sum()) / float(mask.sum()), 0.6)

    def test_estimated_z_is_finite(self):
        z = render.estimate_hand_z(open_hand())
        self.assertEqual(z.shape, (21,))
        self.assertTrue(np.isfinite(z).all())
        self.assertEqual(float(z[0]), 0.0)

    def test_overlay_size(self):
        pts = open_hand()
        person = Person(bbox=(0, 0, 400, 400), body=np.zeros((17, 3), np.float32), hands=[pts, None])
        f = findings.make(findings.HAND_SHAPE, 0, (150, 120, 260, 260), 0.8, 0.9, "≈6 fingers", "right")
        report = Report(people=[person], findings=[f], score=72.0, size=(400, 400))
        over = render.draw_overlay(report, (200, 200))
        self.assertEqual(over.size, (200, 200))
        self.assertEqual(over.mode, "RGBA")
        alpha = np.asarray(over)[:, :, 3]
        self.assertGreater(int((alpha > 0).sum()), 100)
        self.assertEqual(int(alpha[199, 0]), 0)


class ReportData(unittest.TestCase):
    def test_score_and_infotext(self):
        f = findings.make(findings.HAND_SHAPE, 0, (0, 0, 10, 10), 1.0, 1.0, "≈6 fingers", "left")
        r = Report(people=[], findings=[f], score=findings.score_of([f]))
        self.assertAlmostEqual(r.score, 65.0)
        self.assertEqual(r.infotext(), "65 (hand_shape x1)")
        self.assertIn("hand_shape p0 left ≈6 fingers", r.console())
        self.assertEqual(Report(people=[], findings=[]).infotext(), "100")

    def test_round_trip(self):
        pts = open_hand()
        person = Person(bbox=(1, 2, 3, 4), body=np.ones((17, 3), np.float32), hands=[pts, None], score=0.5)
        f = findings.make(findings.EXTRA_HAND, 0, (5, 6, 7, 8), 0.8, 0.4, "x")
        r = Report(people=[person], findings=[f], score=70.0, profile="Anime", size=(64, 32))
        back = Report.from_dict(r.to_dict())
        self.assertEqual(back.findings[0], f)
        self.assertEqual(back.profile, "Anime")
        self.assertEqual(back.size, (64, 32))
        self.assertTrue(np.allclose(back.people[0].hands[0], pts, atol=0.01))
        self.assertIsNone(back.people[0].hands[1])

    def test_overlap_is_tolerant(self):
        self.assertTrue(findings.overlaps((100, 100, 200, 200), (150, 150, 260, 260)))
        # centre of a small finding inside a big detection
        self.assertTrue(findings.overlaps((0, 0, 400, 400), (180, 180, 200, 200)))
        self.assertFalse(findings.overlaps((0, 0, 50, 50), (300, 300, 350, 350)))

    def test_profiles(self):
        self.assertEqual(profiles.get("nonsense")["name"], profiles.REALISTIC)
        self.assertIsNone(profiles.get(profiles.STYLISED)["head_ratio"])
        self.assertEqual(profiles.get(profiles.ANIME)["hand_confidence_cap"], 0.6)


class Gate(unittest.TestCase):
    def test_only_flagged_keeps_flagged_boxes(self):
        from iadetailer.detect import Detection
        from iadetailer.pipeline import _flag_gate

        good = Detection(bbox=(10, 10, 60, 60), score=0.9, mask=None, label="hand")
        bad = Detection(bbox=(200, 200, 260, 260), score=0.9, mask=None, label="hand")
        warn = findings.make(findings.HEAD_RATIO, 0, (0, 0, 70, 70), 0.9, 0.9, "")
        flag = findings.make(findings.HAND_SHAPE, 0, (205, 210, 255, 250), 0.8, 0.9, "≈6 fingers", "left")
        report = Report(people=[], findings=[warn, flag], score=50.0)
        rejected: list = []
        kept = _flag_gate([good, bad], report, rejected)
        self.assertEqual(kept, [bad])
        self.assertEqual(rejected, [(good, "not flagged")])

    def test_only_flagged_without_inspector_keeps_nothing(self):
        from iadetailer.detect import Detection
        from iadetailer.pipeline import _flag_gate

        det = Detection(bbox=(10, 10, 60, 60), score=0.9, mask=None, label="hand")
        rejected: list = []
        self.assertEqual(_flag_gate([det], None, rejected), [])
        self.assertEqual(rejected[0][1], "Inspector off")

    def test_hand_depth_guide_in_crop(self):
        from iadetailer.pipeline import _hand_depth_guide

        pts = open_hand(cx=200, cy=230, size=60)
        person = Person(bbox=(0, 0, 400, 400), body=np.zeros((17, 3), np.float32), hands=[None, pts])
        report = Report(people=[person], findings=[], size=(400, 400))
        image = _hand_depth_guide(report, (100, 80, 300, 280), (512, 512), 0)
        self.assertIsNotNone(image)
        self.assertEqual(image.size, (512, 512))
        self.assertGreater(int(np.asarray(image).max()), 200)
        # a crop with no hand in it gets no guide
        self.assertIsNone(_hand_depth_guide(report, (300, 300, 400, 400), (256, 256), 0))
        self.assertIsNone(_hand_depth_guide(None, (100, 80, 300, 280), (512, 512), 0))


class PoseDecode(unittest.TestCase):
    def test_simcc_maps_back_to_image(self):
        from iadetailer.anatomy import pose

        box = np.array([[100.0, 50.0, 300.0, 450.0]], dtype=np.float32)
        target = np.array([180.0, 260.0])
        center, scale, mat = pose._warp(box[0])
        # where the target lands on the 288x384 input, in SimCC bins (x2)
        px = mat[0, 0] * target[0] + mat[0, 2]
        py = mat[1, 1] * target[1] + mat[1, 2]

        def fake_net(batch):
            n = batch.shape[0]
            sx = np.zeros((n, 133, 576), np.float32)
            sy = np.zeros((n, 133, 768), np.float32)
            sx[:, :, int(round(px * 2))] = 0.9
            sy[:, :, int(round(py * 2))] = 0.8
            return [sx, sy]

        try:
            import cv2  # noqa: F401
        except Exception:
            self.skipTest("opencv needed")
        rgb = np.zeros((500, 400, 3), np.uint8)
        kps, scores = pose.estimate_pose(fake_net, rgb, box)
        self.assertEqual(kps.shape, (1, 133, 2))
        self.assertTrue(np.allclose(kps[0, 0], target, atol=1.5))
        self.assertAlmostEqual(float(scores[0, 0]), 0.8, places=5)
        people = pose.to_people(kps, scores, box, np.array([0.9]))
        self.assertEqual(people[0].hands[0].shape, (21, 3))
        self.assertEqual(people[0].face.shape, (68, 3))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_CV2, "opencv needed")
class InspectFlow(unittest.TestCase):
    """inspect() end to end, with stand-ins for the two onnx graphs and the
    hand-seg detector. Checks the plumbing, not the networks."""

    def _fake_nets(self, size, box, keypoints):
        from iadetailer.anatomy import pose

        w, h = size

        def det(blob):
            out = np.full((1, 8400, 85), -20.0, np.float32)
            ratio = min(640.0 / h, 640.0 / w)
            cx, cy = (box[0] + box[2]) / 2 * ratio, (box[1] + box[3]) / 2 * ratio
            bw, bh = (box[2] - box[0]) * ratio, (box[3] - box[1]) * ratio
            out[0, 0, :4] = (cx / 8.0, cy / 8.0, np.log(bw / 8.0), np.log(bh / 8.0))
            out[0, :, 4:6] = 0.0
            out[0, 0, 4:6] = 1.0
            return [out]

        def pose_net(batch):
            n = batch.shape[0]
            sx = np.zeros((n, 133, 576), np.float32)
            sy = np.zeros((n, 133, 768), np.float32)
            _, _, mat = pose._warp(np.asarray(box, np.float32))
            for k, (x, y, s) in enumerate(keypoints):
                px = mat[0, 0] * x + mat[0, 2]
                py = mat[1, 1] * y + mat[1, 2]
                sx[:, k, int(np.clip(round(px * 2), 0, 575))] = s
                sy[:, k, int(np.clip(round(py * 2), 0, 767))] = s
            return [sx, sy]

        return det, pose_net

    def test_six_finger_hand_found_and_scored(self):
        from PIL import Image

        import iadetailer.anatomy.inspect  # noqa: F401  (load the module)
        from iadetailer.anatomy import models
        from iadetailer.detect import Detection

        mod = sys.modules["iadetailer.anatomy.inspect"]
        size = (400, 400)
        hand = open_hand()
        kps = np.zeros((133, 3), np.float32)
        kps[:, :2] = (200, 200)
        kps[:, 2] = 0.05                       # body and face unseen
        kps[112:133] = hand                    # right hand
        box = (60.0, 20.0, 340.0, 390.0)
        det, pose_net = self._fake_nets(size, box, kps)

        mask = _clean_mask(hand) | extra_finger(hand, SHAPE)
        seg = [Detection(bbox=(130, 90, 280, 300), score=0.9, mask=mask.astype(np.float32), label="hand")]

        saved = (models.sessions, models.pick_detector, models.pick_sam, mod._run)
        try:
            models.sessions = lambda: (det, pose_net)
            models.pick_detector = lambda kind: "hand_seg.pt" if kind in ("hand_seg", "hand") else None
            models.pick_sam = lambda: None
            mod._run = lambda image, name, conf, cache: seg if name else []
            image = Image.new("RGB", size, (90, 90, 90))
            report = mod.inspect(image, "Realistic", cache=object())
        finally:
            models.sessions, models.pick_detector, models.pick_sam, mod._run = saved

        self.assertTrue(report.ok)
        self.assertEqual(len(report.people), 1)
        self.assertTrue(np.allclose(report.people[0].hands[1][:, :2], hand[:, :2], atol=2.0))
        kinds = [f.kind for f in report.findings]
        self.assertIn(findings.HAND_SHAPE, kinds)
        flagged = report.fixable()[0]
        self.assertEqual(flagged.side, "right")
        self.assertTrue(findings.overlaps(seg[0].bbox, flagged.bbox))
        self.assertLess(report.score, 100.0)

    def test_missing_models_is_not_ok(self):
        from PIL import Image

        from iadetailer.anatomy import models

        mod = sys.modules["iadetailer.anatomy.inspect"]
        saved = models.sessions

        def boom():
            raise FileNotFoundError("DWPose model missing: yolox_l.onnx")

        try:
            models.sessions = boom
            report = mod.inspect(Image.new("RGB", (64, 64)), "Anime")
        finally:
            models.sessions = saved
        self.assertFalse(report.ok)
        self.assertIn("yolox_l.onnx", report.notes[0])
