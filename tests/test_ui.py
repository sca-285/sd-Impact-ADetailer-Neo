"""UI layout and infotext round trip. Needs gradio (3.41 or 4.x); skipped without it."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

try:
    import gradio as gr
except Exception:  # pragma: no cover
    gr = None


@unittest.skipIf(gr is None, "gradio not installed")
class Layout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from iadetailer import uistate

        # Keep the remembered-state file out of the repository.
        uistate._PATH = Path(tempfile.mkdtemp()) / "impact-adetailer-ui.json"
        from iadetailer import ui

        cls.ui = ui
        with gr.Blocks():
            cls.widgets = ui.build_ui(False)

    def test_widget_count(self):
        ui = self.ui
        from iadetailer.settings import max_tabs

        expected = len(ui.HEAD_KEYS) + len(ui.FIELD_KEYS) * max_tabs() + len(ui.INSPECTOR_KEYS)
        self.assertEqual(len(self.widgets), expected)

    def test_repaint_not_an_engine_field(self):
        self.assertNotIn("repaint", self.ui.ENGINE_FIELDS)

    def _values(self):
        return [getattr(w, "value", None) for w in self.widgets]

    def test_parse_defaults(self):
        run, units = self.ui.parse_args(tuple(self._values()))
        inspector = self.ui.parse_inspector(tuple(self._values()))
        self.assertFalse(inspector.enable)
        self.assertEqual(inspector.profile, "Realistic")
        self.assertTrue(inspector.overlay)
        self.assertEqual(units[0].repaint, "All detections")
        self.assertEqual(units[0].cn_end, 1.0)

    def test_old_layout_still_parses(self):
        ui = self.ui
        values = self._values()
        per = len(ui.FIELD_KEYS)
        head = len(ui.HEAD_KEYS)
        # strip the Inspector tail and the per-tab Repaint: a call from before both
        units_raw = values[head : len(values) - len(ui.INSPECTOR_KEYS)]
        old = list(values[:head])
        for i in range(0, len(units_raw), per):
            old += units_raw[i : i + per - 1]
        run, units = ui.parse_args(tuple(old))
        self.assertEqual(units[0].model, values[head + 1])
        self.assertEqual(units[0].repaint, "All detections")
        self.assertFalse(ui.parse_inspector(tuple(old)).enable)

    def test_infotext_round_trip(self):
        ui = self.ui
        values = self._values()
        head = len(ui.HEAD_KEYS)
        per = len(ui.FIELD_KEYS)
        values[0] = True
        values[head + 0] = True                       # tab 1 on
        values[head + 1] = "hand_yolov8s.pt"
        values[head + per - 1] = "Only flagged by Inspector"
        values[-4] = True                             # inspector on
        values[-3] = "Anime"
        run, units = ui.parse_args(tuple(values))
        inspector = ui.parse_inspector(tuple(values))
        params = ui.infotext_params(units, inspector)
        self.assertEqual(params["Impact ADetailer inspector"], True)
        self.assertEqual(params["Impact ADetailer inspector profile"], "Anime")
        self.assertEqual(params["Impact ADetailer 1st repaint"], "Only flagged by Inspector")

        # Paste: every field the infotext names gets its value back.
        pairs = ui.infotext_fields(self.widgets)
        index = {id(w): i for i, w in enumerate(self.widgets)}
        pasted = self._values()
        for widget, key in pairs:
            if key in params:
                pasted[index[id(widget)]] = params[key]
        run2, units2 = ui.parse_args(tuple(pasted))
        inspector2 = ui.parse_inspector(tuple(pasted))
        self.assertTrue(inspector2.enable)
        self.assertEqual(inspector2.profile, "Anime")
        self.assertEqual(units2[0].repaint, "Only flagged by Inspector")
        self.assertEqual(units2[0].model, "hand_yolov8s.pt")

    def test_repaint_default_stays_out_without_inspector(self):
        ui = self.ui
        values = self._values()
        head = len(ui.HEAD_KEYS)
        values[head + 0] = True
        values[head + 1] = "face_yolov8n.pt"
        run, units = ui.parse_args(tuple(values))
        params = ui.infotext_params(units, ui.parse_inspector(tuple(values)))
        self.assertNotIn("Impact ADetailer 1st repaint", params)
        self.assertNotIn("Impact ADetailer inspector", params)

    def test_uistate_round_trip(self):
        from iadetailer import uistate
        from iadetailer.args import InspectorArgs, RunArgs, UnitArgs

        uistate.save("txt2img", RunArgs(enable=True), [UnitArgs(repaint="Only flagged by Inspector")],
                     InspectorArgs(enable=True, profile="Stylised", overlay=False))
        data = uistate.load("txt2img")
        self.assertEqual(data["inspector"]["profile"], "Stylised")
        self.assertFalse(data["inspector"]["overlay"])
        self.assertEqual(data["units"][0]["repaint"], "Only flagged by Inspector")


if __name__ == "__main__":
    unittest.main()
