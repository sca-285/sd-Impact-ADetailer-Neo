# Contributing

## Setup

1. Clone into `extensions/sd-Impact-ADetailer-Neo` of a Forge, ReForge or Forge Neo install.
2. Restart the WebUI so `install.py` can pull `ultralytics` if needed.
3. Do not commit `__pycache__` or `impact-adetailer-ui.json`.

## Rules

* Keep comments short. Describe what the code does, not how it got there.
* `UnitArgs` in `iadetailer/args.py` is the source of factory defaults. UI `val()` must read from it.
* Widget order in `ui.py` must match `FIELD_KEYS` and `parse_args`. New per-tab fields go at the end of `FIELD_KEYS`; the Inspector controls are passed after the tabs (`INSPECTOR_KEYS`).
* Inspector thresholds live in `iadetailer/anatomy/profiles.py` only.
* Inner img2img jobs use an empty script runner plus `is_iadetailer_job`. Do not attach other alwayson scripts there except ControlNet.
* Conditioning cache on the inner job must be `[None, None]`, never `None` or `[]`.

## Tests you can run.

* `python -m pytest tests` (no WebUI, no GPU)

* txt2img + one face detector, preview = Detection
* same with **Use different checkpoint** on, confirm the main model is restored
* ControlNet Passthrough on ReForge and Neo
* YOLO-World / YOLOE with **Classes** filled in
