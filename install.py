"""WebUI startup hook. Installs ultralytics / opencv if missing."""

from __future__ import annotations

# (import name, pip spec, human name)
REQUIREMENTS = [
    ("ultralytics", "ultralytics>=8.2.0", "ultralytics (YOLO detectors)"),
    ("cv2", "opencv-python-headless>=4.8.0", "opencv (mask blur / dilate)"),
]


def _main() -> None:
    try:
        import launch
    except Exception:
        # Running outside the WebUI (a linter, a packaging step). Nothing to do.
        return

    for module, spec, label in REQUIREMENTS:
        try:
            if launch.is_installed(module):
                continue
        except Exception:
            continue
        if module == "cv2":
            # Most WebUI installs already ship opencv-python. Pulling in the
            # headless build next to it causes two cv2 copies to fight, so only
            # install it when cv2 genuinely is not importable.
            try:
                import cv2  # noqa: F401

                continue
            except Exception:
                pass
        try:
            launch.run_pip(f'install "{spec}"', f"Impact ADetailer: {label}")
        except Exception as exc:
            print(f"[Impact ADetailer] could not install {label}: {exc}")
            print(f"[Impact ADetailer] install it by hand: pip install \"{spec}\"")


_main()
