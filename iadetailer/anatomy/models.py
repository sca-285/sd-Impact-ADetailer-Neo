"""Find and load what the Inspector runs on.

DWPose is two onnx files. Forge, reForge and Forge Classic Neo all keep them in
the same place when their ControlNet DWPose preprocessor has run once:

    <preprocessor_dir>/openpose/yolox_l.onnx
    <preprocessor_dir>/openpose/dw-ll_ucoco_384.onnx

where preprocessor_dir is modules_forge.shared.preprocessor_dir, i.e.
--controlnet-preprocessor-models-dir or models/ControlNetPreprocessor
(checked in the three trees: forge_legacy_preprocessors/annotator/
annotator_path.py and openpose/__init__.py). The Inspector reuses those copies
and only falls back to models/iadetailer/anatomy.

onnxruntime is used when it imports (GPU when CUDA is there); otherwise the
models run through cv2.dnn on the CPU, exactly as the WebUI's own DWPose
preprocessor does.
"""

from __future__ import annotations

import threading
from pathlib import Path

DET_FILE = "yolox_l.onnx"
POSE_FILE = "dw-ll_ucoco_384.onnx"
FILES = (DET_FILE, POSE_FILE)
URLS = {
    DET_FILE: "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx",
    POSE_FILE: "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx",
}
README_URL = "https://github.com/sca-285/sd-Impact-ADetailer-Neo#anatomy-inspector"

_LOCK = threading.Lock()
_SESSIONS: dict[str, object] = {}
_MP: dict[str, object] = {}
_MP_WARNED = [False]


# ------------------------------------------------------------------ paths


def anatomy_dir() -> Path:
    from ..paths import _webui_root, official_models_path

    base = official_models_path() or (_webui_root() / "models")
    return Path(base) / "iadetailer" / "anatomy"


def _preprocessor_dirs() -> list[Path]:
    dirs: list[Path] = []
    try:
        from modules_forge.shared import preprocessor_dir

        if preprocessor_dir:
            dirs.append(Path(str(preprocessor_dir)))
    except Exception:
        pass
    try:
        from modules import shared

        custom = getattr(shared.cmd_opts, "controlnet_preprocessor_models_dir", None)
        if custom:
            dirs.append(Path(str(custom)))
    except Exception:
        pass
    try:
        from ..paths import _webui_root, official_models_path

        base = official_models_path() or (_webui_root() / "models")
        dirs.append(Path(base) / "ControlNetPreprocessor")
        # A1111's sd-webui-controlnet, if someone runs this there.
        dirs.append(_webui_root() / "extensions" / "sd-webui-controlnet" / "annotator" / "downloads")
    except Exception:
        pass
    return dirs


def search_dirs() -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for base in _preprocessor_dirs():
        for d in (base / "openpose", base):
            key = str(d).lower()
            if key not in seen:
                seen.add(key)
                out.append(d)
    own = anatomy_dir()
    if str(own).lower() not in seen:
        out.append(own)
    return out


def find() -> dict[str, Path | None]:
    found: dict[str, Path | None] = {name: None for name in FILES}
    for d in search_dirs():
        for name in FILES:
            if found[name] is None:
                candidate = d / name
                try:
                    if candidate.is_file() and candidate.stat().st_size > 1024:
                        found[name] = candidate
                except OSError:
                    continue
    return found


def available() -> bool:
    return all(find().values())


# --------------------------------------------------------------- backends


class Net:
    """One onnx model behind onnxruntime or cv2.dnn, with the same call."""

    def __init__(self, path: Path):
        self.path = path
        self.backend = ""
        self._ort = None
        self._dnn = None
        try:
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if _want_gpu() and "CUDAExecutionProvider" in ort.get_available_providers():
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            try:
                self._ort = ort.InferenceSession(str(path), providers=providers)
            except Exception:
                self._ort = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            used = self._ort.get_providers()
            self.backend = "onnxruntime " + ("CUDA" if "CUDAExecutionProvider" in used else "CPU")
            self._input = self._ort.get_inputs()[0].name
            return
        except Exception:
            self._ort = None
        import cv2

        self._dnn = cv2.dnn.readNetFromONNX(str(path))
        self._dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self._dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.backend = "cv2.dnn CPU"

    def __call__(self, batch):
        import numpy as np

        batch = np.ascontiguousarray(batch, dtype=np.float32)
        if self._ort is not None:
            return self._ort.run(None, {self._input: batch})
        # cv2.dnn runs one image at a time with these graphs.
        outs = []
        for i in range(batch.shape[0]):
            self._dnn.setInput(batch[i : i + 1])
            outs.append(self._dnn.forward(self._dnn.getUnconnectedOutLayersNames()))
        return [np.concatenate([o[k] for o in outs], axis=0) for k in range(len(outs[0]))]


def _want_gpu() -> bool:
    try:
        from ..settings import detector_device

        dev = detector_device()
    except Exception:
        dev = "Automatic"
    if dev == "cpu":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return dev == "cuda"


def sessions() -> tuple[Net, Net]:
    """(person detector, whole-body pose), loaded once."""
    with _LOCK:
        if "det" in _SESSIONS and "pose" in _SESSIONS:
            return _SESSIONS["det"], _SESSIONS["pose"]
        found = find()
        missing = [name for name, path in found.items() if path is None]
        if missing:
            raise FileNotFoundError("DWPose model missing: " + ", ".join(missing))
        _SESSIONS["det"] = Net(found[DET_FILE])
        _SESSIONS["pose"] = Net(found[POSE_FILE])
        from ..settings import log

        log(f"Inspector: DWPose loaded from {found[POSE_FILE].parent} ({_SESSIONS['pose'].backend})")
        return _SESSIONS["det"], _SESSIONS["pose"]


def backend() -> str:
    net = _SESSIONS.get("pose")
    if net is not None:
        return net.backend
    try:
        import onnxruntime as ort

        gpu = "CUDAExecutionProvider" in ort.get_available_providers() and _want_gpu()
        return "onnxruntime " + ("CUDA" if gpu else "CPU")
    except Exception:
        return "cv2.dnn CPU (onnxruntime not installed)"


def mediapipe_hands():
    """A MediaPipe Hands solver, or None. One console line the first time it
    cannot be had; wheels do not exist for every Python."""
    if "hands" in _MP:
        return _MP["hands"]
    solver = None
    try:
        import mediapipe as mp

        solver = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1, min_detection_confidence=0.4
        )
    except Exception as exc:
        if not _MP_WARNED[0]:
            _MP_WARNED[0] = True
            print(f"[Impact ADetailer] Inspector: MediaPipe not available ({type(exc).__name__}), using DWPose hand points")
    _MP["hands"] = solver
    return solver


def mediapipe_installed() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("mediapipe") is not None
    except Exception:
        return False


def free() -> None:
    """Drop the onnx sessions and MediaPipe. Called with IAD's other caches."""
    with _LOCK:
        had = bool(_SESSIONS) or bool(_MP.get("hands"))
        _SESSIONS.clear()
        solver = _MP.pop("hands", None)
        try:
            if solver is not None:
                solver.close()
        except Exception:
            pass
    if had:
        import gc

        gc.collect()


# ------------------------------------------------------ helper detectors


def _setting(name: str) -> str:
    try:
        from ..settings import get_opt

        return str(get_opt(name, "") or "").strip()
    except Exception:
        return ""


def _detectors() -> list[str]:
    try:
        from ..detect import detector_index

        return sorted(detector_index().keys())
    except Exception:
        return []


def pick_detector(kind: str) -> str | None:
    """A detector from the user's own models for `kind`: "hand_seg", "hand"
    or "face". A name in Settings wins; "None" there turns it off."""
    key = {"hand_seg": "iad_inspector_hand_model", "hand": "iad_inspector_hand_model",
           "face": "iad_inspector_face_model"}[kind]
    chosen = _setting(key)
    names = _detectors()
    if chosen:
        if chosen.lower() == "none":
            return None
        if chosen in names:
            if kind == "hand_seg" and "seg" not in chosen.lower():
                return None
            return chosen
    word = "face" if kind == "face" else "hand"
    hits = [n for n in names if word in n.lower()]
    if kind == "hand_seg":
        hits = [n for n in hits if "seg" in n.lower()]
    elif kind == "face":
        # A box model is enough to count faces and cheaper than a seg one.
        hits.sort(key=lambda n: ("seg" in n.lower(), n))
    else:
        hits.sort(key=lambda n: ("seg" not in n.lower(), n))
    return hits[0] if hits else None


def pick_sam() -> str | None:
    chosen = _setting("iad_inspector_sam_model")
    if chosen.lower() == "none":
        return None
    try:
        from ..sam import list_sam_models

        names = [n for n in list_sam_models() if n not in ("", "None")]
    except Exception:
        names = []
    if chosen and chosen in names:
        return chosen
    return names[0] if names else None


# ------------------------------------------------------------- status / setup


def status_markdown() -> str:
    found = find()
    lines = []
    if all(found.values()):
        where = found[POSE_FILE].parent
        lines.append(f"**DWPose:** found in `{where}` · {backend()}")
    else:
        missing = ", ".join(f"`{n}`" for n, p in found.items() if p is None)
        lines.append(
            f"**DWPose: missing** {missing}. Put them in `{anatomy_dir()}` or in the ControlNet "
            f"preprocessor folder (`.../ControlNetPreprocessor/openpose`). "
            f"[Setup]({README_URL})"
        )
    seg = pick_detector("hand_seg")
    sam = pick_sam()
    if seg:
        mask = f"hand-seg `{seg}`"
    elif sam:
        mask = f"SAM `{sam}`"
    else:
        mask = "**none**: hands cannot be checked without a hand-seg model or SAM"
    lines.append(f"**Hand masks:** {mask}")
    hand = pick_detector("hand")
    face = pick_detector("face")
    lines.append(
        f"**Extra hands:** {('`' + hand + '`') if hand else 'off (no hand detector)'} · "
        f"**Faces:** {('`' + face + '`') if face else 'off (no face detector)'} · "
        f"**MediaPipe:** {'installed' if mediapipe_installed() else 'not installed (optional)'}"
    )
    return "  \n".join(lines)


def download() -> str:
    """Fetch the missing DWPose files from Hugging Face into
    models/iadetailer/anatomy. Only on a button press."""
    found = find()
    todo = [name for name, path in found.items() if path is None]
    if not todo:
        return "DWPose models already present."
    target = anatomy_dir()
    target.mkdir(parents=True, exist_ok=True)
    done = []
    for name in todo:
        url = URLS[name]
        print(f"[Impact ADetailer] Inspector: downloading {url}")
        try:
            try:
                from modules.modelloader import load_file_from_url

                load_file_from_url(url, model_dir=str(target), file_name=name)
            except ImportError:
                import shutil
                import urllib.request

                part = target / (name + ".part")
                with urllib.request.urlopen(url, timeout=60) as resp, open(part, "wb") as handle:
                    shutil.copyfileobj(resp, handle)
                part.replace(target / name)
            done.append(name)
        except Exception as exc:
            return f"Download of {name} failed: {exc}. Download it by hand into `{target}`."
    free()
    return "Downloaded " + ", ".join(done) + f" into `{target}`."
