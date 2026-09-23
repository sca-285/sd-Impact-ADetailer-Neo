"""Box-prompt SAM / SAM2 refine for YOLO detections.

Does not run on import. Ultralytics SAM is preferred (already in this
WebUI env); segment_anything / sam2 are optional fallbacks.

Weight search: models/sam, models/sam2 — same layout as adetailer .pt files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .detect import Detection
from .paths import official_models_path
from .settings import detector_device, log, offline_detectors

_CACHE: dict[str, object] = {}
_KIND: dict[str, str] = {}


def _webui_root() -> Path:
    try:
        from .paths import _webui_root as root

        return root()
    except Exception:
        return Path.cwd()


def sam_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    official = official_models_path()
    if official:
        dirs.extend([official / "sam", official / "sam2"])
    webui = _webui_root()
    dirs.extend(
        [
            webui / "models" / "sam",
            webui / "models" / "sam2",
            webui / "models" / "sam2.1",
        ]
    )
    pack = webui.parent
    for name in ("Model", "model", "Models", "models"):
        dirs.append(pack / name / "sam")
        dirs.append(pack / name / "sam2")
    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        try:
            key = str(d.expanduser().resolve()).lower()
        except Exception:
            key = str(d).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _iter_weights(folder: Path):
    try:
        files = list(folder.rglob("*"))
    except Exception:
        return
    ok = {".pt", ".pth", ".safetensors", ".bin"}
    for p in files:
        if p.is_file() and p.suffix.lower() in ok:
            yield p


def _label(path: Path) -> str:
    parts = [str(x) for x in path.parts]
    lower = [x.lower() for x in parts]
    for key in ("sam2.1", "sam2", "sam"):
        if key in lower:
            rel = parts[lower.index(key) + 1 :]
            return "/".join(rel).replace("\\", "/")
    return path.name


def _scan() -> dict[str, Path]:
    found: dict[str, Path] = {}
    seen: set[str] = set()
    for folder in sam_search_dirs():
        if not folder.is_dir():
            continue
        for p in sorted(_iter_weights(folder), key=lambda x: x.name.lower()):
            try:
                key = str(p.resolve()).lower()
            except Exception:
                key = str(p).lower()
            if key in seen:
                continue
            seen.add(key)
            label = _label(p)
            if label not in found:
                found[label] = p
    return found


def list_sam_models() -> list[str]:
    names = ["None"]
    for label in sorted(_scan().keys()):
        names.append(label)
    return names


def resolve_sam(name: str) -> Path | None:
    if name in ("", "None"):
        return None
    idx = _scan()
    if name in idx:
        return idx[name]
    for path in idx.values():
        if path.name == name:
            return path
    return None


def _device() -> str:
    raw = detector_device()
    if raw and raw != "Automatic":
        return raw
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _kind_for(path: Path) -> str:
    n = path.name.lower()
    if "sam2" in n or "sam_2" in n:
        return "sam2"
    return "sam"


def _load(path: Path):
    if offline_detectors():
        from .offline import apply, harden_ultralytics

        apply()
        harden_ultralytics()

    kind = _kind_for(path)
    last_err: Exception | None = None

    try:
        from ultralytics import SAM

        model = SAM(str(path))
        _KIND[str(path)] = "ultralytics"
        log(f"SAM via ultralytics: {path.name}")
        return model
    except Exception as exc:
        last_err = exc

    if kind == "sam2":
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam = build_sam2(None, str(path), device=_device())
            pred = SAM2ImagePredictor(sam)
            _KIND[str(path)] = "sam2"
            log(f"SAM2 predictor: {path.name}")
            return pred
        except Exception as exc:
            last_err = exc

    try:
        from segment_anything import SamPredictor, sam_model_registry

        name = "vit_b"
        low = path.name.lower()
        if "vit_h" in low or "_h." in low:
            name = "vit_h"
        elif "vit_l" in low or "_l." in low:
            name = "vit_l"
        sam = sam_model_registry[name](checkpoint=str(path))
        sam.to(device=_device())
        pred = SamPredictor(sam)
        _KIND[str(path)] = "sam1"
        log(f"SAM ViT predictor ({name}): {path.name}")
        return pred
    except Exception as exc:
        last_err = exc

    raise RuntimeError(
        f"Could not load SAM weights {path.name}. "
        f"Install ultralytics SAM support, or segment-anything / sam2. "
        f"Last error: {last_err}"
    )


def _get(name: str):
    path = resolve_sam(name)
    if path is None or not path.exists():
        return None, None
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = _load(path)
    return _CACHE[key], _KIND.get(key, "ultralytics")


def _best_mask(masks: np.ndarray, scores: np.ndarray | None) -> np.ndarray:
    if masks.ndim == 2:
        return masks.astype(np.float32)
    if scores is not None and len(scores) == len(masks):
        i = int(np.argmax(scores))
        return masks[i].astype(np.float32)
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    return masks[int(np.argmax(areas))].astype(np.float32)


def _predict_ultralytics(model, image: Image.Image, bbox) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    arr = np.array(image.convert("RGB"))
    kwargs = {"bboxes": [[x1, y1, x2, y2]], "verbose": False}
    dev = detector_device()
    if dev and dev != "Automatic":
        kwargs["device"] = dev
    try:
        results = model.predict(arr, **kwargs)
    except TypeError:
        kwargs.pop("device", None)
        results = model.predict(arr, **kwargs)
    if not results:
        return None
    masks = getattr(results[0], "masks", None)
    if masks is None or getattr(masks, "data", None) is None:
        return None
    data = masks.data.cpu().numpy()
    return _best_mask(data, None)


def _predict_predictor(pred, image: Image.Image, bbox, kind: str) -> np.ndarray | None:
    arr = np.array(image.convert("RGB"))
    box = np.array(bbox, dtype=np.float32)
    try:
        pred.set_image(arr)
        if kind == "sam2":
            masks, scores, _ = pred.predict(box=box, multimask_output=True)
        else:
            masks, scores, _ = pred.predict(box=box, multimask_output=True)
    except Exception as exc:
        log(f"SAM predict failed: {exc}")
        return None
    return _best_mask(np.asarray(masks), np.asarray(scores) if scores is not None else None)


def refine_with_sam(
    image: Image.Image,
    dets: list[Detection],
    model_name: str,
) -> list[Detection]:
    """Fill / replace each detection mask using the box as a SAM prompt.

    On any failure the original detection is kept so the Impact loop still runs.
    """
    if not dets or model_name in ("", "None", None):
        return dets
    try:
        model, kind = _get(model_name)
    except Exception as exc:
        print(f"[Impact ADetailer] SAM load failed: {exc}")
        return dets
    if model is None:
        print(f"[Impact ADetailer] SAM weights not found: {model_name}")
        return dets

    w, h = image.size
    out: list[Detection] = []
    for det in dets:
        mask = None
        try:
            if kind == "ultralytics":
                mask = _predict_ultralytics(model, image, det.bbox)
            else:
                mask = _predict_predictor(model, image, det.bbox, kind)
        except Exception as exc:
            log(f"SAM refine skipped a box: {exc}")
            mask = None
        if mask is None:
            out.append(det)
            continue
        if mask.shape != (h, w):
            m = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L")
            m = m.resize((w, h), Image.BILINEAR)
            mask = np.asarray(m, dtype=np.float32) / 255.0
        out.append(
            Detection(
                bbox=det.bbox,
                score=det.score,
                mask=mask.astype(np.float32),
                label=det.label,
            )
        )
    return out


def clear_sam_cache() -> None:
    keys = list(_CACHE.keys())
    for key in keys:
        model = _CACHE.pop(key, None)
        try:
            inner = getattr(model, "model", None) or getattr(model, "sam", None)
            if inner is not None and hasattr(inner, "cpu"):
                inner.cpu()
        except Exception:
            pass
        del model
    _KIND.clear()
    if keys:
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass