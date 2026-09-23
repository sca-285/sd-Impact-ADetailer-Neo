from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .offline import apply as apply_offline
from .paths import detector_search_dirs, models_adetailer
from .settings import detector_device, log, offline_detectors

apply_offline()

_CACHE: dict[str, object] = {}
_INDEX: dict[str, Path] | None = None


def _iter_weight_files(folder: Path):
    try:
        files = list(folder.rglob("*"))
    except Exception:
        return
    for p in files:
        if p.is_file() and p.suffix.lower() in {".pt", ".onnx"}:
            yield p


def _label_for(path: Path) -> str:
    parts = [str(x) for x in path.parts]
    lower = [x.lower() for x in parts]
    for key in ("adetailer", "ultralytics"):
        if key in lower:
            rel = parts[lower.index(key) + 1 :]
            return "/".join(rel).replace("\\", "/")
    parent = path.parent.name.lower()
    if parent in {"bbox", "segm"}:
        return f"{parent}/{path.name}"
    return path.name


def _scan() -> dict[str, Path]:
    found: dict[str, Path] = {}
    seen_files: set[str] = set()
    seen_names: set[str] = set()
    for folder in detector_search_dirs():
        if not folder.is_dir():
            continue
        for p in sorted(_iter_weight_files(folder), key=lambda x: x.name.lower()):
            try:
                key = str(p.resolve()).lower()
            except Exception:
                key = str(p).lower()
            if key in seen_files:
                continue
            seen_files.add(key)
            label = _label_for(p)
            if p.name.lower() in seen_names and label.split("/")[-1].lower() == p.name.lower():
                # same filename already listed (bbox/x.pt vs x.pt)
                if "/" not in label:
                    continue
                bare = p.name
                if bare in found:
                    found.pop(bare, None)
                    seen_names.discard(bare.lower())
            if label in found:
                continue
            found[label] = p
            seen_names.add(p.name.lower())
    return found


def detector_index() -> dict[str, Path]:
    global _INDEX
    if _INDEX is None:
        _INDEX = _scan()
    return _INDEX


def resolve_detector(name: str) -> Path | None:
    if name in ("", "None"):
        return None
    idx = detector_index()
    if name in idx:
        return idx[name]
    # raw filename fallback
    for path in idx.values():
        if path.name == name:
            return path
    direct = models_adetailer() / name
    if direct.exists():
        return direct
    return None


@dataclass
class Detection:
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    score: float
    mask: np.ndarray | None  # HxW float32 0..1, same size as image, or None
    label: str = ""  # the detector's class name for this box, when it has one


def parse_classes(spec: str) -> tuple[list[str], list[str]]:
    """"face, hand, -ear" -> (["face", "hand"], ["ear"]).

    Commas or newlines separate terms; a leading - or ! excludes."""
    wanted: list[str] = []
    banned: list[str] = []
    for chunk in str(spec or "").replace("\n", ",").split(","):
        term = chunk.strip()
        if not term:
            continue
        if term[0] in "-!":
            term = term[1:].strip()
            if term:
                banned.append(term.lower())
        else:
            wanted.append(term)
    return wanted, banned


def _model_class_names(model) -> dict:
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, (list, tuple)):
        return {i: str(v) for i, v in enumerate(names)}
    return {}


def _set_vocabulary(model, wanted: list[str]) -> None:
    """Hand the words to the model, whichever signature this build wants.

    YOLOWorld has always been `set_classes(classes)`. YOLOE was
    `set_classes(classes, embeddings)` with embeddings REQUIRED from the version
    that introduced it up to 8.3.x, and only became optional in 8.4.0. Calling
    it the one-argument way on anything older raises TypeError, which the caller
    used to swallow as "open-vocabulary unavailable" - so on those versions a
    YOLOE model silently detected its default vocabulary instead of the words
    you typed, with nothing in the log to say so. The embeddings it wants are
    what get_text_pe returns.
    """
    names = list(wanted)
    try:
        model.set_classes(names)
        return
    except TypeError:
        encoder = getattr(model, "get_text_pe", None)
        if not callable(encoder):
            raise
    model.set_classes(names, encoder(names))


def _open_vocab(model, wanted: list[str]) -> bool:
    """Teach a YOLO-World / YOLOE model what to look for.

    These models carry no fixed class list: `set_classes` runs the terms through
    a text encoder and rebuilds the head, so the detector finds whatever you
    typed. Encoding is not cheap, so the last list applied is remembered on the
    model and only re-applied when it changes.

    A plain YOLO has no set_classes; it comes back False and the caller falls
    back to filtering the fixed class names instead.
    """
    setter = getattr(model, "set_classes", None)
    if not callable(setter) or not wanted:
        return False
    key = tuple(w.lower() for w in wanted)
    if getattr(model, "_iad_classes", None) == key:
        return True
    try:
        _set_vocabulary(model, wanted)
    except Exception as exc:
        # set_classes pulls a text encoder in, which on a machine with no
        # network (or with the model cache missing) simply cannot happen.
        #
        # This prints rather than log()s. log() is gated by 'Quiet terminal
        # logs', which is ON by default, so the one line that says the open
        # vocabulary did NOT take was the line nobody ever saw - and the run
        # carries on detecting the model's default classes, which looks
        # exactly like it working.
        if getattr(model, "_iad_vocab_warned", False) is False:
            print(
                "[Impact ADetailer] WARNING: open-vocabulary classes were NOT "
                f"applied: {type(exc).__name__}: {exc}"
            )
            print(
                "[Impact ADetailer]   This detector is now finding its OWN "
                "default classes, not the words you typed."
            )
            if offline_detectors():
                print(
                    "[Impact ADetailer]   set_classes needs a text encoder "
                    "(CLIP for -world, MobileCLIP for yoloe) and 'Keep "
                    "YOLO/Ultralytics offline' blocks the install and the "
                    "download it would do to get one. Turn that setting off for "
                    "one generation, then turn it back on."
                )
            try:
                model._iad_vocab_warned = True
            except Exception:
                pass
        return False
    try:
        model._iad_classes = key
    except Exception:
        pass
    log("detector classes set to: " + ", ".join(wanted))
    return True


@functools.lru_cache(maxsize=512)
def _term_pattern(term: str):
    """`term` as a whole word, with an optional plural s.

    Plain substring matching was quietly wrong on a COCO vocabulary: "hand"
    matches "handbag", "ear" matches "bear" and "teddy bear", "car" matches
    "carrot", "cat" matches nothing extra but "bus" matches "business". Typing
    "face, hand" against an 80-class model therefore looked like it had found a
    class it understood, kept the filter, and returned only handbags.

    A word boundary fixes that without losing the reason substrings were used
    in the first place: "face" still matches "human face" and "face of person",
    and the optional s still matches "hands" and "eyes".
    """
    return re.compile(r"(?<![a-z0-9])" + re.escape(term.lower()) + r"s?(?![a-z0-9])")


def _matches(label: str, term: str) -> bool:
    return bool(_term_pattern(term).search(label.lower()))


def _keep_label(label: str, wanted: list[str], banned: list[str]) -> bool:
    if banned and any(_matches(label, b) for b in banned):
        return False
    if not wanted:
        return True
    return any(_matches(label, w) for w in wanted)


def list_detector_models() -> list[str]:
    global _INDEX
    _INDEX = _scan()
    names = ["None"]
    for label in sorted(_INDEX.keys()):
        if label not in names:
            names.append(label)
    if len(names) == 1:
        # Nothing found. Printing the folders that were actually looked at
        # saves a round of "where do I put the .pt file".
        from .paths import describe_scan

        print("[Impact ADetailer] no detector models found. Searched:")
        print(describe_scan())
    return names


def _load_yolo(path: Path):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics in the WebUI Python env") from exc

    if offline_detectors():
        # ultralytics is imported now, so its module-level online flag exists
        # and can actually be turned off.
        from .offline import harden_ultralytics

        harden_ultralytics()

    try:
        return YOLO(str(path))
    except Exception as exc:
        # torch 2.6 flipped torch.load to weights_only=True, which refuses the
        # pickled ultralytics classes inside older .pt detectors. These are
        # local files the user put in models/adetailer themselves, the same
        # trust level Impact Subpack's model whitelist assumes.
        text = f"{type(exc).__name__}: {exc}"
        markers = ("weights_only", "WeightsUnpickler", "Unsupported global", "GLOBAL")
        if not any(m in text for m in markers):
            raise
        import torch

        original = torch.load

        def _load_trusted(*args, **kwargs):
            kwargs["weights_only"] = False
            return original(*args, **kwargs)

        torch.load = _load_trusted
        try:
            from ultralytics import YOLO

            log(f"loading {path.name} with weights_only=False (torch 2.6+)")
            return YOLO(str(path))
        finally:
            torch.load = original


def clear_model_cache() -> None:
    """Drop the loaded detectors. Called on UI reload, and after a generation
    when 'Unload detector models' is on."""
    names = list(_CACHE.keys())
    for name in names:
        model = _CACHE.pop(name, None)
        try:
            inner = getattr(model, "model", None)
            if inner is not None and hasattr(inner, "cpu"):
                inner.cpu()
        except Exception:
            pass
        del model
    if names:
        try:
            import gc

            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _warn_if_misrouted(path: Path, model) -> None:
    """Catch an open-vocabulary checkpoint that was loaded as a plain detector.

    ultralytics picks the wrapper class from the FILE NAME:

        if "-world" in path.stem: -> YOLOWorld
        elif "yoloe" in path.stem: -> YOLOE

    (ultralytics/models/yolo/model.py). Verified on 8.3.90, 8.3.130 and 8.4.0:
    the name test is the only test, so renaming yolov8s-world.pt to
    world_v8s.pt hands back a plain detector with no set_classes at all. It
    still loads, still detects, and still returns boxes - the 80 COCO classes
    baked into the checkpoint - so from the outside it looks like a working
    open-vocabulary model that simply never finds what you ask for.

    Recent builds (8.4.153 checked) added an `isinstance(self.model,
    WorldModel)` rescue in the else branch, so a renamed -world file survives
    there. No such rescue exists for YOLOE on any version checked, and older
    ultralytics has neither. Hence this check: it costs nothing and it only
    speaks when set_classes really is missing.
    """
    stem = path.stem.lower()
    if callable(getattr(model, "set_classes", None)):
        return
    if "world" not in stem and "yoloe" not in stem:
        return
    want = "-world" if "world" in stem else "yoloe"
    print(
        f"[Impact ADetailer] WARNING: {path.name} looks like an open-vocabulary "
        f"model, but ultralytics loaded it as a plain detector."
    )
    print(
        f"[Impact ADetailer]   On this ultralytics version the wrapper is "
        f'chosen by file name and needs "{want}" in it. Rename the file (e.g. '
        "yolov8s-world.pt, yoloe-11s-seg.pt) and rescan; until then the "
        "Detector classes box can only filter this checkpoint's built-in class "
        "names."
    )


class DetectorCache:
    def get(self, name: str):
        if name in ("", "None"):
            return None
        if name in _CACHE:
            return _CACHE[name]
        path = resolve_detector(name)
        if path is None or not path.exists():
            raise FileNotFoundError(f"Detector not found: {name}")
        if offline_detectors():
            apply_offline()
        model = _load_yolo(path)
        _warn_if_misrouted(path, model)
        _CACHE[name] = model
        return model


_MISSING = object()


def _forget(model_name: str) -> None:
    """Drop one detector from the cache so the next run loads it clean."""
    _CACHE.pop(model_name, None)


def _reset_predictor(model, device) -> None:
    try:
        model.predictor = None
    except Exception:
        pass
    try:
        model._iad_device = device
    except Exception:
        pass


def _predict(model, arr, model_name: str, confidence: float, device: str):
    """One detector pass, with the device actually applied.

    Two things go wrong if you just call model.predict(device=...).

    THE SETTING STOPS TAKING EFFECT. ultralytics builds a predictor on the first
    predict and reuses it. Through 8.3.x the reuse test is only
    `if not self.predictor:` (engine/model.py:541 in 8.3.90, :543 in 8.3.130), so
    once one exists, a different `device=` in the args changes nothing: the model
    stays on whatever device the first generation happened to pick, for the rest
    of the session. 8.4.0 added `or self.predictor.args.device != args.get(...)`
    to that line. Rather than depend on which build is installed, the device is
    remembered here and the predictor dropped whenever it changes - which is
    exactly what the 8.4 test does, one layer up.

    A BAD DEVICE POISONS THE CACHED MODEL. When select_device raises, it raises
    inside setup_model, one line AFTER self.predictor was assigned. What is left
    behind is a predictor with args.device set and self.model still None. The
    next call finds a predictor, skips the rebuild, reaches
    `if self.model is None: self.setup_model(...)` in stream_inference, and
    raises the same error again. The old handler here retried with `device`
    popped out of kwargs, which could never work - the dead device lives on the
    predictor, not in the arguments. Measured on an ONNX detector: one failed
    cuda attempt killed that tab for the whole session, and switching the
    setting back to Automatic did not revive it. Only Reload UI did.

    So: reset the predictor before retrying, and if the retry fails too, drop the
    model from the cache so the next generation starts from a clean load instead
    of a tab that is permanently dead.
    """
    kwargs = {"conf": confidence, "verbose": False}
    explicit = bool(device) and device != "Automatic"
    if explicit:
        kwargs["device"] = device

    if getattr(model, "_iad_device", _MISSING) != device:
        _reset_predictor(model, device)

    try:
        return model.predict(arr, **kwargs)
    except Exception as exc:
        if not explicit:
            _forget(model_name)
            raise
        print(
            f"[Impact ADetailer] WARNING: {model_name} could not run on "
            f"'{device}': {type(exc).__name__}: {exc}"
        )
        print("[Impact ADetailer]   Retrying on the automatic device.")
        _reset_predictor(model, "Automatic")
        kwargs.pop("device", None)
        try:
            return model.predict(arr, **kwargs)
        except Exception:
            _forget(model_name)
            raise


def run_detector(
    image: Image.Image,
    model_name: str,
    confidence: float,
    cache: DetectorCache | None = None,
    classes: str = "",
    report: dict | None = None,
) -> list[Detection]:
    """Boxes from one detector.

    `report`, when given, is filled in with what the class filter actually did:
    `mode` (`open-vocabulary`, `name filter`, `not in vocabulary`, or `none`),
    `terms`, `vocabulary`, and `dropped` as (Detection, reason) pairs. The
    detection sheet prints it, which is the only way to tell from the outside
    whether an open-vocabulary model took your words or quietly fell back to
    its own class list.
    """
    cache = cache or DetectorCache()
    model = cache.get(model_name)
    if model is None:
        return []

    wanted, banned = parse_classes(classes)
    open_vocab = _open_vocab(model, wanted)
    names = _model_class_names(model)
    mode = "none" if not (wanted or banned) else "name filter"
    unknown: list[str] = []
    if open_vocab:
        # The words went into the model itself, so every box it just returned is
        # already one of them. Filtering the names a second time only risks
        # throwing boxes away over a spelling difference in the head's labels.
        wanted = []
        mode = "open-vocabulary"
    elif wanted and names:
        unknown = [w for w in wanted if not any(_matches(n, w) for n in names.values())]
        if len(unknown) == len(wanted):
            # A fixed-class detector that has never heard of any of these terms.
            #
            # This used to drop the filter and let every box through, on the
            # theory that an untouched image is unhelpful. It is worse than
            # unhelpful: asking a COCO model for "face" then returned the
            # people AND the neckties it found, and the tab inpainted a necktie
            # with a face prompt. Keeping nothing is the safe answer, and the
            # detection sheet now shows the boxes in grey with the reason, so
            # it is no longer a silent nothing.
            mode = "not in vocabulary"
            if not getattr(model, "_iad_class_warned", False):
                print(
                    f"[Impact ADetailer] WARNING: {model_name} has no class "
                    f"matching {wanted}. Nothing will be detailed by this tab."
                )
                print(
                    "[Impact ADetailer]   It only knows: "
                    + ", ".join(sorted(set(names.values())))
                )
                try:
                    model._iad_class_warned = True
                except Exception:
                    pass
    if report is not None:
        report["mode"] = mode
        report["terms"] = [*parse_classes(classes)[0], *(f"-{b}" for b in banned)]
        report["vocabulary"] = sorted(set(names.values())) if names else []
        report["unknown"] = unknown
        report["dropped"] = []

    arr = np.array(image.convert("RGB"))
    h, w = arr.shape[:2]
    results = _predict(model, arr, model_name, confidence, detector_device())
    if not results:
        return []

    out: list[Detection] = []
    res = results[0]
    boxes = getattr(res, "boxes", None)
    masks = getattr(res, "masks", None)
    if boxes is None or boxes.xyxy is None:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
    mask_data = None
    if masks is not None and getattr(masks, "data", None) is not None:
        mask_data = masks.data.cpu().numpy()

    # After set_classes the result carries the new vocabulary, so read the names
    # off the result rather than off the model we probed earlier.
    res_names = getattr(res, "names", None)
    if res_names:
        names = (
            {int(k): str(v) for k, v in res_names.items()}
            if isinstance(res_names, dict)
            else {i: str(v) for i, v in enumerate(res_names)}
        )
    cls_ids = None
    if getattr(boxes, "cls", None) is not None:
        try:
            cls_ids = boxes.cls.cpu().numpy()
        except Exception:
            cls_ids = None

    reject_all = mode == "not in vocabulary"
    for i, box in enumerate(xyxy):
        label = ""
        if cls_ids is not None and i < len(cls_ids):
            label = names.get(int(cls_ids[i]), "")
        cut = reject_all or ((wanted or banned) and not _keep_label(label, wanted, banned))
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        if cut:
            if report is not None:
                reason = "not in vocabulary" if reject_all else "class filter"
                report["dropped"].append((
                    Detection(bbox=(x1, y1, x2, y2), score=float(confs[i]),
                              mask=None, label=label),
                    f"{reason}: {label or '?'}",
                ))
            continue
        mask = None
        if mask_data is not None and i < len(mask_data):
            m = mask_data[i]
            # ultralytics mask is often model-input size
            if m.shape != (h, w):
                from PIL import Image as PImage

                m_img = PImage.fromarray((m * 255).astype(np.uint8), mode="L")
                m_img = m_img.resize((w, h), PImage.BILINEAR)
                m = np.array(m_img).astype(np.float32) / 255.0
            mask = m.astype(np.float32)
        out.append(
            Detection(bbox=(x1, y1, x2, y2), score=float(confs[i]), mask=mask, label=label)
        )
    return out
