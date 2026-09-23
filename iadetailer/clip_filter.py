"""CLIP-ViT filter for YOLO boxes.

Scores each crop against the tab prompt. Boxes below the threshold are
dropped before the Impact loop.

Accepts either:
  - OpenCLIP / OpenAI .pt  (keys like visual.conv1, positional_embedding)
  - HuggingFace CLIP folder (config.json + weights, keys like text_model.*)

Does not download. Missing config or the wrong format is a one-line error,
then the filter is skipped and YOLO boxes are kept.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

from PIL import Image

from .detect import Detection
from .paths import official_models_path
from .settings import detector_device, log, offline_detectors

_CACHE: dict[str, object] = {}


def ensure_clip_model_ready(clip_models_dir: str, model_name: str) -> str:
    """Auto-fixes bare .safetensors files into proper HuggingFace directory structures."""
    root_file_path = os.path.join(clip_models_dir, f"{model_name}.safetensors")
    folder_path = os.path.join(clip_models_dir, model_name)
    
    target_weights = os.path.join(folder_path, "model.safetensors")
    old_weights_in_folder = os.path.join(folder_path, f"{model_name}.safetensors")
    
    required_files = [
        "config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json"
    ]
    
    if os.path.exists(root_file_path) or os.path.exists(old_weights_in_folder):
        os.makedirs(folder_path, exist_ok=True)
        
    if os.path.exists(root_file_path):
        shutil.move(root_file_path, target_weights)
        print(f"[Impact ADetailer] Auto-moved {model_name}.safetensors into folder")
    elif os.path.exists(old_weights_in_folder):
        shutil.move(old_weights_in_folder, target_weights)
        print(f"[Impact ADetailer] Auto-renamed weights to model.safetensors")
        
    if os.path.exists(target_weights):
        base_url = f"https://huggingface.co/laion/{model_name}/resolve/main/"
        for file_name in required_files:
            file_path = os.path.join(folder_path, file_name)
            if not os.path.exists(file_path):
                try:
                    print(f"[Impact ADetailer] Downloading {file_name}...")
                    urllib.request.urlretrieve(base_url + file_name, file_path)
                except Exception as e:
                    print(f"[Impact ADetailer] Failed to download {file_name}: {e}")
                    
        return folder_path
        
    return clip_models_dir


def _short_err(exc: BaseException, limit: int = 180) -> str:
    text = f"{type(exc).__name__}: {exc}".replace("\n", " ").strip()
    if "Missing key" in text or "Unexpected key" in text or "state_dict" in text:
        return (
            "wrong CLIP weight format "
            "(OpenCLIP .pt, or a HuggingFace folder with config.json)"
        )
    return text[:limit]


def _webui_root() -> Path:
    try:
        from .paths import _webui_root as root

        return root()
    except Exception:
        return Path.cwd()


def clip_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    official = official_models_path()
    if official:
        dirs.extend([official / "clip", official / "CLIP", official / "openclip"])
    webui = _webui_root()
    dirs.extend(
        [
            webui / "models" / "clip",
            webui / "models" / "CLIP",
            webui / "models" / "openclip",
        ]
    )
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


def _iter_entries(folder: Path):
    """Yield weight files and HuggingFace CLIP folders."""
    try:
        children = list(folder.iterdir())
    except Exception:
        return
    ok = {".pt", ".pth", ".bin", ".safetensors"}
    for p in children:
        if p.is_file() and p.suffix.lower() in ok:
            yield p
        elif p.is_dir() and (p / "config.json").is_file():
            yield p
        elif p.is_dir():
            for q in _iter_entries(p):
                yield q


def _label(path: Path) -> str:
    parts = [str(x) for x in path.parts]
    lower = [x.lower() for x in parts]
    for key in ("openclip", "clip"):
        if key in lower:
            rel = parts[lower.index(key) + 1 :]
            return "/".join(rel).replace("\\", "/")
    return path.name


def _scan() -> dict[str, Path]:
    found: dict[str, Path] = {}
    seen: set[str] = set()
    for folder in clip_search_dirs():
        if not folder.is_dir():
            continue
        for p in sorted(_iter_entries(folder), key=lambda x: str(x).lower()):
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


def list_clip_models() -> list[str]:
    names = ["None"]
    for label in sorted(_scan().keys()):
        names.append(label)
    return names


def resolve_clip(name: str) -> Path | None:
    if name in ("", "None"):
        return None
    idx = _scan()
    if name in idx:
        return idx[name]
    for path in idx.values():
        if path.name == name:
            return path
    return None


def _device():
    import torch

    raw = detector_device()
    if raw and raw != "Automatic":
        return torch.device(raw)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _arch_from_name(name: str) -> str:
    low = name.lower()
    if "vit-l-14" in low or "vit_l_14" in low or "vitl14" in low:
        return "ViT-L-14"
    if "vit-b-16" in low or "vit_b_16" in low or "vitb16" in low:
        return "ViT-B-16"
    return "ViT-B-32"


def _peek_keys(path: Path) -> list[str]:
    import torch

    target = path
    if path.is_dir():
        for name in ("model.safetensors", "pytorch_model.bin", "model.bin"):
            cand = path / name
            if cand.is_file():
                target = cand
                break
        else:
            return ["text_model.embeddings.token_embedding.weight"]
    if target.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import load_file

            return list(load_file(str(target)).keys())[:12]
        except Exception:
            return []
    raw = torch.load(str(target), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        raw = raw["state_dict"]
    if isinstance(raw, dict):
        return list(raw.keys())[:12]
    return []


def _is_hf(path: Path, keys: list[str]) -> bool:
    if path.is_dir() and (path / "config.json").is_file():
        return True
    if (path.parent / "config.json").is_file():
        return True
    return any(k.startswith("text_model.") or k.startswith("vision_model.") for k in keys)


def _load_hf(path: Path):
    from transformers import CLIPModel, CLIPProcessor
    import torch

    folder = path if path.is_dir() else path.parent
    if not (folder / "config.json").is_file():
        raise RuntimeError(
            f"{path.name} looks like HuggingFace CLIP but {folder} has no config.json"
        )
    model = CLIPModel.from_pretrained(str(folder), local_files_only=True)
    processor = CLIPProcessor.from_pretrained(str(folder), local_files_only=True)
    model.eval()
    log(f"CLIP via transformers: {folder.name}")
    return {
        "kind": "hf",
        "model": model,
        "processor": processor,
        "target_device": _device(),
    }


def _load_openclip(path: Path):
    import open_clip
    import torch

    if path.is_dir():
        raise RuntimeError(f"{path.name} is a folder; OpenCLIP needs a .pt file")
    arch = _arch_from_name(path.name)
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        arch,
        pretrained=str(path),
        device=torch.device("cpu"),
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(arch)
    log(f"CLIP via open_clip {arch}: {path.name}")
    return {
        "kind": "openclip",
        "model": model,
        "preprocess": preprocess,
        "tokenizer": tokenizer,
        "target_device": _device(),
    }


def _load(path: Path):
    if offline_detectors():
        from .offline import apply

        apply()
    keys: list[str] = []
    try:
        keys = _peek_keys(path)
    except Exception:
        keys = []
    if _is_hf(path, keys):
        return _load_hf(path)
    return _load_openclip(path)


def _get(name: str):
    path = resolve_clip(name)
    if path is None or not path.exists():
        return None
        
    if path.is_file() and path.suffix.lower() == ".safetensors":
        model_name = path.stem
        clip_models_dir = str(path.parent)
        new_path_str = ensure_clip_model_ready(clip_models_dir, model_name)
        path = Path(new_path_str)
    elif path.is_dir():
        model_name = path.name
        clip_models_dir = str(path.parent)
        new_path_str = ensure_clip_model_ready(clip_models_dir, model_name)
        path = Path(new_path_str)

    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = _load(path)
    return _CACHE[key]


def _embed(bundle, crop: Image.Image, text: str) -> float:
    import torch
    import torch.nn.functional as F

    target_device = bundle.get("target_device", torch.device("cpu"))
    model = bundle["model"]

    model.to(target_device)

    try:
        if bundle.get("kind") == "hf":
            inputs = bundle["processor"](
                text=[text],
                images=crop.convert("RGB"),
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(target_device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                img = F.normalize(out.image_embeds, dim=-1)
                txt = F.normalize(out.text_embeds, dim=-1)
                return float((img * txt).sum(dim=-1).item())

        image = bundle["preprocess"](crop.convert("RGB")).unsqueeze(0).to(target_device)
        tokens = bundle["tokenizer"]([text]).to(target_device)
        with torch.no_grad():
            img_f = F.normalize(model.encode_image(image), dim=-1)
            txt_f = F.normalize(model.encode_text(tokens), dim=-1)
            return float((img_f * txt_f).sum(dim=-1).item())
            
    finally:
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _hint(prompt: str) -> str:
    text = (prompt or "").strip()
    if not text:
        return "a face"
    return text.split(",")[0].strip()[:120] or "a face"


def filter_by_clip(
    image: Image.Image,
    dets: list[Detection],
    prompt: str,
    model_name: str,
    threshold: float = 0.2,
) -> list[Detection]:
    """Drop boxes whose crop does not match `prompt` under CLIP cosine sim."""
    if not dets or model_name in ("", "None", None):
        return dets
    try:
        bundle = _get(model_name)
    except Exception as exc:
        print(f"[Impact ADetailer] CLIP load failed: {_short_err(exc)}")
        return dets
    if bundle is None:
        print(f"[Impact ADetailer] CLIP weights not found: {model_name}")
        return dets

    hint = _hint(prompt)
    kept: list[Detection] = []
    for det in dets:
        x1, y1, x2, y2 = det.bbox
        crop = image.crop((x1, y1, x2, y2))
        if crop.size[0] < 8 or crop.size[1] < 8:
            continue
        try:
            score = _embed(bundle, crop, hint)
        except Exception as exc:
            log(f"CLIP score failed: {_short_err(exc)}")
            kept.append(det)
            continue
        log(f"CLIP {score:.3f} vs {threshold:.3f} for '{hint}'")
        if score >= float(threshold):
            kept.append(det)
    return kept


def clear_clip_cache() -> None:
    keys = list(_CACHE.keys())
    for key in keys:
        bundle = _CACHE.pop(key, None)
        try:
            model = bundle.get("model") if isinstance(bundle, dict) else None
            if model is not None and hasattr(model, "cpu"):
                model.cpu()
        except Exception:
            pass
        del bundle
    if keys:
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass