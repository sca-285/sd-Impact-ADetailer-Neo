from __future__ import annotations

import os
from pathlib import Path


def _webui_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        modules = parent / "modules"
        if not modules.is_dir():
            continue
        if (parent / "webui.py").exists() or (modules / "scripts.py").exists() or (modules / "paths.py").exists():
            return parent
    try:
        import modules

        return Path(modules.__file__).resolve().parent.parent
    except Exception:
        return Path.cwd()


def _safe_dir(value) -> Path | None:
    if not value:
        return None
    try:
        p = Path(str(value)).expanduser()
    except Exception:
        return None
    return p


def official_models_path() -> Path | None:
    """Same source ADetailer uses: modules.paths.models_path."""
    candidates = []
    try:
        from modules.paths import models_path as mp

        candidates.append(mp)
    except Exception:
        pass
    try:
        from modules import paths_internal

        candidates.append(getattr(paths_internal, "models_path", None))
    except Exception:
        pass
    try:
        from modules import shared

        for attr in ("models_path",):
            candidates.append(getattr(shared, attr, None))
        opts = getattr(shared, "cmd_opts", None)
        if opts is not None:
            for attr in ("ckpt_dir", "models_dir", "models_path"):
                candidates.append(getattr(opts, attr, None))
    except Exception:
        pass

    for raw in candidates:
        p = _safe_dir(raw)
        if p is None:
            continue
        # ckpt_dir is usually models/Stable-diffusion
        if p.name.lower() in {"stable-diffusion", "stable-diffusion-webui"}:
            p = p.parent
        if p.name.lower() != "models" and (p / "adetailer").is_dir():
            return p
        if p.name.lower() == "models" or p.is_dir():
            return p if p.name.lower() == "models" else (p / "models" if (p / "models").is_dir() else p)
    return None


def extra_yaml_dirs() -> list[Path]:
    roots = [_webui_root(), _webui_root().parent]
    names = ["extra_model_paths.yaml", "extra_model_paths.yml"]
    found: list[Path] = []
    for root in roots:
        for name in names:
            yml = root / name
            if not yml.is_file():
                continue
            try:
                text = yml.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for line in text.splitlines():
                if "adetailer" in line.lower() or "ultralytics" in line.lower():
                    if ":" in line:
                        rhs = line.split(":", 1)[1].strip().strip("\"'")
                        p = _safe_dir(rhs)
                        if p:
                            found.append(p)
    return found


def detector_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    webui = _webui_root()
    official = official_models_path()

    def add(*parts: Path | str):
        cur = parts[0] if isinstance(parts[0], Path) and len(parts) == 1 else None
        if cur is None:
            base = parts[0]
            rest = parts[1:]
            cur = Path(base, *rest) if rest else Path(base)
        dirs.append(cur)

    if official:
        add(official / "adetailer")
        add(official / "ultralytics")
        add(official / "ultralytics" / "bbox")
        add(official / "ultralytics" / "segm")

    add(webui / "models" / "adetailer")
    add(webui / "models" / "ultralytics")
    add(webui / "models" / "ultralytics" / "bbox")
    add(webui / "models" / "ultralytics" / "segm")

    # EasyForge / EasyReforge pack layout: ../Model/adetailer
    pack = webui.parent
    for name in ("Model", "model", "Models", "models"):
        add(pack / name / "adetailer")
        add(pack / name / "ultralytics" / "bbox")
        add(pack / name / "ultralytics" / "segm")
        add(pack / name / "ultralytics")

    try:
        from modules import shared

        extra = ""
        if getattr(shared, "opts", None) is not None:
            extra = shared.opts.data.get("ad_extra_models_dir", "") or ""
            extra2 = shared.opts.data.get("iad_extra_models_dir", "") or ""
            extra = extra2 or extra
        cmd = getattr(shared.cmd_opts, "adetailer_dir", None)
        if cmd:
            add(Path(cmd))
        for chunk in str(extra).replace(";", "|").split("|"):
            chunk = chunk.strip()
            if chunk:
                add(Path(chunk))
    except Exception:
        pass

    for p in extra_yaml_dirs():
        add(p)
        add(p / "adetailer")
        add(p / "bbox")
        add(p / "segm")

    seen: set[str] = set()
    out: list[Path] = []
    for d in dirs:
        try:
            key = str(d.expanduser().resolve())
        except Exception:
            key = str(d)
        if key.lower() in seen:
            continue
        seen.add(key.lower())
        out.append(d)
    return out


def models_adetailer() -> Path:
    for d in detector_search_dirs():
        if d.name.lower() == "adetailer":
            try:
                d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return d
    path = _webui_root() / "models" / "adetailer"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return path


def describe_scan() -> str:
    lines = [f"webui root: {_webui_root()}", f"models_path: {official_models_path()}"]
    hits = 0
    for d in detector_search_dirs():
        exists = d.is_dir()
        n = 0
        if exists:
            try:
                n = sum(1 for p in d.rglob("*") if p.is_file() and p.suffix.lower() in {".pt", ".onnx"})
            except Exception:
                n = 0
        mark = "OK" if exists else "—"
        if n:
            hits += n
            lines.append(f"[{mark}] {d}  ({n} files)")
        else:
            lines.append(f"[{mark}] {d}")
    lines.append(f"total .pt/.onnx seen: {hits}")
    return "\n".join(lines)


def session_temp_root() -> Path:
    try:
        from modules import shared

        tmp = getattr(shared.cmd_opts, "tmpdir", None) or getattr(shared, "tmp_dir", None)
        if tmp:
            p = Path(tmp) / "impact-adetailer"
            p.mkdir(parents=True, exist_ok=True)
            return p
    except Exception:
        pass
    p = Path(os.environ.get("TMPDIR") or os.environ.get("TEMP") or "/tmp") / "impact-adetailer"
    p.mkdir(parents=True, exist_ok=True)
    return p
