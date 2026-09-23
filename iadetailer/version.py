"""Build stamp from files on disk. Reload UI does not always reload this package."""

from __future__ import annotations

import hashlib
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent


def _source_files() -> list[Path]:
    files: list[Path] = []
    files.extend(sorted(PACKAGE_DIR.glob("*.py")))
    files.extend(sorted((ROOT / "scripts").glob("*.py")))
    files.extend(sorted((ROOT / "javascript").glob("*.js")))
    return files


def build_id() -> str:
    parts = []
    for path in _source_files():
        try:
            stat = path.stat()
            parts.append(f"{path.name}:{int(stat.st_mtime)}:{stat.st_size}")
        except OSError:
            continue
    if not parts:
        return "unknown"
    return hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=4).hexdigest()


def other_installs() -> list[Path]:
    """Other copies of this extension the WebUI would also load.

    Two installs means two Script instances, two copies of the javascript, and
    whichever one loads last wins. It looks exactly like 'my fix did nothing'."""
    found: list[Path] = []
    roots = [ROOT.parent]
    webui = ROOT.parent.parent
    for name in ("extensions", "extensions-builtin"):
        candidate = webui / name
        if candidate.is_dir() and candidate != ROOT.parent:
            roots.append(candidate)
    for base in roots:
        try:
            entries = sorted(base.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir() or entry.resolve() == ROOT:
                continue
            if (entry / "scripts" / "impact_adetailer.py").is_file():
                found.append(entry)
    return found


def neo_false_positive() -> bool:
    """Whether Forge Neo will have called this extension outdated on the way in.

    modules/extensions.py walks prefer_official_extensions - a single entry,
    "ADetailer" -> ADetailer-Neo - and prints the warning for any folder whose
    name contains it, unless the name also contains "neo". Our folder is
    "impact-adetailer", so the substring matches and the line gets printed. It
    is about the folder name and nothing else: no version, no remote, no code.
    """
    try:
        from modules_forge.config import prefer_official_extensions
    except Exception:
        return False
    name = ROOT.name.lower()
    if "neo" in name:
        return False
    return any(key.lower() in name for key in prefer_official_extensions)


def banner() -> str:
    lines = [f"[Impact ADetailer] build {build_id()}  from {ROOT}"]
    if neo_false_positive():
        lines.append(
            '[Impact ADetailer] Forge Neo\'s "might be outdated" line above is a '
            f'substring match on the folder name "{ROOT.name}", not a version '
            "check. Ignore it, or rename the folder to include \"neo\" to silence it."
        )
    duplicates = other_installs()
    if duplicates:
        lines.append("[Impact ADetailer] WARNING: another copy of this extension is installed:")
        for path in duplicates:
            lines.append(f"[Impact ADetailer]   {path}")
        lines.append(
            "[Impact ADetailer] Both will load and fight over the same panel. "
            "Delete the one you are not editing, then restart."
        )
    return "\n".join(lines)
