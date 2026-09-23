from __future__ import annotations

import re
from typing import Any

import numpy as np
from PIL import Image

from .args import UnitArgs
from .detect import DetectorCache, Detection, run_detector
from .geom import (
    bbox_size,
    crop_image,
    dilate_mask,
    gaussian_feather,
    make_crop_region,
    paste_pixel,
    resize_lanczos,
    scale_for_guide,
    shift_mask,
)
from . import preview as preview_mod
from .sample import sample_crop


def _interrupted() -> bool:
    """Honour the Interrupt / Skip buttons between detections."""
    try:
        from modules import shared

        st = getattr(shared, "state", None)
        if st is None:
            return False
        return bool(getattr(st, "interrupted", False) or getattr(st, "skipped", False))
    except Exception:
        return False


def image_seed(p: Any) -> int:
    """The seed of the image currently being post-processed, not the batch base,
    so every image in a batch gets its own detailer noise."""
    try:
        idx = int(getattr(p, "batch_index", 0) or 0)
        seeds = getattr(p, "seeds", None)
        if seeds and 0 <= idx < len(seeds):
            return int(seeds[idx])
    except Exception:
        pass
    try:
        return int(getattr(p, "seed", 0) or 0)
    except Exception:
        return 0


def image_prompt(p: Any, negative: bool = False) -> str:
    """The prompt that actually produced this image.

    p.prompt is the raw box contents: styles are not folded in and a wildcard
    or prompt-matrix run has already moved on. p.prompts / p.all_prompts carry
    the resolved text per image, so the detailer repaints with the same words
    the base image was made from."""
    batch_key = "negative_prompts" if negative else "prompts"
    all_key = "all_negative_prompts" if negative else "all_prompts"
    try:
        idx = int(getattr(p, "batch_index", 0) or 0)
        batch = getattr(p, batch_key, None)
        if batch and 0 <= idx < len(batch):
            return str(batch[idx])
        every = getattr(p, all_key, None)
        if every:
            return str(every[min(idx, len(every) - 1)])
    except Exception:
        pass
    return str(getattr(p, "negative_prompt" if negative else "prompt", "") or "")


_SEP = re.compile(r"\[SEP\]", re.IGNORECASE)
_SKIP = re.compile(r"^\s*\[SKIP\]\s*$", re.IGNORECASE)
_MAIN = re.compile(r"\[PROMPT\]", re.IGNORECASE)
_CLASS = re.compile(r"\[\s*CLASS\s*:\s*([^\]]*)\]", re.IGNORECASE)


def split_prompt(text: str) -> list[str]:
    """One entry per [SEP]-separated section. Always at least one entry."""
    parts = [chunk.strip() for chunk in _SEP.split(str(text or ""))]
    return parts or [""]


def prompt_for(text: str, index: int) -> str:
    """The section that belongs to detection `index`.

    Fewer sections than detections is normal - they cycle, so a single
    "closed eyes" still applies to every face, while
    "smiling [SEP] [SKIP] [SEP] frowning" gives face 1 a smile, leaves face 2
    alone entirely and sets face 3 scowling."""
    parts = split_prompt(text)
    return parts[index % len(parts)]


def is_skip(text: str) -> bool:
    return bool(_SKIP.match(text or ""))


def class_tags(section: str) -> tuple[list[str], str]:
    """Split "[CLASS:face, eye] soft skin" into (["face", "eye"], "soft skin")."""
    tags: list[str] = []

    def take(match):
        tags.extend(t.strip() for t in match.group(1).split(",") if t.strip())
        return " "

    body = _CLASS.sub(take, str(section or ""))
    # The tag leaves a space behind so "soft[CLASS:hand]skin" does not become
    # one word. Collapsing runs of spaces afterwards keeps a tag written in the
    # middle of a section from leaving a double space in the prompt. Newlines
    # are left alone - the prompt box is multi-line and that is the user's.
    body = re.sub(r"[ \t]+", " ", body).strip()
    return tags, body


def section_for(text: str, index: int, label: str = "") -> str | None:
    """The prompt section for one detection, by class name when it is tagged.

    `[SEP]` hands out sections by POSITION, which is the right model for one
    detector finding several of the same thing: "smiling [SEP] frowning" gives
    face 1 a smile and face 2 a scowl. It is the wrong model for a detector that
    returns several different things - a multi-class detector's boxes arrive in
    whatever order the sort mode produced, so which section lands on which body
    part is not something you can write down.

    `[CLASS:...]` binds a section to a class name instead:

        [CLASS:face] detailed face [SEP] [CLASS:hand] slender fingers [SEP] skin

    Every face gets the first section, every hand the second, and anything else
    falls through to the untagged one. Matching is the same word-boundary rule
    the Detector classes box uses, so [CLASS:breast] matches
    FEMALE_BREAST_EXPOSED and [CLASS:hand] does not match handbag.

    Returns None when the section list is tagged, nothing matched, and there is
    no untagged section to fall back to - the caller leaves that detection
    alone and the detection sheet says why. With no [CLASS:] tag anywhere this
    is exactly prompt_for(), down to the cycling.
    """
    parts = split_prompt(text)
    parsed = [class_tags(part) for part in parts]
    if not any(tags for tags, _ in parsed):
        return parts[index % len(parts)]

    from .detect import _matches

    for tags, body in parsed:
        if tags and any(_matches(label, tag) for tag in tags):
            return body
    for tags, body in parsed:
        if not tags:
            return body
    return None


def _pick_prompts(unit: UnitArgs, p: Any, index: int = 0, label: str = "") -> tuple[str, str]:
    pos = (section_for(unit.prompt, index, label) or "").strip()
    neg = (section_for(unit.negative_prompt, index, label) or "").strip()
    main_pos = image_prompt(p) or ""
    main_neg = image_prompt(p, negative=True) or ""
    # [PROMPT] splices the image's own prompt in, so a tab can add to it
    # instead of replacing it: "[PROMPT], detailed face".
    pos = _MAIN.sub(main_pos, pos) if _MAIN.search(pos) else (pos or main_pos)
    neg = _MAIN.sub(main_neg, neg) if _MAIN.search(neg) else (neg or main_neg)
    return pos, neg


def _pick_sampler(unit: UnitArgs, p: Any) -> tuple[str, str]:
    sampler = p.sampler_name
    scheduler = getattr(p, "scheduler", "Automatic")
    if getattr(unit, "use_separate_sampler", False):
        if unit.sampler and unit.sampler != "Use same sampler":
            sampler = unit.sampler
        if unit.scheduler and unit.scheduler != "Use same scheduler":
            scheduler = unit.scheduler
    return sampler, scheduler


def _controlnet_cfg(unit: UnitArgs, p: Any = None) -> dict | None:
    """What sample_crop needs to run ControlNet, or None to stay in the clean
    room."""
    model = str(getattr(unit, "cn_model", "None") or "None")
    if model == "None":
        return None
    if p is not None and getattr(p, "iad_arch_mismatch", False):
        return None
    return {
        "model": model,
        "module": str(getattr(unit, "cn_module", "None") or "None"),
        "weight": float(getattr(unit, "cn_weight", 1.0) or 0.0),
        "start": float(getattr(unit, "cn_start", 0.0) or 0.0),
        "end": float(getattr(unit, "cn_end", 1.0) or 0.0),
    }


def _effective_guide(unit: UnitArgs) -> tuple[int, int]:
    guide = unit.guide_size
    max_size = unit.max_size
    if getattr(unit, "use_wh", False):
        guide = min(int(unit.inpaint_w), int(unit.inpaint_h))
        max_size = max(int(unit.inpaint_w), int(unit.inpaint_h), max_size)
    return guide, max_size


def _shift_bbox(bbox, dx, dy, w, h):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, x1 + dx))
    x2 = max(1, min(w, x2 + dx))
    y1 = max(0, min(h - 1, y1 - dy))  # ADetailer y offset is up
    y2 = max(1, min(h, y2 - dy))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def _crop_factor(unit: UnitArgs, bbox) -> float:
    """crop_factor with the padding slider folded in.

    They mean the same thing from two directions: crop_factor is a multiple of
    the box, padding is a number of pixels around it. Only fold padding in while
    crop_factor is at or below its default, otherwise deliberately widening the
    crop would be silently overridden by a padding value left at 32."""
    factor = float(unit.crop_factor)
    padding = int(getattr(unit, "padding", 0) or 0)
    if padding and factor <= 3.01:
        bw, bh = bbox_size(bbox)
        factor = max(factor, 1.0 + (2.0 * padding) / max(1.0, float(min(bw, bh))))
    return factor


def _seg_mask_full(det: Detection, w: int, h: int) -> np.ndarray:
    mask = det.mask.astype(np.float32)
    if mask.shape != (h, w):
        m = Image.fromarray((np.clip(mask, 0, 1) * 255).astype(np.uint8), mode="L")
        m = m.resize((w, h), Image.BILINEAR)
        mask = np.asarray(m, dtype=np.float32) / 255.0
    return mask


def crop_region_mask(
    image_size: tuple[int, int],
    det: Detection,
    dilate: int,
    crop_region: tuple[int, int, int, int],
) -> np.ndarray:
    """Mask for the crop region only.

    Erosion/dilation runs on a window padded by the kernel radius so the result
    is identical to masking the whole canvas, without allocating a float array
    the size of the image for every detection.
    """
    w, h = image_size
    x1, y1, x2, y2 = crop_region
    pad = abs(int(dilate)) + 2
    mx1, my1 = max(0, x1 - pad), max(0, y1 - pad)
    mx2, my2 = min(w, x2 + pad), min(h, y2 + pad)

    if det.mask is not None:
        window = _seg_mask_full(det, w, h)[my1:my2, mx1:mx2].copy()
    else:
        window = np.zeros((my2 - my1, mx2 - mx1), dtype=np.float32)
        bx1, by1, bx2, by2 = det.bbox
        bx1, by1 = max(mx1, bx1), max(my1, by1)
        bx2, by2 = min(mx2, bx2), min(my2, by2)
        if bx2 > bx1 and by2 > by1:
            window[by1 - my1 : by2 - my1, bx1 - mx1 : bx2 - mx1] = 1.0

    if dilate:
        window = dilate_mask(window, dilate)

    oy, ox = y1 - my1, x1 - mx1
    return window[oy : oy + (y2 - y1), ox : ox + (x2 - x1)]


def _full_mask(det: Detection, w: int, h: int) -> np.ndarray:
    """This detection as a whole-canvas 0..1 mask, box-shaped when the detector
    gave no segmentation."""
    if det.mask is not None:
        return np.clip(_seg_mask_full(det, w, h), 0.0, 1.0)
    mask = np.zeros((h, w), dtype=np.float32)
    x1, y1, x2, y2 = det.bbox
    mask[max(0, y1) : min(h, y2), max(0, x1) : min(w, x2)] = 1.0
    return mask


def merge_detections(
    dets: list[Detection], size: tuple[int, int], invert: bool
) -> list[Detection]:
    """Fold every detection into one.

    Merge: one region covering all of them, sampled in a single pass. Two faces
    that overlap stop fighting over the same pixels on the seam.
    Merge and invert: the mask becomes everything the detector did NOT find, so
    the tab repaints the background and leaves the faces exactly as they were.
    """
    if not dets:
        return dets
    w, h = size
    union = _full_mask(dets[0], w, h)
    for det in dets[1:]:
        union = np.maximum(union, _full_mask(det, w, h))
    score = max(d.score for d in dets)

    if invert:
        union = 1.0 - union
        if not union.any():
            return []
        return [Detection(bbox=(0, 0, w, h), score=score, mask=union, label="inverted")]

    xs = [d.bbox[0] for d in dets] + [d.bbox[2] for d in dets]
    ys = [d.bbox[1] for d in dets] + [d.bbox[3] for d in dets]
    bbox = (max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys)))
    return [Detection(bbox=bbox, score=score, mask=union, label="merged")]


def enhance_one(
    image: Image.Image,
    det: Detection,
    unit: UnitArgs,
    p: Any,
    seed: int,
    index: int = 0,
    unit_index: int = 0,
) -> Image.Image:
    w, h = image.size
    dx = int(getattr(unit, "x_offset", 0) or 0)
    dy = int(getattr(unit, "y_offset", 0) or 0)

    bbox = _shift_bbox(det.bbox, dx, dy, w, h)
    mask = det.mask
    if mask is not None and (dx or dy):
        # A segmentation mask has to travel with its box, otherwise the offset
        # sliders move the crop while the mask stays behind.
        mask = shift_mask(_seg_mask_full(det, w, h), dx, dy)
    det = Detection(bbox=bbox, score=det.score, mask=mask, label=det.label)

    crop_factor = _crop_factor(unit, bbox)
    crop_region = make_crop_region((w, h), det.bbox, crop_factor)
    x1, y1, x2, y2 = crop_region
    crop = crop_image(image, crop_region)
    crop_mask = crop_region_mask((w, h), det, unit.dilate, crop_region)

    guide, max_size = _effective_guide(unit)
    ref = bbox_size(det.bbox) if unit.guide_size_for_bbox else crop.size
    tw, th, should = scale_for_guide(
        crop.size,
        ref,
        guide,
        unit.guide_size_for_bbox,
        max_size,
        unit.force_inpaint,
    )
    if not should:
        # force_inpaint off and the region is already at or above guide_size, so
        # sampling it would mean scaling it down and back up. Worth a line: from
        # the outside this is indistinguishable from the detector missing it.
        from .settings import log

        log(
            f"detection {index + 1} left alone: crop {crop.size[0]}x{crop.size[1]} "
            f"is already at guide_size and force_inpaint is off"
        )
        return image

    work = resize_lanczos(crop, (tw, th))
    work_mask_img = None
    soft_mask = False
    if unit.noise_mask:
        noise_mask = crop_mask
        feather = int(getattr(unit, "noise_mask_feather", 0) or 0)
        if feather > 0:
            # Impact Pack blurs the noise mask and lets the model blend the
            # boundary, instead of denoising hard up to a line and hiding the
            # seam with the paste mask afterwards.
            noise_mask = gaussian_feather(noise_mask, feather)
            soft_mask = True
        m = Image.fromarray((np.clip(noise_mask, 0, 1) * 255).astype(np.uint8), mode="L")
        m = m.resize((tw, th), Image.BILINEAR)
        work_mask_img = m

    pos, neg = _pick_prompts(unit, p, index, getattr(det, "label", "") or "")
    sampler, scheduler = _pick_sampler(unit, p)
    steps = unit.steps if unit.use_separate_steps else p.steps
    cfg = unit.cfg if unit.use_separate_cfg else p.cfg_scale

    # Impact Pack's `cycle`: re-sample the same region N times, each pass
    # starting from the previous result. Two passes at a low denoise usually
    # read cleaner than one pass at a high one.
    cycles = max(1, min(10, int(getattr(unit, "cycle", 1) or 1)))
    cn = _controlnet_cfg(unit, p)
    refined = work
    for c in range(cycles):
        if c and _interrupted():
            break
        refined = sample_crop(
            controlnet=cn,
            p=p,
            crop=refined,
            mask=work_mask_img,
            denoise=unit.denoise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
            prompt=pos,
            negative_prompt=neg,
            seed=seed + c,
            soft_mask=soft_mask,
        )
    shot = preview_mod.get(p)
    if shot is not None and shot.wants_crops:
        try:
            shot.add(
                _crop_sheet(
                    work, refined, work_mask_img, crop_mask, det, crop_region,
                    (tw, th), unit, unit_index, index, steps, cfg, sampler,
                    scheduler, cycles,
                )
            )
        except Exception as exc:
            print(f"[Impact ADetailer] crop preview failed: {exc}")

    if refined.size != crop.size:
        refined = resize_lanczos(refined, crop.size)
    paste_mask = gaussian_feather(crop_mask, unit.feather)
    return paste_pixel(image, refined, (x1, y1), paste_mask)


def _crop_sheet(
    work, refined, work_mask_img, crop_mask, det, crop_region, target,
    unit, unit_index, index, steps, cfg, sampler, scheduler, cycles,
):
    """The numbers that produced this crop, in the order they were applied.

    The mask drawn is whichever one the sampler actually saw: the feathered
    noise mask when 'Inpaint only masked' is on, and the paste mask otherwise -
    showing the paste mask in the first case would draw a hard edge the sampler
    never had."""
    x1, y1, x2, y2 = crop_region
    cw, ch = max(1, x2 - x1), max(1, y2 - y1)
    tw, th = target
    sx, sy = tw / float(cw), th / float(ch)

    bx1, by1, bx2, by2 = det.bbox
    bbox_in_crop = (
        int((bx1 - x1) * sx), int((by1 - y1) * sy),
        int((bx2 - x1) * sx), int((by2 - y1) * sy),
    )

    if work_mask_img is not None:
        mask = np.asarray(work_mask_img, dtype=np.float32) / 255.0
    else:
        mask = crop_mask

    dot = preview_mod.DOT
    name = (getattr(det, "label", "") or "obj").strip()
    scale = f"{cw}x{ch} -> {tw}x{th}"
    if (tw, th) == (cw, ch):
        scale += " (native)"
    caption = [
        f"tab {unit_index + 1} {dot} detection {index + 1} {dot} {name} "
        f"{det.score:.2f} {dot} box {max(1, bx2 - bx1)}x{max(1, by2 - by1)}",
        f"crop_factor {float(unit.crop_factor):.1f} {dot} crop {scale} {dot} "
        f"guide {unit.guide_size}/{unit.max_size} on the "
        f"{'bbox' if unit.guide_size_for_bbox else 'crop'}",
        f"denoise {float(unit.denoise):.2f} {dot} {steps} steps {dot} cfg {float(cfg):.1f} "
        f"{dot} {sampler} / {scheduler}"
        + (f" {dot} cycle {cycles}" if cycles > 1 else ""),
    ]
    return preview_mod.crop_sheet(work, refined, mask, bbox_in_crop, caption)


def run_unit(
    image: Image.Image,
    unit: UnitArgs,
    p: Any,
    cache: DetectorCache,
    seed: int,
    unit_index: int = 0,
) -> Image.Image:
    if not unit.enabled or unit.model in ("", "None"):
        return image
    report: dict = {}
    dets = run_detector(
        image,
        unit.model,
        unit.confidence,
        cache=cache,
        classes=getattr(unit, "classes", "") or "",
        report=report,
    )
    # Every box that does not survive, with the rule that removed it. The
    # detection sheet draws these in grey; without them a tab that filtered
    # everything away looks exactly like a detector that found nothing.
    rejected: list[tuple[Detection, str]] = list(report.get("dropped") or [])

    img_area = float(image.size[0] * image.size[1])
    drop = int(getattr(unit, "drop_size", 0) or 0)
    kept = []
    for d in dets:
        bw, bh = bbox_size(d.bbox)
        if drop and (bw < drop or bh < drop):
            # Impact Pack's drop_size. A box a few pixels across is noise, and
            # blowing it up to guide_size only ever produces a smear.
            rejected.append((d, f"drop_size {drop}"))
            continue
        ratio = (bw * bh) / max(1.0, img_area)
        if ratio < getattr(unit, "min_ratio", 0.0):
            rejected.append((d, f"area {ratio:.4f} < min"))
            continue
        if ratio > getattr(unit, "max_ratio", 1.0):
            rejected.append((d, f"area {ratio:.4f} > max"))
            continue
        kept.append(d)
    dets = kept

    if getattr(unit, "use_clip_filter", False) and getattr(unit, "clip_model", "None") not in ("", "None"):
        try:
            from .clip_filter import filter_by_clip

            # One CLIP query for the whole tab: the [SEP] sections describe what
            # to paint per box, not what the box should contain, so the first
            # section (or the tab's classes) is the closer match.
            from .detect import parse_classes

            wanted, _ = parse_classes(getattr(unit, "classes", ""))
            hint = (
                ", ".join(wanted)
                or split_prompt(unit.prompt)[0].strip()
                or image_prompt(p)
            )
            before = dets
            dets = filter_by_clip(
                image, dets, hint, unit.clip_model,
                threshold=float(getattr(unit, "clip_threshold", 0.2) or 0.2),
            )
            survived = {id(d) for d in dets}
            rejected.extend((d, "CLIP") for d in before if id(d) not in survived)
        except Exception as exc:
            print(f"[Impact ADetailer] CLIP filter failed: {exc}")

    if getattr(unit, "use_sam", False) and getattr(unit, "sam_model", "None") not in ("", "None"):
        try:
            from .sam import refine_with_sam

            dets = refine_with_sam(image, dets, unit.sam_model)
        except Exception as exc:
            print(f"[Impact ADetailer] SAM refine failed: {exc}")

    from .settings import sort_bboxes

    mode = sort_bboxes()
    if str(getattr(unit, "filter_by", "Area")).lower().startswith("conf"):
        dets.sort(key=lambda d: d.score, reverse=True)
    elif mode.startswith("Position (left"):
        dets.sort(key=lambda d: d.bbox[0])
    elif mode.startswith("Position (center"):
        cx = image.size[0] / 2.0
        cy = image.size[1] / 2.0
        dets.sort(
            key=lambda d: abs((d.bbox[0] + d.bbox[2]) / 2.0 - cx)
            + abs((d.bbox[1] + d.bbox[3]) / 2.0 - cy)
        )
    elif mode.startswith("None"):
        pass
    else:
        dets.sort(key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]), reverse=True)
    if unit.mask_k and unit.mask_k > 0:
        rejected.extend((d, f"below top {unit.mask_k}") for d in dets[unit.mask_k :])
        dets = dets[: unit.mask_k]

    merge = str(getattr(unit, "merge_mode", "None") or "None").lower()
    if merge.startswith("merge"):
        # After sorting and top-k, so "top 2 faces, merged" means what it says.
        dets = merge_detections(dets, image.size, invert="invert" in merge)

    # Resolve every detection's prompt section BEFORE anything is sampled, so
    # the sheet can show which box was going to be left alone and why. The list
    # is not filtered: a skipped detection keeps its index, because with plain
    # [SEP] that index is what decides the sections of the ones after it.
    sections = [
        section_for(unit.prompt, i, getattr(d, "label", "") or "")
        for i, d in enumerate(dets)
    ]
    skipped = {
        i: ("no [CLASS:] section" if s is None else "[SKIP]")
        for i, s in enumerate(sections)
        if s is None or is_skip(s)
    }
    _describe_sections(report, unit, dets, sections)

    shot = preview_mod.get(p)
    if shot is not None:
        try:
            dx = int(getattr(unit, "x_offset", 0) or 0)
            dy = int(getattr(unit, "y_offset", 0) or 0)
            regions = []
            for d in dets:
                shifted = _shift_bbox(d.bbox, dx, dy, image.size[0], image.size[1])
                regions.append(
                    make_crop_region(image.size, shifted, _crop_factor(unit, shifted))
                )
            shot.add(
                preview_mod.detection_sheet(
                    image, dets, rejected, unit_index, unit, regions, report, skipped
                )
            )
        except Exception as exc:
            print(f"[Impact ADetailer] detection preview failed: {exc}")

    from .modelswitch import ModelSwap

    swap = ModelSwap(p)
    try:
        if any(i not in skipped for i in range(len(dets))):
            note = swap.apply(unit)
            if note:
                from .settings import log

                log(f"tab {unit_index + 1} {note}")
        out = image
        for i, det in enumerate(dets):
            if _interrupted():
                break
            if i in skipped:
                # [SKIP] in that slot, or no [CLASS:] section claimed this class:
                # leave the detection exactly as it is.
                continue
            out = enhance_one(out, det, unit, p, seed + i, index=i, unit_index=unit_index)
        return out
    finally:
        swap.restore()


def _describe_sections(report: dict, unit: UnitArgs, dets, sections) -> None:
    """One line for the sheet saying which [CLASS:] tag caught what.

    Only written when tags are in play; a plain [SEP] prompt says nothing, since
    there is nothing there a position index does not already explain.
    """
    tagged = [class_tags(part) for part in split_prompt(unit.prompt)]
    tags = [t for group, _ in tagged for t in group]
    if not tags:
        return
    counts: dict[str, int] = {tag: 0 for tag in tags}
    counts["untagged"] = 0
    unmatched = 0
    from .detect import _matches

    for det, body in zip(dets, sections):
        if body is None:
            unmatched += 1
            continue
        label = getattr(det, "label", "") or ""
        hit = next((t for t in tags if _matches(label, t)), None)
        counts[hit if hit else "untagged"] += 1
    if not any(group for group, _ in tagged if not group):
        counts.pop("untagged", None)
    parts = [f"{name} {count}" for name, count in counts.items()]
    if unmatched:
        parts.append(f"UNCLAIMED {unmatched}")
    report["prompt_sections"] = ", ".join(parts)


def run_units(image: Image.Image, units: list[UnitArgs], p: Any) -> Image.Image:
    cache = DetectorCache()
    seed = image_seed(p)
    out = image.convert("RGB").copy()
    from .settings import same_seed

    keep = seed
    for i, unit in enumerate(units):
        if _interrupted():
            break
        out = run_unit(out, unit, p, cache, seed, unit_index=i)
        if not same_seed():
            seed = keep + 17
            keep = seed
    return out
