"""Process images: what the detector found, and what each crop looked like.

Impact Pack shows you this for free, because every stage is a node you can hang
a preview off. A WebUI script has no such wiring: the detail pass happens inside
one postprocess_image call and the only thing that comes back out is the
finished picture. When a face comes back wrong there is nothing to look at and
no way to tell a bad detection from a bad sample.

So the stages draw themselves. Two kinds of image are produced:

  * one detection sheet per tab - every box the detector returned, kept ones in
    blue with their class and confidence, rejected ones in grey with the rule
    that rejected them, segmentation masks tinted over the top;
  * one crop sheet per detection - the pixels exactly as they were handed to the
    sampler, next to what came back, with the numbers that produced them.

They are appended to p.extra_result_images, which every fork in this family
concatenates onto the gallery after the real results (txt2img.py / img2img.py:
`processed.images + processed.extra_images`). Nothing is written to disk and
nothing touches the image being generated.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageDraw

OFF = "Off"
DETECTION = "Detection"
CROPS = "Detection + crops"
MODES = [OFF, DETECTION, CROPS]

KEPT = (61, 165, 255)          # the blue Impact Pack draws accepted boxes in
DROPPED = (145, 145, 150)
MASK_TINT = (196, 92, 255)     # purple, over the segmentation
MASK_ALPHA = 0.38
CROP_BOX = (255, 196, 61)      # the crop region, on the detection sheet
INK = (242, 242, 245)
PAPER = (26, 26, 30)

# Roboto-Regular.ttf is what the WebUI ships and what get_font returns, and it
# covers Latin-1 but not the arrows and maths operators. A missing glyph draws
# as a hollow box, so the captions stay inside what the font actually has: the
# middle dot is U+00B7, "->" and "x" are what they look like.
DOT = "·"


# --------------------------------------------------------------- collector


class Preview:
    """Somewhere for the pipeline to put its sheets.

    Lives on `p` for the duration of one postprocess_image call, so a batch
    cannot mix its images up and nothing survives into the next generation.
    """

    __slots__ = ("mode", "images", "_unit", "_capped")

    def __init__(self, mode: str):
        self.mode = mode if mode in MODES else OFF
        self.images: list[Image.Image] = []
        self._unit = 0
        self._capped = False

    @property
    def on(self) -> bool:
        return self.mode != OFF

    @property
    def wants_crops(self) -> bool:
        return self.mode == CROPS

    def begin_unit(self, index: int) -> None:
        self._unit = index

    def add(self, image: Image.Image) -> None:
        if image is None:
            return
        try:
            from .settings import max_preview_sheets

            cap = max_preview_sheets()
        except Exception:
            cap = 16
        if cap and len(self.images) >= cap:
            if not getattr(self, "_capped", False):
                self._capped = True
                print(
                    f"[Impact ADetailer] process-image sheets capped at {cap}. "
                    "Raise Settings → Impact ADetailer → Max process-image sheets "
                    "if you need the rest."
                )
            return
        self.images.append(image)
        # Also put it under the progress bar. The detail pass can run for a
        # minute on a batch and the live preview otherwise shows the last
        # sampler step of the base image the whole time, which tells you
        # nothing about what the detector is doing.
        try:
            from modules import shared

            shared.state.assign_current_image(image)
        except Exception:
            pass


def attach(p: Any, mode: str) -> Preview:
    preview = Preview(mode)
    try:
        p.iad_preview = preview
    except Exception:
        pass
    return preview


def get(p: Any) -> Preview | None:
    preview = getattr(p, "iad_preview", None)
    return preview if isinstance(preview, Preview) and preview.on else None


def detach(p: Any) -> None:
    try:
        delattr(p, "iad_preview")
    except Exception:
        pass


# ------------------------------------------------------------------ paint


def _font(size: int):
    try:
        from modules.images import get_font

        return get_font(size)
    except Exception:
        pass
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)     # Pillow >= 10.1
    except TypeError:
        from PIL import ImageFont as _IF

        return _IF.load_default()


def _metrics(size: tuple[int, int]) -> tuple[int, int]:
    """(line width, font size) that stay readable on a 512 and on a 2048."""
    short = max(1, min(size))
    width = max(2, int(round(short / 420.0)))
    font = int(round(short / 38.0))
    return width, max(13, min(34, font))


def _text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    """Width of this string, but the height of a full line.

    textbbox measures the ink, so "returned" comes back shorter than "fed to
    sampler" purely because it has no descender. Tags sized that way sit at
    different heights next to each other and the caption strip gets uneven
    line spacing, so the height always comes from the same reference string."""
    try:
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        width = right - left
        _, ref_top, _, ref_bottom = draw.textbbox((0, 0), "Ag", font=font)
        return width, ref_bottom - ref_top
    except AttributeError:      # Pillow < 8
        width, _ = draw.textsize(text, font=font)
        return width, draw.textsize("Ag", font=font)[1]


def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _label(draw, xy, text, font, colour, bounds=None, placed=None) -> None:
    """A filled tag with the text on it.

    Flipped inside the box when there is no room above it, and pulled back
    inside `bounds` when it would otherwise run off the edge - a box near the
    right margin is exactly where the confidence is worth reading.

    `placed` collects the tags drawn so far. Two detections that overlap put
    their tags in the same place and the second one buries the first, which on
    a crowded frame is precisely where you need to read both; passing the list
    steps a colliding tag down until it is clear.
    """
    x, y = xy
    tw, th = _text_size(draw, text, font)
    pad = max(2, th // 5)
    box_w, box_h = tw + pad * 2, th + pad * 2
    top = y - box_h
    if top < 0:
        top = y
    x, top = int(x), int(top)

    def clamp(bx, by):
        if bounds is None:
            return bx, by
        limit_w, limit_h = bounds
        return (max(0, min(bx, limit_w - box_w)), max(0, min(by, limit_h - box_h)))

    x, top = clamp(x, top)
    if placed is not None:
        for _ in range(8):
            rect = (x, top, x + box_w, top + box_h)
            if not any(_overlaps(rect, other) for other in placed):
                break
            x, top = clamp(x, top + box_h + 1)
        placed.append((x, top, x + box_w, top + box_h))

    draw.rectangle([x, top, x + box_w, top + box_h], fill=colour)
    draw.text((x + pad, top + pad), text, font=font, fill=(16, 16, 20))


def _mask_edge(mask: np.ndarray) -> np.ndarray:
    """The one-pixel outline of a 0..1 mask, without needing opencv."""
    solid = mask > 0.5
    if not solid.any():
        return solid
    inner = solid.copy()
    inner[1:, :] &= solid[:-1, :]
    inner[:-1, :] &= solid[1:, :]
    inner[:, 1:] &= solid[:, :-1]
    inner[:, :-1] &= solid[:, 1:]
    return solid & ~inner


def _tint(arr: np.ndarray, mask: np.ndarray, colour, alpha: float) -> np.ndarray:
    if mask is None:
        return arr
    m = np.clip(mask, 0.0, 1.0)
    if m.shape[:2] != arr.shape[:2]:
        img = Image.fromarray((m * 255).astype(np.uint8), mode="L")
        img = img.resize((arr.shape[1], arr.shape[0]), Image.BILINEAR)
        m = np.asarray(img, dtype=np.float32) / 255.0
    if not m.any():
        return arr
    a = (m * float(alpha))[..., None]
    tint = np.array(colour, dtype=np.float32)
    return arr * (1.0 - a) + tint * a


def _caption(image: Image.Image, lines: list[str]) -> Image.Image:
    """Bolt a dark strip with `lines` onto the bottom of the image."""
    if not lines:
        return image
    _, font_size = _metrics(image.size)
    font = _font(font_size)
    probe = ImageDraw.Draw(image)
    heights = []
    width_needed = image.width
    for line in lines:
        tw, th = _text_size(probe, line, font)
        heights.append(th)
        width_needed = max(width_needed, tw + font_size)
    pad = max(4, font_size // 3)
    strip = sum(heights) + pad * (len(lines) + 1)

    out = Image.new("RGB", (width_needed, image.height + strip), PAPER)
    out.paste(image, (0, 0))
    draw = ImageDraw.Draw(out)
    y = image.height + pad
    for line, th in zip(lines, heights):
        draw.text((pad * 2, y), line, font=font, fill=INK)
        y += th + pad
    return out


# ------------------------------------------------------- detection sheet


def detection_sheet(
    image: Image.Image,
    kept: list,
    dropped: list[tuple[Any, str]],
    unit_index: int,
    unit,
    crop_regions: list[tuple[int, int, int, int]] | None = None,
    report: dict | None = None,
    skipped: dict[int, str] | None = None,
) -> Image.Image:
    """Every box the detector returned, over the image it ran on.

    `kept` is in final order - what the numbers on the boxes mean is the order
    the tab will process them in, which is the thing sort-by and top-k actually
    control. `dropped` carries the rule that rejected each box, so a tab that
    found nothing says why instead of just doing nothing.
    """
    w, h = image.size
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    edges = np.zeros((h, w), dtype=bool)
    for det in kept:
        mask = getattr(det, "mask", None)
        if mask is None:
            continue
        base = _tint(base, mask, MASK_TINT, MASK_ALPHA)
        arr = np.asarray(mask, dtype=np.float32)
        if arr.shape[:2] == (h, w):
            edges |= _mask_edge(arr)
    pixels = np.clip(base, 0, 255).astype(np.uint8)
    if edges.any():
        pixels[edges] = MASK_TINT
    canvas = Image.fromarray(pixels, mode="RGB")

    draw = ImageDraw.Draw(canvas)
    line, font_size = _metrics(canvas.size)
    font = _font(font_size)

    for region in crop_regions or []:
        draw.rectangle(list(region), outline=CROP_BOX, width=max(1, line - 1))

    placed: list[tuple[int, int, int, int]] = []
    for det, reason in dropped:
        draw.rectangle(list(det.bbox), outline=DROPPED, width=max(1, line - 1))
        _label(draw, (det.bbox[0], det.bbox[1]), f"{det.score:.2f} {DOT} {reason}",
               font, DROPPED, (w, h), placed)

    skipped = skipped or {}
    for i, det in enumerate(kept):
        # A skipped detection keeps its number. That number is the [SEP] index,
        # which is what decides the sections of every detection after it, so
        # renumbering around the gap would make the sheet lie about the ones
        # that are NOT skipped.
        why = skipped.get(i)
        colour = DROPPED if why else KEPT
        draw.rectangle(list(det.bbox), outline=colour,
                       width=max(1, line - 1) if why else line)
        name = (getattr(det, "label", "") or "obj").strip()
        text = f"{i + 1}. {name} {det.score:.2f}" + (f" {DOT} {why}" if why else "")
        _label(draw, (det.bbox[0], det.bbox[1]), text, font, colour, (w, h), placed)

    found = len(kept) + len(dropped)
    detailed = len(kept) - len(skipped)
    head = (
        f"tab {unit_index + 1} {DOT} {getattr(unit, 'model', '?')} {DOT} "
        f"confidence >= {float(getattr(unit, 'confidence', 0)):.2f} {DOT} "
        f"{detailed} of {found} detailed"
        + (f" ({len(skipped)} skipped)" if skipped else "")
    )
    second = (
        f"order: {getattr(unit, 'filter_by', 'Area')} {DOT} "
        f"top k: {getattr(unit, 'mask_k', 0) or 'all'} {DOT} "
        f"crop_factor {float(getattr(unit, 'crop_factor', 3.0)):.1f} {DOT} "
        f"dilate {int(getattr(unit, 'dilate', 0))} {DOT} "
        f"denoise {float(getattr(unit, 'denoise', 0)):.2f}"
    )
    return _caption(canvas, [head, second, *_class_line(report)])


# What the class filter did, in one line. This exists because from the outside
# an open-vocabulary model that ignored your words is indistinguishable from one
# that took them: both return boxes, both look busy, and the difference only
# shows up in what gets detailed.
_MODE_TEXT = {
    "open-vocabulary": "the detector was rebuilt around these words",
    "name filter": "matched against this model's fixed class list",
    "not in vocabulary": "THIS MODEL HAS NO SUCH CLASS - nothing kept",
}


def _class_line(report: dict | None) -> list[str]:
    if not report:
        return []
    lines = []
    mode = report.get("mode", "none")
    terms = report.get("terms") or []
    if mode != "none" and terms:
        line = f"classes: {', '.join(terms)} {DOT} {mode} {DOT} {_MODE_TEXT.get(mode, '')}"
        unknown = report.get("unknown") or []
        if unknown and mode == "name filter":
            line += f" {DOT} unmatched: {', '.join(unknown)}"
        lines.append(line)
    sections = report.get("prompt_sections")
    if sections:
        # UNCLAIMED is the one to read: a [CLASS:] tag that matches nothing the
        # detector returned is almost always a class name typed from memory.
        lines.append(f"[CLASS:] sections: {sections}")
    return lines


# ------------------------------------------------------------ crop sheet


def crop_sheet(
    fed: Image.Image,
    result: Image.Image,
    mask: np.ndarray | None,
    bbox_in_crop: tuple[int, int, int, int] | None,
    caption: list[str],
) -> Image.Image:
    """The crop as the sampler received it, beside what the sampler returned.

    Drawing the bbox inside the crop is the point of this sheet: it shows where
    the detection sits in the window crop_factor opened around it, which is the
    only way to see that a crop is off-centre or clipped by the image edge
    before wondering why the paste has a seam.
    """
    left = fed.convert("RGB").copy()
    if mask is not None:
        arr = np.asarray(left, dtype=np.float32)
        arr = _tint(arr, mask, MASK_TINT, 0.25)
        left = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
        m = np.asarray(mask, dtype=np.float32)
        if m.shape[:2] != (left.height, left.width):
            img = Image.fromarray((np.clip(m, 0, 1) * 255).astype(np.uint8), mode="L")
            m = np.asarray(img.resize(left.size, Image.BILINEAR), dtype=np.float32) / 255.0
        edge = _mask_edge(m)
        if edge.any():
            pixels = np.asarray(left, dtype=np.uint8).copy()
            pixels[edge] = MASK_TINT
            left = Image.fromarray(pixels, mode="RGB")

    line, font_size = _metrics(left.size)
    if bbox_in_crop is not None:
        ImageDraw.Draw(left).rectangle(list(bbox_in_crop), outline=KEPT, width=line)

    right = result.convert("RGB")
    if right.size != left.size:
        right = right.resize(left.size, Image.Resampling.LANCZOS)

    gap = max(4, line * 2)
    sheet = Image.new("RGB", (left.width * 2 + gap, left.height), PAPER)
    sheet.paste(left, (0, 0))
    sheet.paste(right, (left.width + gap, 0))

    draw = ImageDraw.Draw(sheet)
    font = _font(font_size)
    # y = gap, so _label's "no room above" branch drops each tag just inside the
    # top corner of its own half rather than hanging off the edge.
    _label(draw, (gap, gap), "fed to sampler", font, KEPT, sheet.size)
    _label(draw, (left.width + gap * 2, gap), "returned", font, CROP_BOX, sheet.size)
    return _caption(sheet, caption)
