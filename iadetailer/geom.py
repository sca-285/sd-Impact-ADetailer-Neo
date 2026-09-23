from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageFilter

try:  # opencv is fast, but the extension must still load without it
    import cv2
except Exception:  # pragma: no cover - exercised only on installs without cv2
    cv2 = None


def bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return max(1, x2 - x1), max(1, y2 - y1)


def make_crop_region(
    image_size: tuple[int, int],
    bbox: tuple[int, int, int, int],
    crop_factor: float,
) -> tuple[int, int, int, int]:
    """Impact-style crop: expand bbox by crop_factor around its center, clip to image."""
    w, h = image_size
    x1, y1, x2, y2 = bbox
    bw, bh = bbox_size(bbox)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    factor = max(1.0, float(crop_factor))
    nw = bw * factor
    nh = bh * factor
    nx1 = int(math.floor(cx - nw / 2.0))
    ny1 = int(math.floor(cy - nh / 2.0))
    nx2 = int(math.ceil(cx + nw / 2.0))
    ny2 = int(math.ceil(cy + nh / 2.0))
    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)
    if nx2 <= nx1:
        nx2 = min(w, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(h, ny1 + 1)
    return nx1, ny1, nx2, ny2


def scale_for_guide(
    crop_wh: tuple[int, int],
    bbox_wh: tuple[int, int],
    guide_size: float,
    guide_size_for_bbox: bool,
    max_size: float,
    force_inpaint: bool,
) -> tuple[int, int, bool]:
    """
    Return (target_w, target_h, should_run).

    Step for step with Impact Pack's enhance_detail, ORDER INCLUDED:

        upscale = guide_size / shorter side of the bbox (or of the crop)
        new     = crop * upscale
        if new > max_size:  upscale *= max_size / longest ; new = crop * upscale
        if upscale <= 1:
            force_inpaint off -> skip the region
            force_inpaint on  -> upscale = 1, new = the crop's NATIVE size

    Two consequences that are easy to get wrong:

    * The region is never shrunk. Sampling a crop below its own resolution
      means scaling it back up on the way out, and that is exactly the detail
      the detailer was supposed to add.
    * Because the cap is tested before the decision, max_size limits how far a
      region may be blown UP. It is not a ceiling on native size: a crop larger
      than max_size is sampled at its own size when force_inpaint is on. That
      is Impact Pack's behaviour and it is what makes large faces hold up.
    """
    cw, ch = crop_wh
    ref_w, ref_h = bbox_wh if guide_size_for_bbox else crop_wh
    short = max(1, min(ref_w, ref_h))

    upscale = float(guide_size) / float(short)
    new_w = cw * upscale
    new_h = ch * upscale

    longest = max(new_w, new_h)
    if longest > max_size:
        upscale *= float(max_size) / longest
        new_w = cw * upscale
        new_h = ch * upscale

    if upscale <= 1.0 or new_w < 1.0 or new_h < 1.0:
        if not force_inpaint:
            return cw, ch, False
        upscale = 1.0
        new_w, new_h = float(cw), float(ch)

    tw = max(64, int(round(new_w / 8.0)) * 8)
    th = max(64, int(round(new_h / 8.0)) * 8)
    return tw, th, True


def crop_image(image: Image.Image, region: tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = region
    return image.crop((x1, y1, x2, y2))


def resize_lanczos(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, Image.Resampling.LANCZOS)


def _to_u8(mask: np.ndarray) -> np.ndarray:
    # Truncating rather than rounding, so mask values stay bit-for-bit what the
    # previous releases produced.
    return (np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8)


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels == 0:
        return mask
    k = abs(int(pixels))
    u8 = _to_u8(mask)
    if cv2 is not None:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k * 2 + 1, k * 2 + 1))
        u8 = cv2.dilate(u8, kernel, 1) if pixels > 0 else cv2.erode(u8, kernel, 1)
    else:
        im = Image.fromarray(u8, mode="L")
        flt = ImageFilter.MaxFilter if pixels > 0 else ImageFilter.MinFilter
        # PIL caps the window at 9, so apply it in passes.
        left = k
        while left > 0:
            step = min(4, left)
            im = im.filter(flt(step * 2 + 1))
            left -= step
        u8 = np.asarray(im)
    return u8.astype(np.float32) / 255.0


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate a mask, filling the vacated edge with zeros.
    dy follows the ADetailer convention: positive moves the mask up."""
    if not dx and not dy:
        return mask
    out = np.zeros_like(mask)
    h, w = mask.shape[:2]
    sy = -int(dy)  # screen coordinates grow downwards
    sx = int(dx)
    src_y1, src_y2 = max(0, -sy), min(h, h - sy)
    src_x1, src_x2 = max(0, -sx), min(w, w - sx)
    if src_y2 <= src_y1 or src_x2 <= src_x1:
        return out
    out[src_y1 + sy : src_y2 + sy, src_x1 + sx : src_x2 + sx] = mask[src_y1:src_y2, src_x1:src_x2]
    return out


def bbox_mask(h: int, w: int, bbox: tuple[int, int, int, int]) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.float32)
    x1, y1, x2, y2 = bbox
    m[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)] = 1.0
    return m


def gaussian_feather(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask
    u8 = _to_u8(mask)
    if cv2 is not None:
        k = radius * 2 + 1
        u8 = cv2.GaussianBlur(u8, (k, k), sigmaX=radius * 0.5)
    else:
        u8 = np.asarray(
            Image.fromarray(u8, mode="L").filter(ImageFilter.GaussianBlur(radius * 0.5))
        )
    return u8.astype(np.float32) / 255.0


def paste_pixel(
    canvas: Image.Image,
    patch: Image.Image,
    origin: tuple[int, int],
    mask: np.ndarray,
) -> Image.Image:
    """Impact-style tensor_paste analogue.

    Only the overlapping rectangle is promoted to float, so a small face on a
    large canvas no longer costs a full-image float32 round trip per detection.
    """
    x, y = origin
    cw, ch = canvas.size
    pw, ph = patch.size

    if mask.shape[:2] != (ph, pw):
        mask_img = Image.fromarray(_to_u8(mask), mode="L").resize((pw, ph), Image.BILINEAR)
        mask = np.asarray(mask_img).astype(np.float32) / 255.0

    x2 = min(cw, x + pw)
    y2 = min(ch, y + ph)
    cx1 = max(0, x)
    cy1 = max(0, y)
    if x2 <= cx1 or y2 <= cy1:
        return canvas

    px1 = cx1 - x
    py1 = cy1 - y
    sl_w = x2 - cx1
    sl_h = y2 - cy1

    out = canvas.convert("RGB") if canvas.mode != "RGB" else canvas.copy()
    dest = np.asarray(out.crop((cx1, cy1, x2, y2)), dtype=np.float32)
    src = np.asarray(patch.convert("RGB"), dtype=np.float32)[py1 : py1 + sl_h, px1 : px1 + sl_w]
    alpha = np.clip(mask[py1 : py1 + sl_h, px1 : px1 + sl_w], 0.0, 1.0)[..., None]

    blended = dest * (1.0 - alpha) + src * alpha
    out.paste(
        Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB"),
        (cx1, cy1),
    )
    return out
