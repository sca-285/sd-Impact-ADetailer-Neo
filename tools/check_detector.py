"""Is this detector actually doing what you asked?

Run it from the WebUI's own Python, with the WebUI's own env:

    cd <webui>
    python extensions/impact-adetailer/tools/check_detector.py \
        "E:/models/adetailer/yolov8s-worldv2.pt" --classes "face, hand" \
        --image outputs/txt2img-images/2026-01-01/00001.png

It answers three questions no log line answers on its own:

  1. Which ultralytics class did YOLO() build? Open-vocabulary routing is done
     on the FILE NAME - "-world" or "yoloe" in the stem - so a renamed
     checkpoint loads as a plain detector that still returns boxes.
  2. Did set_classes actually take? It needs a text encoder that ultralytics
     downloads on first use, and it fails loudly here instead of silently
     falling back to the model's own class list.
  3. What score does each of your terms get on a real image? That is the number
     to set the confidence threshold from.

Nothing here writes to the model or to your settings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="path to the .pt detector")
    ap.add_argument("--classes", default="", help='comma separated, e.g. "face, hand"')
    ap.add_argument("--image", default="", help="an image to run on (optional)")
    ap.add_argument("--conf", type=float, default=0.05,
                    help="threshold for the trial run; keep it low, the point is "
                         "to see the scores (default 0.05)")
    args = ap.parse_args()

    path = Path(args.model).expanduser()
    if not path.is_file():
        print(f"not a file: {path}")
        return 2

    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is not installed in this Python. Run this with the "
              "WebUI's interpreter, not the system one.")
        return 2

    print(f"ultralytics {ultralytics.__version__}")
    print(f"file        {path.name}   (stem: {path.stem})")

    model = YOLO(str(path))
    kind = type(model).__name__
    open_vocab = callable(getattr(model, "set_classes", None))
    print(f"loaded as   {kind}   open-vocabulary: {open_vocab}")

    hint = "world" in path.stem.lower() or "yoloe" in path.stem.lower()
    if hint and not open_vocab:
        print("  ^^ the name looks open-vocabulary but the class is not. "
              'ultralytics routes on "-world" / "yoloe" in the stem; rename the '
              "file and it will route correctly.")

    names = getattr(model, "names", {}) or {}
    values = list(names.values()) if isinstance(names, dict) else list(names)
    print(f"default vocabulary: {len(values)} classes")
    if values:
        print("  " + ", ".join(values[:20]) + (" ..." if len(values) > 20 else ""))

    terms = [t.strip() for t in args.classes.replace("\n", ",").split(",") if t.strip()]
    if terms and open_vocab:
        print(f"\napplying classes: {terms}")
        try:
            try:
                model.set_classes(list(terms))
            except TypeError:
                # YOLOE before 8.4.0 wants the embeddings passed in.
                model.set_classes(list(terms), model.get_text_pe(list(terms)))
            print("  set_classes OK — the detector head was rebuilt around these "
                  "words")
        except Exception as exc:
            print(f"  set_classes FAILED: {type(exc).__name__}: {exc}")
            print("  The model will detect its OWN default classes instead.")
            print("  It needs a text encoder (CLIP for -world, MobileCLIP for "
                  "yoloe); the first call downloads one, so allow it out once.")
            return 1
    elif terms:
        print(f"\nfixed-class model: {terms} can only filter the list above")

    if not args.image:
        print("\n(no --image given, stopping here)")
        return 0

    from PIL import Image
    import numpy as np

    image = Image.open(args.image).convert("RGB")
    print(f"\nrunning on {args.image}  {image.size}  conf >= {args.conf}")
    results = model.predict(np.array(image), conf=args.conf, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print("  nothing found at all")
        return 0

    res = results[0]
    res_names = getattr(res, "names", names)
    lookup = (res_names if isinstance(res_names, dict)
              else {i: v for i, v in enumerate(res_names)})
    rows = []
    for i in range(len(res.boxes)):
        label = ""
        if res.boxes.cls is not None:
            label = lookup.get(int(res.boxes.cls[i]), "")
        score = float(res.boxes.conf[i]) if res.boxes.conf is not None else 1.0
        x1, y1, x2, y2 = (int(v) for v in res.boxes.xyxy[i])
        rows.append((score, label, x2 - x1, y2 - y1))
    rows.sort(reverse=True)
    for score, label, bw, bh in rows:
        print(f"  {score:5.2f}  {label or '?':20}  {bw}x{bh}px")
    print(f"\nSet the tab's confidence just under the lowest score you want to "
          f"keep. Highest here: {rows[0][0]:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
