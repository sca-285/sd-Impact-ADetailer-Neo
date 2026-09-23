from __future__ import annotations

import argparse

# Unique to this extension. dest becomes cmd_opts.iad_no_huggingface
_FLAG = "--iad-no-huggingface"
_DEST = "iad_no_huggingface"


def preload(parser: argparse.ArgumentParser) -> None:
    already = {
        opt
        for action in getattr(parser, "_actions", [])
        for opt in getattr(action, "option_strings", [])
    }
    if _FLAG in already:
        return

    try:
        parser.add_argument(
            _FLAG,
            action="store_true",
            dest=_DEST,
            default=False,
            help=(
                "Impact ADetailer: skip automatic download of detector / CLIP "
                "weights from the internet"
            ),
        )
    except argparse.ArgumentError:
        # Another copy of this same extension already registered the flag.
        pass
