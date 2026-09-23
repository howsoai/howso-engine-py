from __future__ import annotations

import os
import re
from typing import Any

from rich import get_console, print as rich_print

#: Non-ASCII marks that appear in messages, with their ASCII downgrades.
CHAR_MAP = {
    "™": "(tm)",
    "®": "(R)",
}


def is_databricks() -> bool:
    """Check environment is on Databricks."""
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION", None))


def _console_safe(text: str) -> str:
    """Downgrade the trademark marks where the console cannot show them.

    Messages are written with the real signs; this downgrades them at output
    time so every one of them is covered without each having to think about
    it. The two checks cover different failures: an encoding that cannot hold
    a mark, and the legacy Windows console, which reports utf-8 even under a
    code page whose font cannot draw the glyph. Each mark is probed on its own
    because they are not carried by the same set of codecs -- latin-1 holds
    "(R)" but not "(tm)".
    """
    def downgrade(value: str) -> str:
        """Replace every mark with its ASCII stand-in."""
        for char, fallback in CHAR_MAP.items():
            value = value.replace(char, fallback)
        return value

    if not any(char in text for char in CHAR_MAP):
        return text
    try:
        console = get_console()
        if console.legacy_windows:
            return downgrade(text)
        for char, fallback in CHAR_MAP.items():
            if char in text:
                try:
                    char.encode(console.encoding)
                except UnicodeEncodeError:
                    text = text.replace(char, fallback)
    except Exception:  # noqa: BLE001
        return downgrade(text)
    return text


def iv_print(*args: Any, **kwargs: Any) -> None:
    """Print wrapper for handling prints in different environments."""
    args = tuple(_console_safe(a) if isinstance(a, str) else a for a in args)
    if is_databricks():
        # strip out rich formatting before printing
        a_list = list(args)
        a_list[0] = re.sub(r"\[.*\]", "", a_list[0])
        args = tuple(a_list)
        print(*args, **kwargs)
    else:
        rich_print(*args, **kwargs)
