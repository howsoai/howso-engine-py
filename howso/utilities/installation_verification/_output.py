from __future__ import annotations

import os
import re
from typing import Any

from rich import get_console, print as rich_print


def is_databricks() -> bool:
    """Check environment is on Databricks."""
    return bool(os.environ.get("DATABRICKS_RUNTIME_VERSION", None))


def _console_safe(text: str) -> str:
    """Swap the trademark sign for "(tm)" where the console cannot show it.

    Messages are written with the real sign; this downgrades it at output time
    so every one of them is covered without each having to think about it. The
    legacy Windows console renders it badly even where the encoding accepts
    it, and cp437 -- a common console code page -- cannot encode it at all.
    """
    if "™" not in text:
        return text
    try:
        console = get_console()
        if console.legacy_windows:
            return text.replace("™", "(tm)")
        "™".encode(console.encoding or "ascii")
    except Exception:  # noqa: BLE001
        return text.replace("™", "(tm)")
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
