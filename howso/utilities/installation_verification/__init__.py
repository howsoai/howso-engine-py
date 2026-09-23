"""Verify that a Howso installation works on this machine.

The supported way to run this is the ``verify_howso_install`` command, or
equivalently ``python -m howso.utilities.installation_verification``. Both
call :func:`main`, which is the whole of this package's interface.

Everything else lives in submodules and is free to move between them. Import
from those directly only if you are working on this package.
"""
from __future__ import annotations

from .main import main

__all__ = ["main"]
