"""PHAxis: unified Arabidopsis primary-root and root-hair phenotyping.

The package import is intentionally dependency-light.  Runtime-heavy modules
are loaded only when their public objects are requested, which also lets the
source-release verifier run before third-party dependencies are installed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["fuse_hybrid_root_with_stageb_hairs"]
__version__ = "1.0.0"


def __getattr__(name: str) -> Any:
    if name == "fuse_hybrid_root_with_stageb_hairs":
        from .fusion import fuse_hybrid_root_with_stageb_hairs

        return fuse_hybrid_root_with_stageb_hairs
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
