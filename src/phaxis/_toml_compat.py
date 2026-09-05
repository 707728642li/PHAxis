"""One deterministic TOML backend across ordinary and no-site execution.

Python 3.11+ uses :mod:`tomllib`.  Python 3.10 always uses the byte-identical,
MIT-licensed Tomli 2.4.0 source vendored with PHAxis.  Deliberately avoiding a
site-package-first branch prevents parser differentials between ordinary CI
and the source verifier's mandatory ``python -S`` isolation probes.
"""

from __future__ import annotations

try:
    import tomllib as _backend
except ModuleNotFoundError as error:
    if error.name != "tomllib":
        raise
    from ._vendor import tomli as _backend

loads = _backend.loads
load = _backend.load
TOMLDecodeError = _backend.TOMLDecodeError
BACKEND_NAME = _backend.__name__

__all__ = ("BACKEND_NAME", "TOMLDecodeError", "load", "loads")
