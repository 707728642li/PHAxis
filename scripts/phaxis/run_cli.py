"""Run the PHAxis CLI directly from the repository ``src`` layout.

Formal release stages must not depend on PHAxis already being installed in the
calling conda environment or on an inherited ``PYTHONPATH``.  This wrapper
adds only this checkout's resolved ``src`` directory before importing the
public CLI, then forwards argv unchanged.
"""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = (PROJECT_ROOT / "src").resolve()
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from phaxis.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
