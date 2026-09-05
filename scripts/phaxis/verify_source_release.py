#!/usr/bin/env python
"""Verify a PHAxis source-release tree without importing PHAxis runtime code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# Verification must be observational.  Without this guard, importing the
# verifier from a writable release checkout can create ``__pycache__`` before
# the exact manifest closure is inspected and make a pristine tree fail itself.
sys.dont_write_bytecode = True


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_ROOT), str(PROJECT_ROOT / "src")]

from source_release_common import SourceReleaseError, verify_source_release  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    parser.add_argument(
        "--project-root",
        type=Path,
        help="also prove every copied byte is current relative to project authority",
    )
    args = parser.parse_args()
    try:
        result = verify_source_release(
            args.release_root,
            project_root=args.project_root,
        )
    except SourceReleaseError as error:
        print(f"PHAxis source-release verification failed: {error}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
