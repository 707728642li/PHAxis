#!/usr/bin/env python
"""Verify a PHAxis 1.0.0 reuse handover package without runtime imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))
from handover_package_common import HandoverError, verify_handover_package


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    try:
        result = verify_handover_package(args.package)
    except HandoverError as error:
        print(f"PHAxis handover verification failed: {error}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
