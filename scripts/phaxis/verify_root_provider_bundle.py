#!/usr/bin/env python
"""Verify every byte in a materialized PHAxis root-provider model bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.root_provider import verify_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
