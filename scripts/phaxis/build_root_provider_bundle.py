#!/usr/bin/env python
"""Build or plan the external, hash-locked PHAxis root-provider model bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.root_provider import build_bundle  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    result = build_bundle(
        args.project_root.resolve(),
        args.output.resolve(),
        mode=args.mode,
        plan_only=args.plan_only,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "bundle_id",
                    "files_count",
                    "bytes",
                    "root_effect_slice_files",
                    "bundle_identity_sha256",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
