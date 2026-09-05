#!/usr/bin/env python
"""Build a formal PHAxis 1.0.0 reuse handover package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))
from handover_package_common import (
    HandoverError,
    build_handover_package,
    inspect_handover_contract,
)


PROJECT_ROOT = SCRIPT_ROOT.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an atomic, deterministic PHAxis 1.0.0 handover package")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="validate every formal binding without copying package payloads",
    )
    args = parser.parse_args()
    try:
        if args.check_only:
            report = inspect_handover_contract(args.project_root, args.contract)
            print(
                json.dumps(
                    {
                        "status": report["status"],
                        "checks": report["checks"],
                        "contract_identity_sha256": report["contract"][
                            "contract_identity_sha256"
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.output is None:
            parser.error("--output is required unless --check-only is used")
        manifest = build_handover_package(project_root=args.project_root, contract_path=args.contract, output=args.output)
    except HandoverError as error:
        print(f"PHAxis handover build blocked: {error}")
        return 2
    print(json.dumps({"status": "built", "output": str(args.output.resolve()), "tree_identity_sha256": manifest["tree_identity_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
