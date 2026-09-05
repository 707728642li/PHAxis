#!/usr/bin/env python
"""Lock the three-stage 283-image root equivalence target without labels/GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.root_provider import build_reference_registry  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--v20-root", type=Path, required=True)
    parser.add_argument("--final-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_reference_registry(
        v1_root=args.v1_root,
        v20_root=args.v20_root,
        final_root=args.final_root,
        output=args.output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "gates": result["gates"],
                "reference_identity_sha256": result[
                    "reference_identity_sha256"
                ],
                "claim_boundary": result["claim_boundary"],
                "blind_images_used": result["blind_images_used"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
