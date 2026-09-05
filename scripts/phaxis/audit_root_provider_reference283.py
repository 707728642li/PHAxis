#!/usr/bin/env python
"""Audit one fresh portable raw-image rerun against the three locked layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.root_provider.reference import audit_fresh_reference  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-registry", type=Path, required=True)
    parser.add_argument("--fresh-v1-root", type=Path, required=True)
    parser.add_argument("--fresh-v20-root", type=Path, required=True)
    parser.add_argument("--fresh-final-root", type=Path, required=True)
    parser.add_argument("--pipeline-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_fresh_reference(
        reference_registry=args.reference_registry,
        fresh_v1_root=args.fresh_v1_root,
        fresh_v20_root=args.fresh_v20_root,
        fresh_final_root=args.fresh_final_root,
        pipeline_state=args.pipeline_state,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "pass_exact_283":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
