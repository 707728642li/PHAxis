#!/usr/bin/env python3
"""Build or check the complete PHAxis/v1 same-hardware benchmark inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from handover_manifest_producers import ProducerError, build_benchmark_manifest, publish_report_no_overwrite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--same-hardware-receipt", type=Path, required=True)
    parser.add_argument("--artifact-inventory", type=Path, required=True)
    parser.add_argument("--release-attestation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        report = build_benchmark_manifest(
            project_root=args.project_root,
            same_hardware_receipt=args.same_hardware_receipt,
            artifact_inventory=args.artifact_inventory,
            release_attestation=args.release_attestation,
            output=args.output,
            execute=args.execute,
        )
    except ProducerError as error:
        parser.error(str(error))
    if args.receipt is not None:
        if not args.execute:
            parser.error("--receipt requires --execute")
        try:
            publish_report_no_overwrite(project_root=args.project_root, report=report, receipt=args.receipt, rollback_outputs=(args.output,))
        except ProducerError as error:
            parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
