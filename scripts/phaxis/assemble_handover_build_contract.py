#!/usr/bin/env python3
"""Assemble/check the sealed 16-binding PHAxis handover build contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from handover_manifest_producers import (
    ProducerError,
    assemble_handover_build_contract,
    handover,
    publish_report_no_overwrite,
)


def _option(role: str) -> str:
    return "--" + role.replace("_", "-")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    for role in handover.REQUIRED_BINDINGS:
        parser.add_argument(_option(role), dest=role, type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="repeat exactly five times in member_index=0..4 order",
    )
    parser.add_argument("--release-attestation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    bindings = {role: getattr(args, role) for role in handover.REQUIRED_BINDINGS}
    try:
        report = assemble_handover_build_contract(
            project_root=args.project_root,
            bindings=bindings,
            checkpoint_paths=args.checkpoint,
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
