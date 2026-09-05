#!/usr/bin/env python3
"""Seal and inventory the final PHAxis checkpoint/root-provider model assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from handover_manifest_producers import ProducerError, build_model_asset_manifest, publish_report_no_overwrite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--applied-model-contract", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="repeat exactly five times in member_index=0..4 order",
    )
    parser.add_argument("--root-provider-bundle-root", type=Path, required=True)
    parser.add_argument("--root-provider-bundle-manifest", type=Path, required=True)
    parser.add_argument("--root-provider-verification-receipt", type=Path, required=True)
    parser.add_argument("--release-example-root", type=Path, required=True)
    parser.add_argument("--portable-capsule-output", type=Path, required=True)
    parser.add_argument("--bundle-manifest-output", type=Path, required=True)
    parser.add_argument("--release-attestation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        report = build_model_asset_manifest(
            project_root=args.project_root,
            applied_model_contract=args.applied_model_contract,
            candidate_manifest=args.candidate_manifest,
            checkpoint_paths=args.checkpoint,
            root_provider_bundle_root=args.root_provider_bundle_root,
            root_provider_bundle_manifest=args.root_provider_bundle_manifest,
            root_provider_verification_receipt=args.root_provider_verification_receipt,
            release_example_root=args.release_example_root,
            portable_capsule_output=args.portable_capsule_output,
            bundle_manifest_output=args.bundle_manifest_output,
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
            created = report.get("outputs_created_this_invocation")
            if not isinstance(created, list) or not all(
                isinstance(value, str) and value for value in created
            ):
                parser.error("model-asset publication recovery metadata is absent")
            publish_report_no_overwrite(
                project_root=args.project_root,
                report=report,
                receipt=args.receipt,
                rollback_outputs=tuple(created),
            )
        except ProducerError as error:
            parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
