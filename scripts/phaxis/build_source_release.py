#!/usr/bin/env python
"""Build a new PHAxis source-release tree from authoritative project sources."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# This entry point may be executed from the exact-manifest source tree.  Keep
# every project-module import observational even when the caller forgot -B.
sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(SCRIPT_ROOT), str(PROJECT_ROOT / "src")]

from source_release_common import SourceReleaseError, build_source_release  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically build a deterministic PHAxis source tree from an explicit "
            "allowlist. The output must be absent or empty."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root-provider-exact283-receipt", type=Path)
    parser.add_argument("--train399-candidate-manifest", type=Path)
    parser.add_argument("--train399-selection-receipt", type=Path)
    parser.add_argument("--train399-evaluation-receipt", type=Path)
    parser.add_argument("--final-fusion-summary", type=Path)
    parser.add_argument("--final-traits-summary", type=Path)
    parser.add_argument(
        "--release-human-metadata",
        type=Path,
        help=(
            "separately sealed author/repository/PyPI/GitHub/rights authority; "
            "required for a formal release"
        ),
    )
    parser.add_argument(
        "--allow-blocked-development-staging",
        action="store_true",
        help=(
            "permit a conspicuously blocked, non-publishable development tree when "
            "one or more formal gates fail"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        manifest = build_source_release(
            project_root=args.project_root,
            output=args.output,
            allow_blocked_development_staging=args.allow_blocked_development_staging,
            root_provider_receipt=args.root_provider_exact283_receipt,
            train399_candidate_manifest=args.train399_candidate_manifest,
            train399_selection_receipt=args.train399_selection_receipt,
            train399_evaluation_receipt=args.train399_evaluation_receipt,
            final_fusion_summary=args.final_fusion_summary,
            final_traits_summary=args.final_traits_summary,
            release_human_metadata=args.release_human_metadata,
        )
    except SourceReleaseError as error:
        print(f"PHAxis source-release build blocked: {error}")
        return 2
    print(
        json.dumps(
            {
                "status": "built",
                "release_mode": manifest["release_mode"],
                "output": str(args.output.resolve()),
                "files": len(manifest["files"]),
                "tree_identity_sha256": manifest["tree_identity_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
