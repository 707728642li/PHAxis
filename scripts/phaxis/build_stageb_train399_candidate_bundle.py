"""Build a non-promoting Stage-B train399 candidate receipt on CPU.

The command accepts exactly five completed checkpoints.  It never starts a
CUDA context and never edits PHAxis constants or the official model contract.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.hair_stageb.candidate_bundle import (  # noqa: E402
    build_candidate_manifest,
    write_candidate_manifest,
)


DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "outputs"
    / "phaxis_stageb_train399_dataset_audit_run1"
    / "dataset_audit.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail closed unless five fixed-seed, epoch-60 checkpoints prove the "
            "same strict train399-only contract. A pass remains a candidate only."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        required=True,
        help="repeat exactly five times; input order is canonicalized by fixed seed",
    )
    parser.add_argument("--dataset-audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--amp-amendment",
        type=Path,
        default=None,
        help=(
            "validated AMP backward-retry amendment; when omitted, exactly one "
            "matching amendment must be safely inferable beside the five seed directories"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-identical-existing",
        action="store_true",
        help="idempotently accept an existing byte-semantically identical receipt",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if len(args.checkpoint) != 5:
        raise RuntimeError("exactly five --checkpoint arguments are required")
    manifest = build_candidate_manifest(
        args.checkpoint,
        dataset_audit_path=args.dataset_audit,
        amp_amendment_path=args.amp_amendment,
    )
    write_candidate_manifest(
        args.output,
        manifest,
        allow_identical_existing=bool(args.allow_identical_existing),
    )
    print(f"candidate gate: {manifest['status']}")
    print(f"candidate identity: {manifest['candidate_bundle_identity_sha256']}")
    print(f"manifest identity: {manifest['candidate_manifest_identity_sha256']}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
