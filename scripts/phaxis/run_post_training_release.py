"""Plan or explicitly execute the sealed PHAxis post-training release chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.release_orchestrator import (  # noqa: E402
    EXPECTED_GPU_HOLD_EXIT_CODE,
    EXPECTED_HUMAN_GATE_EXIT_CODE,
    ReleaseOrchestratorError,
    build_release_plan,
    execute_release,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "execute the validated commands; omission is deterministic "
            "plan/check-only and writes nothing"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate the complete sentinel prefix and continue; requires --execute",
    )
    parser.add_argument(
        "--hold-physical-gpu",
        action="append",
        default=[],
        type=int,
        metavar="INDEX",
        help=(
            "pause normally before the first stage requiring this physical GPU; "
            "repeat for multiple cards"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.resume and not args.execute:
        parser.error("--resume requires --execute")
    if args.hold_physical_gpu and not args.execute:
        parser.error("--hold-physical-gpu requires --execute")
    try:
        if args.execute:
            result = execute_release(
                args.manifest,
                args.output,
                resume=bool(args.resume),
                held_physical_gpus=tuple(args.hold_physical_gpu),
            )
        else:
            result = build_release_plan(args.manifest, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if result.get("status") == "paused_for_deferred_human_authority":
            return EXPECTED_HUMAN_GATE_EXIT_CODE
        if result.get("status") == "paused_for_user_gpu_hold":
            return EXPECTED_GPU_HOLD_EXIT_CODE
        return 0
    except (ReleaseOrchestratorError, OSError, ValueError) as error:
        print(f"PHAxis post-training release blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
