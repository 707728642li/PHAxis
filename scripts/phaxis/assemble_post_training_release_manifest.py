"""Check, assemble, or assemble-and-launch the formal PHAxis release manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.release_manifest_assembler import (  # noqa: E402
    ReleaseManifestAssemblyError,
    _atomic_write_new_json,
    assemble_release_manifest,
    configured_output_root,
    inspect_release_readiness,
)
from phaxis.release_orchestrator import (  # noqa: E402
    EXPECTED_GPU_HOLD_EXIT_CODE,
    EXPECTED_HUMAN_GATE_EXIT_CODE,
    ReleaseOrchestratorError,
    execute_release,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check", action="store_true", help="read-only readiness inspection; no manifest is required")
    parser.add_argument("--report-output", type=Path, help="optional create-only JSON readiness report")
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--run-output", type=Path)
    parser.add_argument("--assemble", action="store_true", help="atomically create the sealed manifest")
    parser.add_argument("--launch", action="store_true", help="assemble then execute the release DAG")
    parser.add_argument(
        "--hold-physical-gpu",
        action="append",
        default=[],
        type=int,
        metavar="INDEX",
        help="with --launch, pause normally before a stage requiring this physical GPU",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    modes = sum(bool(value) for value in (args.check, args.assemble, args.launch))
    if modes != 1:
        parser.error("choose exactly one of --check, --assemble, or --launch")
    if args.report_output is not None and not args.check:
        parser.error("--report-output is only valid with --check")
    if (args.assemble or args.launch) and args.manifest_output is None:
        parser.error("--assemble/--launch require --manifest-output")
    if args.hold_physical_gpu and not args.launch:
        parser.error("--hold-physical-gpu is only valid with --launch")
    try:
        if args.check:
            result = inspect_release_readiness(args.config)
            if args.report_output is not None:
                _atomic_write_new_json(args.report_output.resolve(), result)
            print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
            return (
                0
                if result["status"]
                in {
                    "ready_to_assemble",
                    "ready_to_assemble_science_prefix_human_gate_deferred",
                }
                else 3
            )
        run_output = args.run_output or configured_output_root(args.config)
        assembly = assemble_release_manifest(
            args.config,
            args.manifest_output,
            run_dir=run_output,
        )
        if args.launch:
            execution = execute_release(
                args.manifest_output,
                run_output,
                held_physical_gpus=tuple(args.hold_physical_gpu),
            )
            result = {"assembly": assembly, "execution": execution}
        else:
            result = assembly
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        if (
            args.launch
            and result["execution"].get("status")
            == "paused_for_deferred_human_authority"
        ):
            return EXPECTED_HUMAN_GATE_EXIT_CODE
        if (
            args.launch
            and result["execution"].get("status")
            == "paused_for_user_gpu_hold"
        ):
            return EXPECTED_GPU_HOLD_EXIT_CODE
        return 0
    except (ReleaseManifestAssemblyError, ReleaseOrchestratorError, OSError, ValueError, TypeError) as error:
        print(f"PHAxis release manifest assembly blocked: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
