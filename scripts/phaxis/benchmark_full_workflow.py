#!/usr/bin/env python3
"""Direct full-workflow PHAxis benchmark CLI (dry-run by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if (SOURCE_ROOT / "phaxis").is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from phaxis.benchmark import (
    benchmark_plan,
    compare_benchmarks,
    compile_sequential_latency_benchmark,
    inspect_frozen_v1_exact283_benchmark_producer,
    publish_same_hardware_benchmark_receipt,
    run_production_batch_benchmark,
    same_hardware_benchmark_plan,
)
from phaxis.contracts import ContractError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--manifest")
    parser.add_argument("--workflow-output")
    parser.add_argument("--output")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--execute-production-batch",
        action="store_true",
        help="run one new, fresh production batch; otherwise plan only",
    )
    mode.add_argument(
        "--compile-sequential-trace",
        action="store_true",
        help="validate a sealed direct per-source trace; does not run GPU inference",
    )
    mode.add_argument(
        "--aggregate-same-hardware",
        action="store_true",
        help="validate six explicit exact283 receipts; remains check-only unless --publish-receipt",
    )
    mode.add_argument(
        "--inspect-frozen-v1-producer",
        action="store_true",
        help=(
            "CPU-only inspection of the legacy exact283 producer interface; "
            "prints a non-formal blocked Gate and never executes a benchmark"
        ),
    )
    mode.add_argument(
        "--compare-benchmarks",
        action="store_true",
        help="compare two explicit same-mode direct summaries and publish one receipt",
    )
    parser.add_argument("--producer-interface")
    parser.add_argument("--trace-csv")
    parser.add_argument("--trace-receipt")
    parser.add_argument(
        "--baseline-receipt",
        help="optional same-mode frozen-v1 direct receipt; incomparable scopes get no speedup",
    )
    parser.add_argument("--phaxis-summary")
    parser.add_argument("--phaxis-production-summary")
    parser.add_argument("--phaxis-sequential-summary")
    parser.add_argument("--frozen-v1-production-summary")
    parser.add_argument("--frozen-v1-sequential-summary")
    parser.add_argument("--production-comparison")
    parser.add_argument("--sequential-comparison")
    parser.add_argument(
        "--publish-receipt",
        action="store_true",
        help="explicitly publish the sealed same-hardware receipt to --output",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.inspect_frozen_v1_producer:
        forbidden = (
            args.output,
            args.workflow_output,
            args.trace_csv,
            args.trace_receipt,
            args.baseline_receipt,
            args.phaxis_production_summary,
            args.phaxis_sequential_summary,
            args.frozen_v1_production_summary,
            args.frozen_v1_sequential_summary,
            args.production_comparison,
            args.sequential_comparison,
            args.publish_receipt,
        )
        if any(forbidden):
            parser.error(
                "--inspect-frozen-v1-producer is CPU-only/check-only and forbids "
                "benchmark, publication, and --output arguments"
            )
        if not args.manifest:
            parser.error("--inspect-frozen-v1-producer requires --manifest")
        try:
            gate = inspect_frozen_v1_exact283_benchmark_producer(
                project_root=args.project_root,
                source_manifest=args.manifest,
                producer_interface=args.producer_interface,
            )
        except (ContractError, OSError, ValueError) as error:
            print(
                f"frozen-v1 exact283 producer inspection failed closed: {error}",
                file=sys.stderr,
            )
            return 2
        print(json.dumps(gate, indent=2, sort_keys=True))
        return 0 if gate["status"] == "ready_interface_only_non_formal" else 3
    if args.compare_benchmarks:
        if not args.output or not args.phaxis_summary or not args.baseline_receipt:
            parser.error(
                "--compare-benchmarks requires --phaxis-summary, "
                "--baseline-receipt, and --output"
            )
        forbidden = (
            args.manifest,
            args.workflow_output,
            args.trace_csv,
            args.trace_receipt,
            args.publish_receipt,
            args.producer_interface,
        )
        if any(forbidden):
            parser.error("--compare-benchmarks accepts only explicit summary inputs")
        summary = compare_benchmarks(
            phaxis_summary=args.phaxis_summary,
            baseline_summary=args.baseline_receipt,
            output=Path(args.output).resolve(),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.producer_interface:
        parser.error(
            "--producer-interface applies only to --inspect-frozen-v1-producer"
        )
    if not args.output:
        parser.error("benchmark modes require --output")
    output = Path(args.output).resolve()
    if args.aggregate_same_hardware:
        names = (
            "phaxis_production_summary",
            "phaxis_sequential_summary",
            "frozen_v1_production_summary",
            "frozen_v1_sequential_summary",
            "production_comparison",
            "sequential_comparison",
        )
        missing = [name for name in names if getattr(args, name) is None]
        if missing:
            parser.error("--aggregate-same-hardware requires: " + ", ".join(name.replace("_", "-") for name in missing))
        inputs = {name: getattr(args, name) for name in names}
        if args.publish_receipt:
            summary = publish_same_hardware_benchmark_receipt(
                output=output, **inputs
            )
        else:
            summary = same_hardware_benchmark_plan(**inputs)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if not args.manifest or not args.workflow_output:
        parser.error("workflow benchmark modes require --manifest and --workflow-output")
    if args.publish_receipt:
        parser.error("--publish-receipt applies only to --aggregate-same-hardware")
    if args.compile_sequential_trace:
        if not args.trace_csv or not args.trace_receipt:
            parser.error("--compile-sequential-trace requires --trace-csv and --trace-receipt")
        summary = compile_sequential_latency_benchmark(
            manifest=args.manifest,
            workflow_output=args.workflow_output,
            trace_csv=args.trace_csv,
            trace_receipt=args.trace_receipt,
            benchmark_output=output,
        )
    elif args.execute_production_batch:
        if args.trace_csv or args.trace_receipt:
            parser.error("trace inputs apply only to --compile-sequential-trace")
        summary = run_production_batch_benchmark(
            manifest=args.manifest,
            workflow_output=args.workflow_output,
            benchmark_output=output,
        )
    else:
        if args.baseline_receipt or args.trace_csv or args.trace_receipt:
            parser.error("baseline/trace inputs require an explicit execution or compile mode")
        summary = benchmark_plan(
            manifest=args.manifest,
            workflow_output=args.workflow_output,
            benchmark_output=output,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.baseline_receipt:
        comparison = compare_benchmarks(
            phaxis_summary=output / "runtime_summary.json",
            baseline_summary=args.baseline_receipt,
            output=output / "benchmark_comparison.json",
        )
        summary = {"runtime": summary, "comparison": comparison}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
