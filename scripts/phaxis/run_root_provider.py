#!/usr/bin/env python
"""Explicit raw-image -> Hybrid-Max root-provider entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.io import atomic_write_json  # noqa: E402
from phaxis.root_provider.runtime import (  # noqa: E402
    PipelineConfig,
    build_execution_plan,
    run_pipeline,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--acquisition-gate", type=Path, required=True)
    parser.add_argument("--deployment-metadata", type=Path, required=True)
    parser.add_argument("--canonical-manifest", type=Path, required=True)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--deployment-lock", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--v1-physical-gpu",
        type=int,
        action="append",
        required=True,
        help="Explicit V1 physical GPU; repeat to round-robin V1 shards.",
    )
    parser.add_argument(
        "--q8-physical-gpu",
        type=int,
        action="append",
        required=True,
        help="Explicit Q8 physical GPU; repeat to round-robin Q8 shards.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--v1-shards", type=int, default=4)
    parser.add_argument("--v1-concurrency", type=int, default=2)
    parser.add_argument("--v20-shards", type=int, default=8)
    parser.add_argument("--v20-concurrency", type=int, default=8)
    parser.add_argument("--q8-shards", type=int, default=8)
    parser.add_argument("--q8-concurrency", type=int, default=1)
    parser.add_argument("--field-batch-size", type=int, default=10)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--reference-registry", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run the plan. Without this flag the command is strictly plan-only.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--strict-physical-gpu",
        action="store_true",
        help=(
            "Require every Q8 shard to stay on its requested physical GPU and "
            "validate index/UUID bindings before merge."
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    config = PipelineConfig(
        project=args.project,
        bundle=args.bundle,
        input_manifest=args.input_manifest,
        acquisition_gate=args.acquisition_gate,
        deployment_metadata=args.deployment_metadata,
        canonical_manifest=args.canonical_manifest,
        deployment_manifest=args.deployment_manifest,
        deployment_lock=args.deployment_lock,
        image_root=args.image_root,
        output=args.output,
        v1_physical_gpus=tuple(args.v1_physical_gpu),
        q8_physical_gpus=tuple(args.q8_physical_gpu),
        python_executable=args.python,
        v1_shards=args.v1_shards,
        v1_concurrency=args.v1_concurrency,
        v20_shards=args.v20_shards,
        v20_concurrency=args.v20_concurrency,
        q8_shards=args.q8_shards,
        q8_concurrency=args.q8_concurrency,
        field_batch_size=args.field_batch_size,
        query_batch_size=args.query_batch_size,
        reference_registry=args.reference_registry,
        strict_physical_gpu=args.strict_physical_gpu,
    )
    if args.execute:
        result = run_pipeline(config, resume=args.resume)
    else:
        result = build_execution_plan(config)
    if args.plan_output is not None:
        atomic_write_json(args.plan_output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
