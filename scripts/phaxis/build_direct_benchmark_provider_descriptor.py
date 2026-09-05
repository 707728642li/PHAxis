#!/usr/bin/env python3
"""Build the sealed four-mode PHAxis direct-benchmark provider descriptor.

The builder is CPU-only.  It hashes the actual adapter, its implementation
closure and frozen-v1 static inputs.  ``--assemble`` is create-only: an
existing descriptor is never replaced, including under a concurrent writer.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from phaxis.benchmark import (  # noqa: E402
    COLD_LATENCY_MODE,
    FROZEN_V1_BENCHMARK_SYSTEM,
    MEASUREMENT_SCOPE,
    PHAXIS_BENCHMARK_SYSTEM,
    PRODUCTION_MODE,
)
from phaxis.io import sha256_file, sha256_json  # noqa: E402


INTERFACE_SCHEMA = "PHAxis-formal-direct-benchmark-provider-descriptor-1.0"
MODES = (
    "phaxis_production",
    "frozen_v1_production",
    "phaxis_sequential",
    "frozen_v1_sequential",
)
ADAPTER = "scripts/phaxis/run_external_direct_benchmark.py"
FROZEN_ACQUISITION_GATE = (
    "outputs/rhaxis_six_condition_v1_inputs_full283_run2_measured_gate/"
    "prephenotype_qc_delegation_gate.json"
)
FROZEN_RUNTIME_CONFIG = "configs/rhpheno_dual_mode_runtime_v6_v19_prospective.json"


class DescriptorBuildError(RuntimeError):
    """The repository cannot support a sealed direct provider descriptor."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DescriptorBuildError(message)


def _relative_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise DescriptorBuildError(f"path leaves project root: {relative}") from error
    _require(path.is_file() and not path.is_symlink(), f"required file is absent or symlinked: {relative}")
    return path


def _record(root: Path, relative: str) -> dict[str, str]:
    path = _relative_file(root, relative)
    return {"path": relative.replace("\\", "/"), "sha256": sha256_file(path)}


def _python_closure(root: Path, package: str, extras: Iterable[str]) -> list[dict[str, str]]:
    package_root = _relative_file(root, f"src/{package}/__init__.py").parent
    paths = {
        path.relative_to(root).as_posix()
        for path in package_root.rglob("*.py")
        if path.is_file() and not path.is_symlink()
    }
    paths.update(str(value).replace("\\", "/") for value in extras)
    return [_record(root, relative) for relative in sorted(paths)]


def build_descriptor(project_root: str | Path, *, physical_gpu: int = 0) -> dict[str, Any]:
    """Return a fully hash-sealed descriptor without writing it."""

    root = Path(project_root).resolve()
    _require(root == PROJECT_ROOT.resolve(), "descriptor must bind the current project root")
    _require(
        physical_gpu == 0,
        "the frozen-v1 V19 runtime is immutably bound to physical GPU0; "
        "all four formal modes must therefore use GPU0",
    )

    phaxis_closure = _python_closure(
        root,
        "phaxis",
        (
            ADAPTER,
            "scripts/phaxis/benchmark_full_workflow.py",
        ),
    )
    frozen_closure = _python_closure(
        root,
        "rhizoweave",
        (
            ADAPTER,
            "scripts/run_rhpheno_dual_mode_v3.py",
            "scripts/run_rhpheno_dual_mode_v5.py",
        ),
    )
    entrypoints: dict[str, Any] = {}
    for mode in MODES:
        is_phaxis = mode.startswith("phaxis_")
        is_production = mode.endswith("production")
        entrypoints[mode] = {
            **_record(root, ADAPTER),
            "benchmark_system": (
                PHAXIS_BENCHMARK_SYSTEM if is_phaxis else FROZEN_V1_BENCHMARK_SYSTEM
            ),
            "benchmark_mode": PRODUCTION_MODE if is_production else COLD_LATENCY_MODE,
            "warmup_runs": 0,
            "measured_repeats": 1,
            "physical_gpus": [physical_gpu],
            "implementation_closure": deepcopy(
                phaxis_closure if is_phaxis else frozen_closure
            ),
        }
    payload: dict[str, Any] = {
        "schema_version": INTERFACE_SCHEMA,
        "status": "ready_hash_locked_direct_execution",
        "measurement_scope": MEASUREMENT_SCOPE,
        "exact_images": 283,
        "formal_result_receipts_emitted": True,
        "blind_images_used": 0,
        "same_physical_gpu_required_for_all_modes": True,
        "entrypoints": entrypoints,
        "static_inputs": {
            "frozen_v1_acquisition_gate": _record(root, FROZEN_ACQUISITION_GATE),
            "frozen_v1_runtime_config": _record(root, FROZEN_RUNTIME_CONFIG),
        },
    }
    payload["descriptor_identity_sha256"] = sha256_json(payload)
    return payload


def write_descriptor_create_only(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically create *path* without ever replacing an existing file."""

    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require(not destination.exists(), f"descriptor output already exists: {destination}")
    temporary = destination.parent / f".{destination.name}.attempt-{os.getpid()}"
    _require(not temporary.exists(), f"descriptor attempt path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # On Windows os.rename is atomic and fails rather than replacing an
        # existing destination, which preserves the create-only contract.
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--physical-gpu", type=int, default=0)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--assemble", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        payload = build_descriptor(args.project_root, physical_gpu=args.physical_gpu)
        if args.assemble:
            _require(args.output is not None, "--assemble requires --output")
            written = write_descriptor_create_only(args.output, payload)
        else:
            _require(args.output is None, "--check does not accept --output")
            written = None
    except (DescriptorBuildError, OSError, ValueError) as error:
        parser.error(str(error))
    result = {
        "status": "assembled_create_only" if written else "ready_cpu_preflight_only",
        "descriptor_identity_sha256": payload["descriptor_identity_sha256"],
        "descriptor_output": str(written) if written else None,
        "gpu_program_started": False,
        "nvidia_smi_called": False,
        "blind_images_used": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
