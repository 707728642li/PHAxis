"""Materialize an exact, hash-verified PHAxis root-provider bundle closure.

Runtime imports can create ``__pycache__`` files beside a frozen read-only
bundle.  Those cache files are neither model assets nor registry members.  This
producer never mutates the frozen source: it copies only registry-bound members
and ``root_provider_bundle.json`` into ``<output>/bundle``, rejects
missing/tampered members, verifies exact closure, writes
``<output>/verification.json``, then publishes that complete container with one
directory rename.  The receipt and bundle therefore cannot be split by a
process interruption.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.io import atomic_write_json, read_json, sha256_file  # noqa: E402
from phaxis.root_provider import BundleError, verify_bundle  # noqa: E402


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BundleError(message)


def materialize_verified_bundle(
    *, source_bundle: str | Path, output: str | Path
) -> dict:
    source = Path(source_bundle).resolve()
    destination = Path(output).resolve()
    _require(source.is_dir() and not source.is_symlink(), "source bundle is absent or a symlink")
    _require(not destination.exists(), f"refusing to overwrite: {destination}")
    registry_path = source / "root_provider_bundle.json"
    _require(registry_path.is_file() and not registry_path.is_symlink(), "source bundle registry is absent")
    registry = read_json(registry_path)
    records = registry.get("files")
    _require(isinstance(records, list) and records, "source bundle file registry is empty")

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".root-bundle-container-", dir=parent)).resolve()
    try:
        staged_bundle = staging / "bundle"
        staged_bundle.mkdir()
        seen: set[str] = set()
        for index, record in enumerate(records):
            _require(isinstance(record, dict), f"bundle record {index} is invalid")
            relative = Path(str(record.get("path", "")))
            _require(
                bool(relative.parts)
                and not relative.is_absolute()
                and ".." not in relative.parts,
                f"bundle record {index} path is unsafe",
            )
            normalized = relative.as_posix().casefold()
            _require(normalized not in seen, f"duplicate/case-colliding bundle record: {relative}")
            seen.add(normalized)
            source_file = (source / relative).resolve()
            try:
                source_file.relative_to(source)
            except ValueError as error:
                raise BundleError(f"bundle path escape: {relative}") from error
            _require(
                source_file.is_file() and not source_file.is_symlink(),
                f"registered bundle member is absent or a symlink: {relative}",
            )
            _require(
                sha256_file(source_file) == record.get("sha256")
                and source_file.stat().st_size == record.get("bytes"),
                f"registered bundle member hash/size mismatch: {relative}",
            )
            target = staged_bundle / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target)
        shutil.copy2(registry_path, staged_bundle / "root_provider_bundle.json")

        verification = verify_bundle(staged_bundle, require_exact_closure=True)
        verification.update(
            {
                "status": "pass",
                "source_bundle_registry_sha256": sha256_file(registry_path),
                "source_bundle_mutated": False,
                "materialized_exact_closure": True,
            }
        )
        # The receipt is adjacent to the exact bundle *inside the atomic
        # container*.  It is not inside bundle/, so exact closure remains true.
        atomic_write_json(staging / "verification.json", verification)
        os.replace(staging, destination)
        return verification
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = materialize_verified_bundle(
        source_bundle=args.source_bundle,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BundleError, OSError, ValueError, TypeError) as error:
        print(f"blocked: {error}", file=sys.stderr)
        raise SystemExit(2)
