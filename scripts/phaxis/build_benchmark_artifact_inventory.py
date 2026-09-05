#!/usr/bin/env python3
"""Build a sealed, explicit inventory of formal benchmark evidence files."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if SOURCE_ROOT.is_dir():
    sys.path.insert(0, str(SOURCE_ROOT))

from phaxis.io import read_json, sha256_file, sha256_json  # noqa: E402


SCHEMA_VERSION = "PHAxis-benchmark-artifact-inventory-1.0"
FIELDS = (
    "source_path",
    "package_path",
    "sha256",
    "bytes",
    "provenance",
    "notes",
    "release_authorized",
    "artifact_role",
)
EXACT_ROLES = frozenset(
    {
        "same_hardware_receipt",
        "phaxis_production_summary",
        "v1_production_summary",
        "phaxis_sequential_summary",
        "v1_sequential_summary",
        "production_comparison_receipt",
        "sequential_comparison_receipt",
    }
)
REPEATED_ROLES = frozenset(
    {"per_image_latency_csv", "gpu_telemetry", "hardware_preflight"}
)


class InventoryError(RuntimeError):
    """An explicit benchmark inventory binding is unsafe or incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InventoryError(message)


def _safe_relative(value: str, *, field: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    _require(
        bool(path.parts)
        and not path.is_absolute()
        and "." not in path.parts
        and ".." not in path.parts
        and ":" not in value,
        f"{field} is not a safe relative path: {value}",
    )
    return path.as_posix()


def _project_relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise InventoryError(f"benchmark artifact leaves project root: {path}") from error


def _parse_artifact(value: str) -> tuple[str, str, Path]:
    parts = value.split("=", 2)
    _require(
        len(parts) == 3 and all(part.strip() for part in parts),
        "--artifact must be ROLE=PACKAGE_PATH=SOURCE_PATH",
    )
    role = parts[0].strip()
    package = _safe_relative(parts[1].strip(), field="package_path")
    _require(
        role in EXACT_ROLES | REPEATED_ROLES,
        f"unsupported benchmark artifact role: {role}",
    )
    _require(
        package.startswith("model/benchmark/"),
        f"benchmark package path leaves model/benchmark: {package}",
    )
    return role, package, Path(parts[2].strip())


def _publish_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise InventoryError(f"refusing to overwrite: {path}") from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _sealed(payload: Mapping[str, Any], field: str) -> None:
    unsigned = deepcopy(dict(payload))
    observed = unsigned.pop(field, None)
    _require(observed == sha256_json(unsigned), f"{field} does not seal receipt")


def build_inventory(
    *,
    project_root: str | Path,
    artifacts: Sequence[str],
    output: str | Path,
    receipt: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    destination = Path(output)
    receipt_path = Path(receipt)
    if not destination.is_absolute():
        destination = root / destination
    if not receipt_path.is_absolute():
        receipt_path = root / receipt_path
    destination = destination.resolve()
    receipt_path = receipt_path.resolve()
    _project_relative(root, destination)
    _project_relative(root, receipt_path)
    _require(destination != receipt_path, "inventory CSV and receipt paths collide")
    _require(not destination.exists(), f"refusing to overwrite: {destination}")
    _require(not receipt_path.exists(), f"refusing to overwrite: {receipt_path}")

    parsed = [_parse_artifact(value) for value in artifacts]
    roles: dict[str, int] = {}
    package_paths: set[str] = set()
    rows: list[dict[str, Any]] = []
    for role, package, source_value in parsed:
        source = source_value if source_value.is_absolute() else root / source_value
        source = source.resolve()
        _project_relative(root, source)
        _require(source.is_file() and not source.is_symlink(), f"artifact is absent: {source}")
        _require(package.casefold() not in package_paths, f"duplicate package path: {package}")
        package_paths.add(package.casefold())
        roles[role] = roles.get(role, 0) + 1
        rows.append(
            {
                "source_path": _project_relative(root, source),
                "package_path": package,
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
                "provenance": "direct formal same-hardware exact283 benchmark stage",
                "notes": f"hash-locked benchmark evidence role={role}",
                "release_authorized": "true",
                "artifact_role": role,
            }
        )
    for role in EXACT_ROLES:
        _require(roles.get(role) == 1, f"inventory requires exactly one {role}")
    _require(roles.get("per_image_latency_csv", 0) >= 2, "two per-image latency CSVs are required")
    _require(roles.get("gpu_telemetry", 0) >= 1, "GPU telemetry evidence is required")
    _require(roles.get("hardware_preflight", 0) >= 1, "hardware preflight evidence is required")
    rows.sort(key=lambda row: str(row["package_path"]))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    encoded = stream.getvalue().encode("utf-8")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_explicit_benchmark_inventory",
        "rows": len(rows),
        "role_counts": dict(sorted(roles.items())),
        "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
        "inventory_bytes": len(encoded),
        "canonical_annotations_read": False,
        "condition_metadata_used_for_routing": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    payload["inventory_identity_sha256"] = sha256_json(payload)
    receipt_bytes = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _publish_new(destination, encoded)
    try:
        _publish_new(receipt_path, receipt_bytes)
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    published = read_json(receipt_path)
    _sealed(published, "inventory_identity_sha256")
    return published


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="repeat ROLE=PACKAGE_PATH=SOURCE_PATH; no discovery is performed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = build_inventory(
            project_root=args.project_root,
            artifacts=args.artifact,
            output=args.output,
            receipt=args.receipt,
        )
    except (InventoryError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
