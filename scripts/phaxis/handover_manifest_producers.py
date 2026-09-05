#!/usr/bin/env python3
"""Explicit, deterministic producers for PHAxis handover authorities.

These producers deliberately do not discover a "latest" run.  Every upstream
authority, source root, checkpoint, and output path is supplied by the caller.
Check-only mode performs the same streaming hash/size and semantic validation
as execution, but publishes no output.  Execution always uses no-overwrite
publication.

The module is standard-library only.  In particular, it never imports Torch,
runs inference, or copies any payload into a handover package.
"""

from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
import handover_package_common as handover


PRODUCT = "PHAxis"
VERSION = "1.0.0"
ATTESTATION_SCHEMA = "PHAxis-handover-scope-license-attestation-1.0"
MODEL_BUNDLE_SCHEMA = "PHAxis-model-bundle-release-manifest-1.0"
PLAN_SCHEMA = "PHAxis-handover-materialisation-plan-1.0"
FORMAL_TRAIN399_SEEDS = (
    2026082801,
    2026082802,
    2026082803,
    2026082804,
    2026082805,
)
EXACT_MANUAL_TASKS = frozenset(
    f"RHAUD-{index:03d}" for index in range(1, 501)
)
MATERIALISATION_ROLES = (
    "dataset_manifest",
    "image_manifest",
    "model_source_manifest",
    "model_asset_manifest",
    "benchmark_manifest",
)
SCOPE_FIELDS = (
    "all_legally_deliverable_manual_annotations_included",
    "training_and_validation_annotations_may_be_mixed",
    "annotation_notes_provenance_hashes_preserved",
    "biological283_includes_temperature_and_rhd6_design",
    "image_assembly_excluded",
    "blind_and_final_partitions_excluded",
    "frozen_v1_untouched",
)
COMMON_FIELDS = (
    "source_path",
    "package_path",
    "sha256",
    "bytes",
    "provenance",
    "notes",
    "release_authorized",
)
DATASET_FIELDS = ("task_id", "dataset_id", "annotation_kind")
IMAGE_FIELDS = ("task_id", "temperature_c", "genotype_or_construct")
MODEL_ASSET_FIELDS = ("asset_role", "member_index", "seed")
BENCHMARK_FIELDS = ("artifact_role",)
DATASET_ID_ALL500 = "RHAxis-Arabidopsis-HumanAnnotated500"
DATASET_ID_CANONICAL443 = "RHAxis-Arabidopsis-HumanCurated443-v1.0"
DATASET_SUPPORT = (
    ("DATASET_CARD.md", "data/human_annotated500/DATASET_CARD.md"),
    ("LICENSE_DATA.md", "data/human_annotated500/LICENSE_DATA.md"),
    ("README_CN.md", "data/human_annotated500/README_CN.md"),
    ("build_summary.json", "data/human_annotated500/build_summary.json"),
    ("label_schema.json", "data/human_annotated500/label_schema.json"),
    (
        "manifests/dataset_manifest.csv",
        "data/human_annotated500/manifests/dataset_manifest.csv",
    ),
    (
        "manifests/filter_decisions_all500.csv",
        "data/human_annotated500/manifests/filter_decisions_all500.csv",
    ),
    (
        "manifests/integrity_sha256.csv",
        "data/human_annotated500/manifests/integrity_sha256.csv",
    ),
    (
        "manifests/split_manifest.csv",
        "data/human_annotated500/manifests/split_manifest.csv",
    ),
    ("provenance.json", "data/human_annotated500/provenance.json"),
    (
        "verification_report.json",
        "data/human_annotated500/verification_report.json",
    ),
)
FORBIDDEN_ASSEMBLY_TOKENS = (
    "stitch",
    "mosaic",
    "tile_assembly",
    "image_assembly",
    "拼接",
)
EXECUTABLE_OR_CODE_SUFFIXES = {".bat", ".cmd", ".exe", ".ps1", ".py", ".sh"}
SHA256_LENGTH = 64
PORTABLE_CAPSULE_SCHEMA = "PHAxis-portable-model-runtime-capsule-1.0"
REQUIRED_SOURCE_SUPPLY_CHAIN_FILES = frozenset(
    {
        "NOTICE",
        "THIRD_PARTY_NOTICES.md",
        "THIRD_PARTY_LICENSES.json",
        "SBOM.cdx.json",
    }
)


class ProducerError(RuntimeError):
    """A producer input or proposed output failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProducerError(message)


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and value == value.casefold()
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_relative(value: Any, *, field: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{field} is absent")
    path = PurePosixPath(str(value).replace("\\", "/"))
    _require(
        not path.is_absolute() and ".." not in path.parts and "." not in path.parts,
        f"{field} is not a safe relative path: {value}",
    )
    for part in path.parts:
        _require(
            bool(part)
            and ":" not in part
            and not any(ord(character) < 32 for character in part)
            and not part.endswith((" ", ".")),
            f"{field} is not portable: {value}",
        )
    return path.as_posix()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_project_file(project_root: Path, value: str | Path, *, field: str) -> Path:
    root = project_root.resolve()
    supplied = Path(value)
    lexical = supplied if supplied.is_absolute() else root / supplied
    try:
        lexical_relative = lexical.relative_to(root)
    except ValueError as error:
        raise ProducerError(f"{field} escapes the project root: {value}") from error
    _require(".." not in lexical_relative.parts, f"{field} contains parent traversal")
    cursor = root
    for part in lexical_relative.parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), f"{field} traverses a symlink: {value}")
    resolved = lexical.resolve()
    _require(_inside(resolved, root), f"{field} escapes the project root: {value}")
    _require(resolved.is_file(), f"{field} is absent or not a file: {value}")
    return resolved


def _resolve_project_directory(
    project_root: Path, value: str | Path, *, field: str
) -> Path:
    root = project_root.resolve()
    supplied = Path(value)
    lexical = supplied if supplied.is_absolute() else root / supplied
    try:
        lexical_relative = lexical.relative_to(root)
    except ValueError as error:
        raise ProducerError(f"{field} escapes the project root: {value}") from error
    _require(".." not in lexical_relative.parts, f"{field} contains parent traversal")
    cursor = root
    for part in lexical_relative.parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), f"{field} traverses a symlink: {value}")
    resolved = lexical.resolve()
    _require(_inside(resolved, root), f"{field} escapes the project root: {value}")
    _require(resolved.is_dir(), f"{field} is absent or not a directory: {value}")
    return resolved


def _planned_project_output(
    project_root: Path, value: str | Path, *, field: str
) -> Path:
    root = project_root.resolve()
    supplied = Path(value)
    lexical = supplied if supplied.is_absolute() else root / supplied
    try:
        lexical_relative = lexical.relative_to(root)
    except ValueError as error:
        raise ProducerError(f"{field} must stay inside the project root") from error
    _require(".." not in lexical_relative.parts, f"{field} contains parent traversal")
    _safe_relative(lexical_relative.as_posix(), field=field)
    cursor = root
    for part in lexical_relative.parts[:-1]:
        cursor = cursor / part
        _require(not cursor.is_symlink(), f"{field} traverses a symlink: {value}")
    parent = lexical.parent.resolve()
    _require(_inside(parent, root), f"{field} must stay inside the project root")
    _require(not lexical.exists(), f"{field} already exists; overwrite is forbidden")
    _require(not lexical.is_symlink(), f"{field} may not be a symlink")
    return parent / lexical.name


def _recoverable_project_output(
    project_root: Path,
    value: str | Path,
    *,
    field: str,
    directory: bool,
) -> Path:
    """Resolve an output that may be an exact atomically-published retry prefix.

    This is deliberately narrower than overwrite support: an existing target is
    accepted only so its complete closure/hash can be checked before a producer
    completes missing sibling outputs after a process/host interruption.
    """

    root = project_root.resolve()
    supplied = Path(value)
    lexical = supplied if supplied.is_absolute() else root / supplied
    try:
        lexical_relative = lexical.relative_to(root)
    except ValueError as error:
        raise ProducerError(f"{field} must stay inside the project root") from error
    _require(".." not in lexical_relative.parts, f"{field} contains parent traversal")
    _safe_relative(lexical_relative.as_posix(), field=field)
    cursor = root
    for part in lexical_relative.parts[:-1]:
        cursor = cursor / part
        _require(not cursor.is_symlink(), f"{field} traverses a symlink: {value}")
    parent = lexical.parent.resolve()
    _require(_inside(parent, root), f"{field} must stay inside the project root")
    destination = parent / lexical.name
    _require(not destination.is_symlink(), f"{field} may not be a symlink")
    if destination.exists():
        _require(
            destination.is_dir() if directory else destination.is_file(),
            f"{field} existing retry prefix has the wrong filesystem type",
        )
    return destination


def _project_relative(project_root: Path, path: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProducerError(f"cannot read {role} JSON {path}: {error}") from error
    _require(isinstance(payload, dict), f"{role} JSON is not an object")
    return payload


def _read_csv(path: Path, *, required: Iterable[str], role: str) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or ())
            missing = set(required) - fieldnames
            _require(not missing, f"{role} lacks columns: {sorted(missing)}")
            return [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise ProducerError(f"cannot read {role} CSV {path}: {error}") from error


def _sealed_identity(payload: Mapping[str, Any], field: str, *, role: str) -> str:
    unsigned = deepcopy(dict(payload))
    claimed = unsigned.pop(field, None)
    _require(_is_sha256(claimed), f"{role} {field} is absent or invalid")
    _require(_sha256_json(unsigned) == claimed, f"{role} identity is invalid")
    return str(claimed)


def validate_release_attestation(
    project_root: Path, path: str | Path, *, required_role: str | None = None
) -> tuple[Path, dict[str, Any]]:
    authority = _resolve_project_file(project_root, path, field="release attestation")
    payload = _read_json(authority, role="release attestation")
    _require(
        payload.get("schema_version") == ATTESTATION_SCHEMA
        and payload.get("status") == "approved_for_formal_handover"
        and payload.get("product") == PRODUCT
        and payload.get("product_version") == VERSION,
        "scope/license attestation identity or status is invalid",
    )
    _sealed_identity(payload, "attestation_identity_sha256", role="release attestation")
    scope = payload.get("scope_attestation")
    _require(isinstance(scope, Mapping), "release attestation lacks scope_attestation")
    for field in SCOPE_FIELDS:
        _require(scope.get(field) is True, f"release attestation scope is false: {field}")
    roles = payload.get("authorized_materialisation_roles")
    _require(
        isinstance(roles, list)
        and len(roles) == len(set(map(str, roles)))
        and set(map(str, roles)) == set(MATERIALISATION_ROLES),
        "release attestation materialisation-role authorization is incomplete or expanded",
    )
    bases = payload.get("license_basis_by_materialisation_role")
    _require(
        isinstance(bases, Mapping)
        and set(bases) == set(MATERIALISATION_ROLES)
        and all(bool(str(value).strip()) for value in bases.values()),
        "release attestation lacks a license/release basis for every materialisation role",
    )
    _require(
        bool(str(payload.get("authority_name", "")).strip())
        and bool(str(payload.get("approval_reference", "")).strip()),
        "release attestation lacks authority_name/approval_reference",
    )
    _require(payload.get("blind_images_used") == 0, "release attestation is blind-tainted")
    _require(
        payload.get("root_cap_region_statistics_included") is False,
        "release attestation includes root-cap region statistics",
    )
    _require(
        payload.get("historical_or_provisional_backfill_used") is False,
        "historical/provisional backfill is forbidden",
    )
    if required_role is not None:
        _require(required_role in roles, f"release attestation does not authorize {required_role}")
    return authority, payload


def _manifest_bytes(rows: Sequence[Mapping[str, Any]], extra_fields: Sequence[str]) -> bytes:
    fields = [*COMMON_FIELDS, *extra_fields]
    unexpected = sorted(
        {key for row in rows for key in row if key not in set(fields)}
    )
    fields.extend(unexpected)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _publish_no_overwrite(path: Path, data: bytes) -> None:
    """Atomically publish bytes without replacing an existing authority.

    Bytes are fsynced in a same-directory temporary file before the atomic hard
    link.  A hard crash can therefore leave either no target or the complete
    target, never a truncated target that makes the stage unrecoverable.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".publish-tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ProducerError(f"refusing to overwrite output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def publish_report_no_overwrite(
    *,
    project_root: str | Path,
    report: Mapping[str, Any],
    receipt: str | Path,
    rollback_outputs: Sequence[str | Path] = (),
) -> Path:
    """Publish one self-sealed producer report without replacing a file.

    Materialisation payloads are the data authorities.  This companion JSON is
    the receipt consumed by the post-training release orchestrator.  Publishing
    it in the same process closes the former stdout-only authority gap.
    """

    root = Path(project_root).resolve()
    try:
        destination = _planned_project_output(
            root, receipt, field="materialisation receipt output"
        )
        _publish_no_overwrite(destination, _json_bytes(report))
    except BaseException:
        # Each materialisation producer is no-overwrite.  Therefore a payload
        # path named here can only have been created by this invocation.  Roll
        # it back so the orchestrator can safely retry an unsealed stage.
        for value in rollback_outputs:
            rollback = _planned_project_existing_output(root, value)
            if rollback.is_dir():
                shutil.rmtree(rollback)
            else:
                rollback.unlink(missing_ok=True)
        raise
    return destination


def _planned_project_existing_output(project_root: Path, value: str | Path) -> Path:
    """Resolve one exact rollback file within the project, without discovery."""

    supplied = Path(value)
    lexical = supplied if supplied.is_absolute() else project_root / supplied
    resolved = lexical.resolve()
    _require(_inside(resolved, project_root), "receipt rollback output escapes project root")
    _require(
        resolved.exists() and not resolved.is_symlink(),
        "receipt rollback output is absent",
    )
    return resolved


def _materialisation_report(
    *,
    project_root: Path,
    role: str,
    output: Path,
    rows: Sequence[Mapping[str, Any]],
    encoded: bytes,
    execute: bool,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "status": "created" if execute else "passed_check_only_not_written",
        "materialisation_role": role,
        "output": _project_relative(project_root, output),
        "rows": len(rows),
        "payload_bytes": sum(int(row["bytes"]) for row in rows),
        "manifest_bytes": len(encoded),
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
        "historical_or_provisional_backfill_used": False,
    }
    if extra:
        report.update(extra)
    report["plan_identity_sha256"] = _sha256_json(report)
    return report


def _row(
    *,
    project_root: Path,
    source: Path,
    package_path: str,
    provenance: str,
    notes: str,
    **extra: Any,
) -> dict[str, Any]:
    safe_package = _safe_relative(package_path, field="package_path")
    _require(bool(provenance.strip()), "materialisation provenance is absent")
    return {
        "source_path": _project_relative(project_root, source),
        "package_path": safe_package,
        "sha256": _sha256_file(source),
        "bytes": source.stat().st_size,
        "provenance": provenance,
        "notes": notes,
        "release_authorized": "true",
        **extra,
    }


def _validate_declared_file(
    *,
    path: Path,
    expected_sha256: Any,
    expected_bytes: Any | None,
    role: str,
) -> None:
    _require(_is_sha256(expected_sha256), f"{role} declared SHA-256 is invalid")
    _require(_sha256_file(path) == expected_sha256, f"{role} SHA-256 mismatch")
    if expected_bytes is not None and str(expected_bytes).strip():
        try:
            size = int(expected_bytes)
        except (TypeError, ValueError) as error:
            raise ProducerError(f"{role} declared byte size is invalid") from error
        _require(path.stat().st_size == size, f"{role} byte-size mismatch")


def _dataset_authority_gates(dataset_root: Path) -> None:
    summary = _read_json(dataset_root / "build_summary.json", role="dataset build summary")
    _require(
        summary.get("schema_version") == "RHAxis-standard-dataset-build-summary-1.0"
        and summary.get("returned_tasks") == 500
        and summary.get("accepted_core_tasks") == 443
        and summary.get("train_tasks") == 399
        and summary.get("val_tasks") == 44
        and summary.get("verification_status") == "passed"
        and summary.get("blind_images_used") == 0,
        "HumanCurated443 build summary is not the final exact500/443 authority",
    )
    verification = _read_json(
        dataset_root / "verification_report.json", role="dataset verification"
    )
    _require(
        verification.get("schema_version")
        == "RHAxis-standard-dataset-verification-1.0"
        and verification.get("status") == "passed"
        and verification.get("tasks") == 443
        and verification.get("train") == 399
        and verification.get("val") == 44
        and verification.get("blind_images_used") == 0,
        "HumanCurated443 verification receipt is not passed exact443",
    )
    provenance = _read_json(dataset_root / "provenance.json", role="dataset provenance")
    _require(
        provenance.get("schema_version") == "RHAxis-standard-dataset-provenance-1.0"
        and provenance.get("dataset_version") == DATASET_ID_CANONICAL443
        and provenance.get("raw_annotations_modified") is False
        and provenance.get("canonical_geometry_modified") is False
        and provenance.get("blind_images_used") == 0,
        "HumanCurated443 provenance is not immutable/blind-free",
    )


def build_dataset_manifest(
    *,
    project_root: str | Path,
    manual_image_manifest: str | Path,
    all500_decisions: str | Path,
    canonical_dataset_root: str | Path,
    canonical_dataset_manifest: str | Path,
    canonical_integrity_manifest: str | Path,
    all500_notes: str | Path,
    release_attestation: str | Path,
    output: str | Path,
    execute: bool = False,
) -> dict[str, Any]:
    """Validate and produce the exact500+443 dataset materialisation CSV."""

    root = Path(project_root).resolve()
    destination = _planned_project_output(root, output, field="dataset manifest output")
    image_authority = _resolve_project_file(
        root, manual_image_manifest, field="manual500 image manifest"
    )
    decisions_authority = _resolve_project_file(
        root, all500_decisions, field="all500 decisions manifest"
    )
    dataset_root = _resolve_project_directory(
        root, canonical_dataset_root, field="canonical443 dataset root"
    )
    canonical_authority = _resolve_project_file(
        root, canonical_dataset_manifest, field="canonical443 dataset manifest"
    )
    integrity_authority = _resolve_project_file(
        root, canonical_integrity_manifest, field="canonical443 integrity manifest"
    )
    notes_authority = _resolve_project_file(root, all500_notes, field="ALL500 notes")
    attestation_path, _attestation = validate_release_attestation(
        root, release_attestation, required_role="dataset_manifest"
    )
    _require(
        canonical_authority == dataset_root / "manifests/dataset_manifest.csv"
        and decisions_authority
        == dataset_root / "manifests/filter_decisions_all500.csv"
        and integrity_authority == dataset_root / "manifests/integrity_sha256.csv",
        "canonical dataset authorities must be the explicitly named files inside canonical_dataset_root",
    )
    for relative, _package in DATASET_SUPPORT:
        _resolve_project_file(root, dataset_root / relative, field=f"dataset support {relative}")
    _dataset_authority_gates(dataset_root)

    image_rows = _read_csv(
        image_authority,
        required=(
            "task_id",
            "staged_tiff_path",
            "source_image_sha256",
            "staged_tiff_sha256",
            "bytes",
            "status",
        ),
        role="manual500 image manifest",
    )
    _require(len(image_rows) == 500, "manual500 image manifest is not exact500")
    images_by_task: dict[str, tuple[Path, dict[str, str]]] = {}
    for row_index, row in enumerate(image_rows, 2):
        task = str(row.get("task_id", "")).strip()
        _require(
            task in EXACT_MANUAL_TASKS and task not in images_by_task,
            f"manual500 image row {row_index} has an invalid/duplicate task_id",
        )
        _require(row.get("status") == "copied_verified", f"manual500 image {task} is not copied_verified")
        _require(
            row.get("source_image_sha256") == row.get("staged_tiff_sha256"),
            f"manual500 image {task} source/staged identities differ",
        )
        source = _resolve_project_file(
            root, row["staged_tiff_path"], field=f"manual500 image {task}"
        )
        _validate_declared_file(
            path=source,
            expected_sha256=row.get("staged_tiff_sha256"),
            expected_bytes=row.get("bytes"),
            role=f"manual500 image {task}",
        )
        images_by_task[task] = (source, row)
    _require(set(images_by_task) == EXACT_MANUAL_TASKS, "manual500 image task set is not RHAUD-001--500")

    decision_rows = _read_csv(
        decisions_authority,
        required=(
            "task_id",
            "dataset_decision",
            "decision_reasons",
            "review_notes",
            "returned_annotation_path",
        ),
        role="all500 decisions manifest",
    )
    _require(len(decision_rows) == 500, "all500 decisions manifest is not exact500")
    decisions_by_task: dict[str, tuple[Path, dict[str, str], str]] = {}
    for row_index, row in enumerate(decision_rows, 2):
        task = str(row.get("task_id", "")).strip()
        _require(
            task in EXACT_MANUAL_TASKS and task not in decisions_by_task,
            f"all500 decision row {row_index} has an invalid/duplicate task_id",
        )
        raw = _resolve_project_file(
            root,
            row["returned_annotation_path"],
            field=f"manual500 raw return {task}",
        )
        try:
            parsed = json.loads(raw.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ProducerError(f"manual500 raw return is invalid JSON: {task}") from error
        _require(isinstance(parsed, Mapping), f"manual500 raw return is not an object: {task}")
        raw_hash = _sha256_file(raw)
        decisions_by_task[task] = (raw, row, raw_hash)
    _require(set(decisions_by_task) == EXACT_MANUAL_TASKS, "manual500 raw-return task set is not RHAUD-001--500")

    canonical_rows = _read_csv(
        canonical_authority,
        required=(
            "task_id",
            "dataset_version",
            "image_relpath",
            "raw_annotation_relpath",
            "canonical_annotation_relpath",
            "image_sha256",
            "raw_annotation_sha256",
        ),
        role="canonical443 dataset manifest",
    )
    _require(len(canonical_rows) == 443, "canonical dataset manifest is not exact443")
    canonical_by_task: dict[str, dict[str, str]] = {}
    for row_index, row in enumerate(canonical_rows, 2):
        task = str(row.get("task_id", "")).strip()
        _require(
            task in EXACT_MANUAL_TASKS and task not in canonical_by_task,
            f"canonical443 row {row_index} has an invalid/duplicate task_id",
        )
        _require(
            row.get("dataset_version") == DATASET_ID_CANONICAL443,
            f"canonical443 dataset_version drift: {task}",
        )
        _require(
            images_by_task[task][1]["staged_tiff_sha256"] == row.get("image_sha256"),
            f"manual500/canonical443 source image identity mismatch: {task}",
        )
        _require(
            decisions_by_task[task][2] == row.get("raw_annotation_sha256"),
            f"manual500/canonical443 raw return identity mismatch: {task}",
        )
        canonical_by_task[task] = row
    accepted = {
        task
        for task, (_path, row, _digest) in decisions_by_task.items()
        if row.get("dataset_decision") == "accepted_core"
    }
    _require(accepted == set(canonical_by_task), "accepted_core/all500 and canonical443 task sets differ")

    integrity_rows = _read_csv(
        integrity_authority,
        required=("task_id", "role", "relative_path", "sha256", "size_bytes"),
        role="canonical443 integrity manifest",
    )
    integrity: dict[tuple[str, str], dict[str, str]] = {}
    for row in integrity_rows:
        key = (str(row.get("task_id", "")).strip(), str(row.get("role", "")).strip())
        _require(key not in integrity, f"duplicate canonical integrity identity: {key}")
        integrity[key] = row

    authority_provenance = (
        f"manual_images={_project_relative(root, image_authority)};"
        f"all500_decisions={_project_relative(root, decisions_authority)};"
        f"canonical443={_project_relative(root, canonical_authority)};"
        f"release_attestation={_project_relative(root, attestation_path)}"
    )
    rows: list[dict[str, Any]] = []
    for task in sorted(EXACT_MANUAL_TASKS):
        image, image_record = images_by_task[task]
        raw, decision, _raw_hash = decisions_by_task[task]
        decision_note = "; ".join(
            value
            for value in (
                str(decision.get("dataset_decision", "")).strip(),
                str(decision.get("decision_reasons", "")).strip(),
                str(decision.get("review_notes", "")).strip(),
            )
            if value
        )
        rows.append(
            _row(
                project_root=root,
                source=image,
                package_path=f"data/human_annotated500/images/{task}.ome.tif",
                provenance=authority_provenance,
                notes=f"exact500 source image; {decision_note}",
                task_id=task,
                dataset_id=DATASET_ID_ALL500,
                annotation_kind="manual500_source_image",
            )
        )
        _require(
            rows[-1]["sha256"] == image_record["staged_tiff_sha256"],
            f"manual500 image changed during row construction: {task}",
        )
        rows.append(
            _row(
                project_root=root,
                source=raw,
                package_path=f"data/human_annotated500/annotations/raw_return/{task}.json",
                provenance=authority_provenance,
                notes=f"unmodified current raw human return; {decision_note}",
                task_id=task,
                dataset_id=DATASET_ID_ALL500,
                annotation_kind="manual500_raw_return_json",
            )
        )
    for task in sorted(canonical_by_task):
        record = canonical_by_task[task]
        relative = _safe_relative(
            record["canonical_annotation_relpath"], field="canonical annotation path"
        )
        source = _resolve_project_file(
            root, dataset_root / relative, field=f"canonical443 vector {task}"
        )
        integrity_record = integrity.get((task, "canonical_annotation"))
        _require(integrity_record is not None, f"canonical443 integrity lock is absent: {task}")
        _require(
            integrity_record.get("relative_path") == relative,
            f"canonical443 integrity path mismatch: {task}",
        )
        _validate_declared_file(
            path=source,
            expected_sha256=integrity_record.get("sha256"),
            expected_bytes=integrity_record.get("size_bytes"),
            role=f"canonical443 vector {task}",
        )
        rows.append(
            _row(
                project_root=root,
                source=source,
                package_path=f"data/human_annotated500/annotations/rhaxis_canonical/{task}.json",
                provenance=authority_provenance,
                notes="accepted_core canonical vector JSON; immutable HumanCurated443 projection",
                task_id=task,
                dataset_id=DATASET_ID_CANONICAL443,
                annotation_kind="canonical443_vector_json",
            )
        )
    for relative, package_path in DATASET_SUPPORT:
        source = _resolve_project_file(
            root, dataset_root / relative, field=f"dataset support {relative}"
        )
        rows.append(
            _row(
                project_root=root,
                source=source,
                package_path=package_path,
                provenance=authority_provenance,
                notes="canonical dataset support authority",
                task_id="",
                dataset_id=DATASET_ID_ALL500,
                annotation_kind="dataset_support",
            )
        )
    rows.append(
        _row(
            project_root=root,
            source=notes_authority,
            package_path="data/human_annotated500/notes/ALL500_DATA_NOTES_CN.md",
            provenance=authority_provenance,
            notes="owner-authored exact500 disposition and reuse notes",
            task_id="",
            dataset_id=DATASET_ID_ALL500,
            annotation_kind="dataset_support",
        )
    )
    rows.sort(key=lambda row: str(row["package_path"]))
    _require(
        len({str(row["package_path"]).casefold() for row in rows}) == len(rows),
        "dataset package paths collide",
    )
    encoded = _manifest_bytes(rows, DATASET_FIELDS)
    report = _materialisation_report(
        project_root=root,
        role="dataset_manifest",
        output=destination,
        rows=rows,
        encoded=encoded,
        execute=execute,
        extra={
            "manual500_source_images": 500,
            "manual500_raw_returns": 500,
            "canonical443_vectors": 443,
            "dataset_support_files": len(DATASET_SUPPORT) + 1,
        },
    )
    if execute:
        _publish_no_overwrite(destination, encoded)
    return report


def build_image_manifest(
    *,
    project_root: str | Path,
    deployment_manifest: str | Path,
    deployment_lock: str | Path,
    image_root: str | Path,
    release_attestation: str | Path,
    output: str | Path,
    execute: bool = False,
) -> dict[str, Any]:
    """Project one explicit locked biological deployment into exact283 rows."""

    root = Path(project_root).resolve()
    destination = _planned_project_output(root, output, field="image manifest output")
    manifest_path = _resolve_project_file(
        root, deployment_manifest, field="biological283 deployment manifest"
    )
    lock_path = _resolve_project_file(
        root, deployment_lock, field="biological283 deployment lock"
    )
    images_root = _resolve_project_directory(root, image_root, field="biological283 image root")
    attestation_path, _attestation = validate_release_attestation(
        root, release_attestation, required_role="image_manifest"
    )
    lock = _read_json(lock_path, role="biological283 deployment lock")
    _require(
        lock.get("schema_version") == "RHAxis-NextGen-deployment-manifest-lock-1.0"
        and lock.get("status") == "locked_before_phenotype_inference"
        and lock.get("samples") == 283
        and lock.get("manifest_sha256") == _sha256_file(manifest_path)
        and lock.get("canonical_annotations_read") is False
        and lock.get("phenotype_model_predictions_used") is False
        and lock.get("condition_used_for_model_routing") is False
        and lock.get("blind_images_used") == 0,
        "biological deployment lock is not the blind-free exact283 source authority",
    )
    _require(
        lock.get("manifest") == manifest_path.name,
        "biological deployment lock names a different manifest",
    )
    source_rows = _read_csv(
        manifest_path,
        required=(
            "task_id",
            "image_relpath",
            "image_sha256",
            "temperature_c",
            "genotype_or_construct",
        ),
        role="biological283 deployment manifest",
    )
    _require(len(source_rows) == 283, "biological deployment manifest is not exact283")
    provenance = (
        f"deployment_manifest={_project_relative(root, manifest_path)};"
        f"deployment_lock={_project_relative(root, lock_path)};"
        f"release_attestation={_project_relative(root, attestation_path)}"
    )
    rows: list[dict[str, Any]] = []
    tasks: set[str] = set()
    image_hashes: set[str] = set()
    package_paths: set[str] = set()
    for row_index, source_row in enumerate(source_rows, 2):
        task = str(source_row.get("task_id", "")).strip()
        _require(task and task not in tasks, f"biological283 row {row_index} has duplicate/absent task_id")
        tasks.add(task)
        relative = _safe_relative(source_row.get("image_relpath"), field="biological image_relpath")
        source = _resolve_project_file(
            root, images_root / relative, field=f"biological283 image {task}"
        )
        _validate_declared_file(
            path=source,
            expected_sha256=source_row.get("image_sha256"),
            expected_bytes=source_row.get("image_bytes") or source_row.get("bytes"),
            role=f"biological283 image {task}",
        )
        digest = str(source_row["image_sha256"])
        _require(digest not in image_hashes, f"biological283 image bytes are duplicated: {task}")
        image_hashes.add(digest)
        temperature = str(source_row.get("temperature_c", "")).strip()
        genotype = str(source_row.get("genotype_or_construct", "")).strip()
        _require(temperature and genotype, f"biological283 metadata is incomplete: {task}")
        package_path = f"data/biological283/images/{relative}"
        key = package_path.casefold()
        _require(key not in package_paths, f"biological283 package path collides: {package_path}")
        package_paths.add(key)
        notes = "; ".join(
            part
            for part in (
                f"condition={source_row.get('condition_code', '')}",
                f"study_role={source_row.get('study_role', '')}",
                f"qc_disposition={source_row.get('qc_disposition', '')}",
            )
            if not part.endswith("=")
        )
        rows.append(
            _row(
                project_root=root,
                source=source,
                package_path=package_path,
                provenance=provenance,
                notes=notes or "locked exact283 biological source image",
                task_id=task,
                temperature_c=temperature,
                genotype_or_construct=genotype,
            )
        )
    _require(len(tasks) == 283 and len(image_hashes) == 283, "biological image closure is not 283 unique tasks/images")
    _require(
        len({row["temperature_c"] for row in rows}) >= 2
        and any("rhd6" in str(row["genotype_or_construct"]).casefold() for row in rows),
        "biological283 lacks temperature or RHD6 design coverage",
    )
    rows.sort(key=lambda row: str(row["package_path"]))
    encoded = _manifest_bytes(rows, IMAGE_FIELDS)
    report = _materialisation_report(
        project_root=root,
        role="image_manifest",
        output=destination,
        rows=rows,
        encoded=encoded,
        execute=execute,
        extra={
            "images": 283,
            "deployment_identity_sha256": lock.get("deployment_identity_sha256"),
            "temperature_levels": sorted({str(row["temperature_c"]) for row in rows}),
        },
    )
    if execute:
        _publish_no_overwrite(destination, encoded)
    return report


def _is_forbidden_assembly_component(relative: str) -> bool:
    path = PurePosixPath(relative.casefold())
    return path.suffix in EXECUTABLE_OR_CODE_SUFFIXES and any(
        token in path.stem for token in FORBIDDEN_ASSEMBLY_TOKENS
    )


def build_model_source_manifest(
    *,
    project_root: str | Path,
    source_release_root: str | Path,
    source_release_manifest: str | Path,
    release_attestation: str | Path,
    output: str | Path,
    execute: bool = False,
) -> dict[str, Any]:
    """Project the exact source tree plus its named manifest into handover rows.

    ``SOURCE_MANIFEST.json.files`` deliberately excludes the manifest itself to
    avoid a recursive digest.  The reuse package nevertheless needs the named
    manifest at ``model/source_release/SOURCE_MANIFEST.json`` so that the copied
    tree remains independently verifiable and installable exactly as documented.
    """

    root = Path(project_root).resolve()
    destination = _planned_project_output(root, output, field="model source manifest output")
    release_root = _resolve_project_directory(root, source_release_root, field="formal source release root")
    manifest_path = _resolve_project_file(
        root, source_release_manifest, field="formal SOURCE_MANIFEST"
    )
    _require(manifest_path == release_root / "SOURCE_MANIFEST.json", "SOURCE_MANIFEST must be the named manifest in source_release_root")
    attestation_path, _attestation = validate_release_attestation(
        root, release_attestation, required_role="model_source_manifest"
    )
    manifest = _read_json(manifest_path, role="formal SOURCE_MANIFEST")
    records = manifest.get("files")
    _require(
        manifest.get("schema_version") == "PHAxis-source-release-manifest-2.0"
        and manifest.get("distribution") == "phaxis"
        and manifest.get("version") == VERSION
        and manifest.get("release_mode") == "formal"
        and manifest.get("source_policy") == "explicit_path_bounded_allowlist"
        and isinstance(records, list)
        and manifest.get("tree_identity_sha256") == _sha256_json(records),
        "source release is not a sealed formal PHAxis 1.0.0 manifest",
    )
    expected_paths: list[str] = []
    rows: list[dict[str, Any]] = []
    provenance_suffix = (
        f"SOURCE_MANIFEST={_project_relative(root, manifest_path)};"
        f"release_attestation={_project_relative(root, attestation_path)}"
    )
    for row_index, record in enumerate(records):
        _require(isinstance(record, Mapping), f"SOURCE_MANIFEST.files[{row_index}] is invalid")
        relative = _safe_relative(record.get("path"), field="source release file path")
        _require(relative not in expected_paths, f"duplicate source release path: {relative}")
        _require(not _is_forbidden_assembly_component(relative), f"forbidden image-assembly component: {relative}")
        expected_paths.append(relative)
        source = _resolve_project_file(
            root, release_root / relative, field=f"source release file {relative}"
        )
        _validate_declared_file(
            path=source,
            expected_sha256=record.get("sha256"),
            expected_bytes=record.get("bytes"),
            role=f"source release file {relative}",
        )
        origin = str(record.get("origin", "")).strip()
        _require(origin, f"source release file lacks origin: {relative}")
        rows.append(
            _row(
                project_root=root,
                source=source,
                package_path=f"model/source_release/{relative}",
                provenance=f"{origin};{provenance_suffix}",
                notes="exact formal SOURCE_MANIFEST projection",
            )
        )
    rows.append(
        _row(
            project_root=root,
            source=manifest_path,
            package_path="model/source_release/SOURCE_MANIFEST.json",
            provenance=f"named-source-release-control;{provenance_suffix}",
            notes=(
                "named formal SOURCE_MANIFEST control; excluded from its own "
                "files array to avoid recursive hashing"
            ),
        )
    )
    _require(expected_paths == sorted(expected_paths), "SOURCE_MANIFEST.files is not canonically sorted")
    actual_paths = sorted(
        path.relative_to(release_root).as_posix()
        for path in release_root.rglob("*")
        if path.is_file() and path != manifest_path
    )
    _require(actual_paths == expected_paths, "formal source release tree differs from SOURCE_MANIFEST.files")
    _require(
        REQUIRED_SOURCE_SUPPLY_CHAIN_FILES.issubset(expected_paths),
        "formal source release lacks PHAxis NOTICE/license inventory/SBOM closure",
    )
    notice_text = (release_root / "NOTICE").read_text(encoding="utf-8")
    third_party_notice_text = (
        release_root / "THIRD_PARTY_NOTICES.md"
    ).read_text(encoding="utf-8")
    _require(
        "PHAxis 1.0.0" in notice_text
        and "PHAxis 1.0.0 third-party notices" in third_party_notice_text
        and "RHPheno" not in notice_text
        and "RHPheno" not in third_party_notice_text,
        "formal source release contains a legacy/non-PHAxis notice",
    )
    license_inventory = _read_json(
        release_root / "THIRD_PARTY_LICENSES.json",
        role="formal source third-party license inventory",
    )
    _require(
        license_inventory.get("schema_version")
        == "PHAxis-third-party-license-inventory-1.0"
        and license_inventory.get("status")
        == "complete_declared_direct_dependency_inventory"
        and license_inventory.get("product") == PRODUCT
        and license_inventory.get("product_version") == VERSION,
        "formal source third-party license inventory is not PHAxis 1.0.0",
    )
    _sealed_identity(
        license_inventory,
        "inventory_identity_sha256",
        role="formal source third-party license inventory",
    )
    sbom = _read_json(
        release_root / "SBOM.cdx.json",
        role="formal source CycloneDX SBOM",
    )
    sbom_metadata = sbom.get("metadata")
    sbom_component = (
        sbom_metadata.get("component")
        if isinstance(sbom_metadata, Mapping)
        else None
    )
    _require(
        sbom.get("bomFormat") == "CycloneDX"
        and sbom.get("specVersion") == "1.6"
        and isinstance(sbom_component, Mapping)
        and sbom_component.get("name") == "phaxis"
        and sbom_component.get("version") == VERSION
        and isinstance(sbom.get("components"), list)
        and bool(sbom["components"]),
        "formal source SBOM is not the PHAxis 1.0.0 CycloneDX authority",
    )
    gate_record = next(
        (record for record in records if record.get("path") == "FORMAL_RELEASE_GATE_RECEIPT.json"),
        None,
    )
    _require(isinstance(gate_record, Mapping), "formal source release lacks FORMAL_RELEASE_GATE_RECEIPT.json")
    gate = _read_json(release_root / "FORMAL_RELEASE_GATE_RECEIPT.json", role="formal source release gate")
    _require(
        gate.get("schema_version") == "PHAxis-source-release-gate-1.0"
        and gate.get("status") == "passed"
        and gate.get("formal_release_allowed") is True,
        "formal source release gate receipt did not pass",
    )
    rows.sort(key=lambda row: str(row["package_path"]))
    encoded = _manifest_bytes(rows, ())
    report = _materialisation_report(
        project_root=root,
        role="model_source_manifest",
        output=destination,
        rows=rows,
        encoded=encoded,
        execute=execute,
        extra={
            "source_files": len(rows),
            "manifest_control_files": 1,
            "source_tree_identity_sha256": manifest["tree_identity_sha256"],
        },
    )
    if execute:
        _publish_no_overwrite(destination, encoded)
    return report


def _validate_candidate_and_applied(
    *,
    candidate_path: Path,
    applied_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    candidate = _read_json(candidate_path, role="train399 candidate manifest")
    _require(
        candidate.get("schema_version")
        == "PHAxis-StageB-train399-candidate-bundle-1.0"
        and candidate.get("status") == "candidate_gate_passed_not_promoted"
        and candidate.get("blind_images_used") == 0,
        "train399 candidate manifest is not the formal blind-free candidate",
    )
    _sealed_identity(
        candidate,
        "candidate_manifest_identity_sha256",
        role="train399 candidate manifest",
    )
    identity = candidate.get("identity_payload")
    _require(
        isinstance(identity, Mapping)
        and candidate.get("candidate_bundle_identity_sha256")
        == _sha256_json(identity),
        "train399 candidate bundle identity is invalid",
    )
    members = identity.get("members")
    _require(
        isinstance(members, list)
        and len(members) == 5
        and all(isinstance(row, Mapping) for row in members),
        "train399 candidate does not contain five members",
    )
    normalized_members = [dict(row) for row in members]
    _require(
        [row.get("member_index") for row in normalized_members] == list(range(5))
        and tuple(row.get("seed") for row in normalized_members)
        == FORMAL_TRAIN399_SEEDS,
        "train399 candidate member/seed order is invalid",
    )
    candidate_hashes = [row.get("checkpoint_sha256") for row in normalized_members]
    _require(
        len(set(candidate_hashes)) == 5
        and all(_is_sha256(value) for value in candidate_hashes),
        "train399 candidate checkpoint hashes are not five distinct SHA-256 values",
    )

    applied = _read_json(applied_path, role="applied model contract")
    _require(
        applied.get("schema_version") == "PHAxis-model-contract-1.0.0"
        and applied.get("product") == PRODUCT
        and applied.get("product_version") == VERSION
        and applied.get("formal_release_status") == "passed",
        "applied model contract is not formal PHAxis 1.0.0",
    )
    _sealed_identity(
        applied,
        "model_contract_identity_sha256",
        role="applied model contract",
    )
    promotion = applied.get("promotion")
    red_lines = applied.get("red_lines")
    expert = applied.get("hair_identity_count_expert")
    root_expert = applied.get("root_expert")
    _require(
        isinstance(promotion, Mapping)
        and promotion.get("status") == "applied_formal_release"
        and promotion.get("official_apply_performed") is True,
        "model contract was not formally CAS-applied",
    )
    _require(
        isinstance(red_lines, Mapping)
        and red_lines.get("blind_images_used") == 0
        and red_lines.get("root_cap_region_statistics_included", False) is False
        and red_lines.get("formal_train399_only_stageb_weights_available") is True,
        "applied model contract red lines are invalid",
    )
    _require(
        isinstance(expert, Mapping)
        and isinstance(expert.get("expert_id"), str)
        and bool(expert.get("expert_id"))
        and expert.get("checkpoint_sha256_in_member_order") == candidate_hashes,
        "applied model contract checkpoint order differs from the candidate",
    )
    _require(
        isinstance(root_expert, Mapping)
        and isinstance(root_expert.get("expert_id"), str)
        and bool(root_expert.get("expert_id"))
        and _is_sha256(root_expert.get("bundle_identity_sha256"))
        and _is_sha256(promotion.get("proposal_file_sha256"))
        and _is_sha256(promotion.get("proposal_identity_sha256")),
        "applied model contract lacks final proposal/public expert identities",
    )
    return candidate, applied, normalized_members


def _validate_root_bundle(
    *,
    project_root: Path,
    bundle_root: Path,
    manifest_path: Path,
    verification_path: Path,
) -> tuple[dict[str, Any], list[tuple[Path, dict[str, Any]]]]:
    _require(
        manifest_path == bundle_root / "root_provider_bundle.json",
        "root-provider bundle manifest must be root_provider_bundle.json in bundle_root",
    )
    payload = _read_json(manifest_path, role="root-provider model bundle")
    records = payload.get("files")
    _require(
        payload.get("schema_version") == "PHAxis-root-provider-model-bundle-1.0"
        and payload.get("status") == "materialized_unverified"
        and isinstance(records, list)
        and bool(records),
        "root-provider bundle is absent, planned-only, or has the wrong schema",
    )
    portable = payload.get("portable_execution_contract")
    _require(
        isinstance(portable, Mapping)
        and portable.get("implicit_download") is False
        and portable.get("blind_images_used") == 0,
        "root-provider bundle portability/blind boundary is invalid",
    )
    identity_payload = {
        "schema_version": payload.get("schema_version"),
        "bundle_id": payload.get("bundle_id"),
        "files": records,
        "contracts": payload.get("contracts"),
    }
    _require(
        _is_sha256(payload.get("bundle_identity_sha256"))
        and payload.get("bundle_identity_sha256") == _sha256_json(identity_payload),
        "root-provider bundle identity is invalid",
    )
    resolved: list[tuple[Path, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, raw_record in enumerate(records):
        _require(isinstance(raw_record, Mapping), f"root-provider asset row {index} is invalid")
        record = dict(raw_record)
        relative = _safe_relative(record.get("path"), field="root-provider asset path")
        key = relative.casefold()
        _require(key not in seen, f"duplicate/case-colliding root-provider asset: {relative}")
        seen.add(key)
        source = _resolve_project_file(
            project_root,
            bundle_root / relative,
            field=f"root-provider asset {relative}",
        )
        _validate_declared_file(
            path=source,
            expected_sha256=record.get("sha256"),
            expected_bytes=record.get("bytes"),
            role=f"root-provider asset {relative}",
        )
        record["path"] = relative
        resolved.append((source, record))
    _require(
        payload.get("files_count") == len(records)
        and payload.get("bytes") == sum(int(row[1]["bytes"]) for row in resolved),
        "root-provider bundle aggregate count/bytes are invalid",
    )
    expected_closure = {
        "root_provider_bundle.json",
        *(str(row[1]["path"]).replace("\\", "/") for row in resolved),
    }
    observed_closure = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
    }
    _require(
        observed_closure == expected_closure,
        "root-provider bundle contains unlisted or missing files",
    )
    verification = _read_json(verification_path, role="root-provider bundle verification")
    _require(
        verification.get("schema_version")
        == "PHAxis-root-provider-model-bundle-verification-1.0"
        and verification.get("status") == "pass"
        and verification.get("bundle_id") == payload.get("bundle_id")
        and verification.get("bundle_identity_sha256")
        == payload.get("bundle_identity_sha256")
        and verification.get("files_verified") == len(records)
        and verification.get("bytes_verified") == payload.get("bytes")
        and verification.get("exact_file_closure_required") is True
        and verification.get("exact_file_closure_passed") is True
        and verification.get("unlisted_file_count") == 0
        and verification.get("missing_closure_file_count") == 0
        and verification.get("blind_images_used") == 0,
        "root-provider bundle verification does not bind the complete bundle",
    )
    return payload, resolved


def _workflow_locked_file(
    *, project_root: Path, base: Path, value: Any, role: str
) -> Path:
    _require(isinstance(value, Mapping), f"{role} locked file reference is absent")
    supplied = Path(str(value.get("path") or ""))
    source = supplied if supplied.is_absolute() else base / supplied
    source = _resolve_project_file(project_root, source, field=role)
    digest = str(value.get("sha256") or "").casefold()
    _require(_is_sha256(digest), f"{role} SHA-256 is invalid")
    _require(_sha256_file(source) == digest, f"{role} SHA-256 mismatch")
    return source


def _capsule_relative(package_path: str) -> str:
    """Return a manifest-relative path that remains inside the capsule."""

    safe = _safe_relative(package_path, field="portable capsule package path")
    relative = posixpath.relpath(safe, "model/examples/clean_install")
    _require(
        not relative.startswith("../../../.."),
        f"portable capsule reference escapes capsule: {package_path}",
    )
    return relative


def _portable_capsule_sources(
    *,
    project_root: Path,
    applied_path: Path,
    candidate_path: Path,
    checkpoint_records: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Path],
    bundle_root: Path,
    root_manifest_path: Path,
    verification_path: Path,
    root_assets: Sequence[tuple[Path, Mapping[str, Any]]],
    example_root: Path,
    example_manifest: Mapping[str, Any],
    example_manifest_path: Path,
    example_receipt_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, tuple[Path | None, bytes | None, str]]]:
    """Create the complete relocatable raw-image-to-profile runtime closure.

    The returned source map is keyed by final package path.  Generated files
    carry bytes and copied files carry a source path; no entry depends on an
    unrecorded discovery rule.
    """

    base = example_manifest_path.parent
    root = example_manifest.get("root_provider")
    stageb = example_manifest.get("stageb")
    traits = example_manifest.get("traits")
    profiles = example_manifest.get(
        "distal_axis_profiles", example_manifest.get("profiles")
    )
    _require(
        isinstance(root, Mapping)
        and isinstance(stageb, Mapping)
        and isinstance(traits, Mapping)
        and isinstance(profiles, Mapping),
        "release-example workflow runtime sections are incomplete",
    )
    bundle_ref = root.get("bundle")
    _require(
        isinstance(bundle_ref, Mapping)
        and bundle_ref.get("registry_sha256") == _sha256_file(root_manifest_path),
        "release-example root bundle registry differs from the supplied bundle",
    )

    sources: dict[str, tuple[Path | None, bytes | None, str]] = {}

    def copied(package: str, source: Path, role: str) -> None:
        safe = _safe_relative(package, field=f"{role} package path")
        _require(safe not in sources, f"portable capsule path collision: {safe}")
        sources[safe] = (source, None, role)

    def generated(package: str, data: bytes, role: str) -> None:
        safe = _safe_relative(package, field=f"{role} package path")
        _require(safe not in sources, f"portable capsule path collision: {safe}")
        sources[safe] = (None, data, role)

    for source, record in zip(checkpoints, checkpoint_records, strict=True):
        copied(str(record["package_path"]), source, "stageb_checkpoint")
    for source, record in root_assets:
        copied(
            f"model/assets/root_provider/{record['path']}",
            source,
            "root_provider_asset",
        )
    copied(
        "model/assets/root_provider/root_provider_bundle.json",
        root_manifest_path,
        "root_provider_authority",
    )
    copied(
        "model/assets/root_provider/root_provider_bundle_verification.json",
        verification_path,
        "root_provider_authority",
    )
    copied(
        "model/assets/runtime/applied_model_contract.json",
        applied_path,
        "runtime_authority",
    )

    proposal = _workflow_locked_file(
        project_root=project_root,
        base=base,
        value=example_manifest.get("model_contract_proposal"),
        role="release-example model-contract proposal",
    )
    copied(
        "model/assets/runtime/model_contract_proposal.json",
        proposal,
        "runtime_authority",
    )

    stageb_authority_targets = {
        "candidate_manifest": "model/assets/runtime/stageb/candidate_manifest.json",
        "selected_model_metadata": "model/assets/runtime/stageb/selected_model_metadata.json",
        "selection_receipt": "model/assets/runtime/stageb/selection_receipt.json",
    }
    stageb_authorities: dict[str, Path] = {}
    for field, package in stageb_authority_targets.items():
        source = _workflow_locked_file(
            project_root=project_root,
            base=base,
            value=stageb.get(field),
            role=f"release-example Stage-B {field}",
        )
        stageb_authorities[field] = source
        copied(package, source, "runtime_authority")
    _require(
        _sha256_file(stageb_authorities["candidate_manifest"])
        == _sha256_file(candidate_path),
        "release-example candidate differs from formal model-asset candidate",
    )

    checkpoint_values = stageb.get("checkpoints")
    _require(
        isinstance(checkpoint_values, list) and len(checkpoint_values) == 5,
        "release-example does not bind exactly five Stage-B checkpoints",
    )
    for index, (value, record) in enumerate(
        zip(checkpoint_values, checkpoint_records, strict=True)
    ):
        source = _workflow_locked_file(
            project_root=project_root,
            base=base,
            value=value,
            role=f"release-example Stage-B checkpoint {index}",
        )
        _require(
            _sha256_file(source) == record["sha256"],
            f"release-example Stage-B checkpoint {index} differs from formal member",
        )

    profile = _workflow_locked_file(
        project_root=project_root,
        base=base,
        value=profiles.get("contract_json"),
        role="release-example distal-axis profile contract",
    )
    copied(
        "model/assets/runtime/distal_axis_profile_contract.json",
        profile,
        "runtime_authority",
    )

    root_targets = {
        "acquisition_gate": "model/assets/runtime/root/acquisition_gate.json",
        "deployment_metadata": "model/assets/runtime/root/deployment_metadata.csv",
        "canonical_manifest": "model/assets/runtime/root/canonical_manifest.csv",
        "deployment_manifest": "model/assets/runtime/root/deployment_manifest.csv",
        "deployment_lock": "model/assets/runtime/root/deployment_manifest_lock.json",
    }
    root_authorities: dict[str, Path] = {}
    for field, package in root_targets.items():
        source = _workflow_locked_file(
            project_root=project_root,
            base=base,
            value=root.get(field),
            role=f"release-example root {field}",
        )
        root_authorities[field] = source
        copied(package, source, "runtime_authority")
    reference_source: Path | None = None
    if root.get("reference_registry") is not None:
        reference_source = _workflow_locked_file(
            project_root=project_root,
            base=base,
            value=root.get("reference_registry"),
            role="release-example root reference registry",
        )
        copied(
            "model/assets/runtime/root/reference_registry.json",
            reference_source,
            "runtime_authority",
        )

    input_targets = {
        "root_input_manifest": _workflow_locked_file(
            project_root=project_root,
            base=base,
            value=root.get("input_manifest"),
            role="release-example root input manifest",
        ),
        "stageb_input_manifest": _workflow_locked_file(
            project_root=project_root,
            base=base,
            value=stageb.get("input_manifest", stageb.get("manifest")),
            role="release-example Stage-B input manifest",
        ),
        "traits_metadata": _workflow_locked_file(
            project_root=project_root,
            base=base,
            value=traits.get("metadata_csv"),
            role="release-example traits metadata",
        ),
    }
    for name, source in input_targets.items():
        copied(
            f"model/examples/clean_install/inputs/{name}.csv",
            source,
            "release_example_input",
        )
    release_example = example_manifest.get("release_example")
    _require(isinstance(release_example, Mapping), "release-example identity block is absent")
    source_relpath = str(release_example.get("source_image_relpath") or "")
    source_image = _resolve_project_file(
        project_root,
        base / source_relpath,
        field="release-example source image",
    )
    _require(
        _sha256_file(source_image)
        == str(release_example.get("source_image_sha256") or "").casefold(),
        "release-example source image differs from its workflow lock",
    )
    source_package = (
        "model/examples/clean_install/inputs/" + source_image.name
    )
    copied(source_package, source_image, "release_example_input")
    copied(
        "model/examples/clean_install/projection_receipt.json",
        example_receipt_path,
        "release_example_provenance",
    )

    portable = deepcopy(dict(example_manifest))
    portable.pop("manifest_identity_sha256", None)
    portable["model_contract_proposal"] = {
        "path": _capsule_relative(
            "model/assets/runtime/model_contract_proposal.json"
        ),
        "sha256": _sha256_file(proposal),
    }
    portable_root = dict(portable["root_provider"])
    portable_root["project"] = "."
    portable_root.pop("python_executable", None)
    portable_root["bundle"] = {
        "path": _capsule_relative("model/assets/root_provider"),
        "registry_sha256": _sha256_file(root_manifest_path),
        "bundle_identity_sha256": str(
            root["bundle"]["bundle_identity_sha256"]
        ).casefold(),
    }
    portable_root["input_manifest"] = {
        "path": "inputs/root_input_manifest.csv",
        "sha256": _sha256_file(input_targets["root_input_manifest"]),
    }
    portable_root["image_root"] = "inputs"
    for field, package in root_targets.items():
        portable_root[field] = {
            "path": _capsule_relative(package),
            "sha256": _sha256_file(root_authorities[field]),
        }
    # A one-image portability run cannot be compared to a 283-image cached
    # chain.  Preserve the reference authority in the capsule for provenance,
    # but deliberately omit the optional fresh-exact283 audit from execution.
    portable_root.pop("reference_registry", None)
    portable["root_provider"] = portable_root

    portable_stageb = dict(portable["stageb"])
    portable_stageb["input_manifest"] = {
        "path": "inputs/stageb_input_manifest.csv",
        "sha256": _sha256_file(input_targets["stageb_input_manifest"]),
    }
    portable_stageb["image_root"] = "inputs"
    portable_stageb["checkpoints"] = [
        {
            "path": _capsule_relative(str(record["package_path"])),
            "sha256": str(record["sha256"]),
        }
        for record in checkpoint_records
    ]
    for field, package in stageb_authority_targets.items():
        portable_stageb[field] = {
            "path": _capsule_relative(package),
            "sha256": _sha256_file(stageb_authorities[field]),
        }
    portable["stageb"] = portable_stageb
    portable["traits"] = {
        **dict(portable["traits"]),
        "metadata_csv": {
            "path": "inputs/traits_metadata.csv",
            "sha256": _sha256_file(input_targets["traits_metadata"]),
        },
    }
    profile_key = "distal_axis_profiles" if "distal_axis_profiles" in portable else "profiles"
    portable[profile_key] = {
        **dict(portable[profile_key]),
        "contract_json": {
            "path": _capsule_relative(
                "model/assets/runtime/distal_axis_profile_contract.json"
            ),
            "sha256": _sha256_file(profile),
        },
    }
    if isinstance(portable.get("benchmark_contract"), Mapping):
        portable["benchmark_contract"] = {
            **dict(portable["benchmark_contract"]),
            "ordered_raw_source_manifest": {
                "path": "inputs/stageb_input_manifest.csv",
                "sha256": _sha256_file(input_targets["stageb_input_manifest"]),
            },
        }
    portable_release = dict(portable["release_example"])
    portable_release["source_image_relpath"] = (
        "inputs/" + source_image.name
    )
    portable_release.update(
        {
            "portable_capsule_finalized": True,
            "authoring_workspace_paths_required": False,
            "runtime_python_inherited_from_clean_environment": True,
            "exact283_reference_registry_executed_for_one_task": False,
        }
    )
    portable["release_example"] = portable_release
    portable["manifest_identity_sha256"] = _sha256_json(portable)
    portable_bytes = _json_bytes(portable)
    generated(
        "model/examples/clean_install/release_example_manifest.json",
        portable_bytes,
        "release_example_input",
    )

    runtime_authority_records = []
    for package, (source, data, role) in sorted(sources.items()):
        if role not in {"runtime_authority", "root_provider_authority"}:
            continue
        digest = _sha256_file(source) if source is not None else hashlib.sha256(data or b"").hexdigest()
        runtime_authority_records.append({"path": package, "sha256": digest})
    capsule_receipt: dict[str, Any] = {
        "schema_version": PORTABLE_CAPSULE_SCHEMA,
        "status": "completed_self_contained_raw_to_profiles_runtime",
        "workflow_manifest_sha256": hashlib.sha256(portable_bytes).hexdigest(),
        "workflow_manifest_identity_sha256": portable["manifest_identity_sha256"],
        "source_projection_manifest_sha256": _sha256_file(example_manifest_path),
        "source_projection_receipt_sha256": _sha256_file(example_receipt_path),
        "runtime_authorities": runtime_authority_records,
        "runtime_authority_identity_sha256": _sha256_json(runtime_authority_records),
        "root_reference_registry_packaged": reference_source is not None,
        "root_reference_registry_executed_for_one_task": False,
        "root_subprocess_python_rebound_to_active_interpreter": True,
        "authoring_workspace_paths_required": False,
        "canonical_annotations_read": False,
        "condition_metadata_used_for_routing": False,
        "blind_images_used": 0,
    }
    capsule_receipt["portable_capsule_identity_sha256"] = _sha256_json(
        capsule_receipt
    )
    generated(
        "model/examples/clean_install/receipt.json",
        _json_bytes(capsule_receipt),
        "release_example_input",
    )

    records: list[dict[str, Any]] = []
    for package, (source, data, role) in sorted(sources.items()):
        if source is not None:
            digest = _sha256_file(source)
            size = source.stat().st_size
        else:
            assert data is not None
            digest = hashlib.sha256(data).hexdigest()
            size = len(data)
        record: dict[str, Any] = {
            "role": role,
            "path": package,
            "sha256": digest,
            "bytes": size,
        }
        checkpoint = next(
            (
                row
                for row in checkpoint_records
                if str(row["package_path"]) == package
            ),
            None,
        )
        if checkpoint is not None:
            record["member_index"] = checkpoint["member_index"]
            record["seed"] = checkpoint["seed"]
        records.append(record)
    return portable, records, sources


def build_model_asset_manifest(
    *,
    project_root: str | Path,
    applied_model_contract: str | Path,
    candidate_manifest: str | Path,
    checkpoint_paths: Sequence[str | Path],
    root_provider_bundle_root: str | Path,
    root_provider_bundle_manifest: str | Path,
    root_provider_verification_receipt: str | Path,
    release_example_root: str | Path,
    portable_capsule_output: str | Path,
    bundle_manifest_output: str | Path,
    release_attestation: str | Path,
    output: str | Path,
    execute: bool = False,
) -> dict[str, Any]:
    """Seal checkpoints, root bundle, and the portable real release example."""

    root = Path(project_root).resolve()
    if execute:
        destination = _recoverable_project_output(
            root,
            output,
            field="model asset manifest output",
            directory=False,
        )
        bundle_destination = _recoverable_project_output(
            root,
            bundle_manifest_output,
            field="combined model bundle manifest output",
            directory=False,
        )
        capsule_destination = _recoverable_project_output(
            root,
            portable_capsule_output,
            field="portable model/runtime capsule output",
            directory=True,
        )
    else:
        destination = _planned_project_output(
            root, output, field="model asset manifest output"
        )
        bundle_destination = _planned_project_output(
            root, bundle_manifest_output, field="combined model bundle manifest output"
        )
        capsule_destination = _planned_project_output(
            root, portable_capsule_output, field="portable model/runtime capsule output"
        )
    _require(
        len({destination, bundle_destination, capsule_destination}) == 3,
        "model asset outputs collide",
    )
    applied_path = _resolve_project_file(
        root, applied_model_contract, field="applied model contract"
    )
    candidate_path = _resolve_project_file(
        root, candidate_manifest, field="train399 candidate manifest"
    )
    bundle_root = _resolve_project_directory(
        root, root_provider_bundle_root, field="root-provider bundle root"
    )
    root_manifest_path = _resolve_project_file(
        root, root_provider_bundle_manifest, field="root-provider bundle manifest"
    )
    verification_path = _resolve_project_file(
        root,
        root_provider_verification_receipt,
        field="root-provider bundle verification receipt",
    )
    example_root = _resolve_project_directory(
        root, release_example_root, field="clean-install release-example root"
    )
    attestation_path, _attestation = validate_release_attestation(
        root, release_attestation, required_role="model_asset_manifest"
    )
    candidate, applied, members = _validate_candidate_and_applied(
        candidate_path=candidate_path, applied_path=applied_path
    )
    _require(len(checkpoint_paths) == 5, "exactly five explicit checkpoint paths are required")
    checkpoints = [
        _resolve_project_file(root, path, field=f"train399 checkpoint member {index}")
        for index, path in enumerate(checkpoint_paths)
    ]
    _require(len(set(checkpoints)) == 5, "train399 checkpoint source paths are not distinct")
    checkpoint_records: list[dict[str, Any]] = []
    for index, (source, member) in enumerate(zip(checkpoints, members, strict=True)):
        digest = _sha256_file(source)
        _require(
            digest == member.get("checkpoint_sha256"),
            f"checkpoint member {index} differs from the sealed train399 candidate",
        )
        checkpoint_records.append(
            {
                "member_index": index,
                "seed": FORMAL_TRAIN399_SEEDS[index],
                "sha256": digest,
                "bytes": source.stat().st_size,
                "package_path": (
                    f"model/assets/stageb/member-{index}-seed-{FORMAL_TRAIN399_SEEDS[index]}"
                    f"{source.suffix.casefold()}"
                ),
            }
        )
    root_bundle, root_assets = _validate_root_bundle(
        project_root=root,
        bundle_root=bundle_root,
        manifest_path=root_manifest_path,
        verification_path=verification_path,
    )
    example_receipt_path = _resolve_project_file(
        root, example_root / "receipt.json", field="release-example suite receipt"
    )
    example_manifest_path = _resolve_project_file(
        root,
        example_root / "release_example_manifest.json",
        field="release-example workflow manifest",
    )
    example_receipt = _read_json(
        example_receipt_path, role="release-example suite receipt"
    )
    example_identity = _sealed_identity(
        example_receipt,
        "sample_input_suite_identity_sha256",
        role="release-example suite receipt",
    )
    example_manifest = _read_json(
        example_manifest_path, role="release-example workflow manifest"
    )
    example_manifest_identity = _sealed_identity(
        example_manifest,
        "manifest_identity_sha256",
        role="release-example workflow manifest",
    )
    _require(
        example_receipt.get("schema_version")
        == "PHAxis-clean-install-sample-input-suite-1.0"
        and example_receipt.get("status")
        == "completed_real_nonblind_release_example_manifest"
        and example_receipt.get("input_kind") == "real_nonblind_release_example"
        and example_receipt.get("release_authorized") is True
        and example_receipt.get("development_or_synthetic_smoke") is False
        and example_receipt.get("tasks") == 1
        and example_receipt.get("release_example_manifest_sha256")
        == _sha256_file(example_manifest_path)
        and example_receipt.get("release_example_manifest_identity_sha256")
        == example_manifest_identity
        and example_receipt.get("blind_images_used") == 0,
        "release-example suite is not the final real/nonblind one-task authority",
    )
    _require(
        example_manifest.get("schema_version")
        == "PHAxis-analysis-workflow-manifest-1.0"
        and isinstance(example_manifest.get("release_example"), Mapping)
        and example_manifest["release_example"].get("tasks") == 1
        and example_manifest["release_example"].get("release_authorized") is True
        and example_manifest["release_example"].get("blind_images_used") == 0,
        "release-example workflow manifest contract changed",
    )
    example_assets: list[tuple[Path, dict[str, Any]]] = []
    for source in sorted(example_root.rglob("*"), key=lambda path: path.as_posix()):
        _require(not source.is_symlink(), f"release-example asset is symlinked: {source}")
        if not source.is_file():
            continue
        relative = source.relative_to(example_root).as_posix()
        _safe_relative(relative, field="release-example asset path")
        example_assets.append(
            (
                source,
                {
                    "path": relative,
                    "sha256": _sha256_file(source),
                    "bytes": source.stat().st_size,
                },
            )
        )
    _require(
        len(example_assets) >= 6
        and {record["path"] for _source, record in example_assets}.issuperset(
            {
                "receipt.json",
                "release_example_manifest.json",
                "inputs/root_input_manifest.csv",
                "inputs/stageb_input_manifest.csv",
                "inputs/traits_metadata.csv",
            }
        )
        and any(
            record["path"].startswith("inputs/sample_source_image")
            for _source, record in example_assets
        ),
        "release-example suite file closure is incomplete",
    )
    root_expert = applied.get("root_expert")
    _require(
        isinstance(root_expert, Mapping)
        and root_expert.get("bundle_identity_sha256")
        == root_bundle.get("bundle_identity_sha256"),
        "applied model contract root bundle identity differs from the supplied bundle",
    )
    portable_manifest, bundle_members, capsule_sources = _portable_capsule_sources(
        project_root=root,
        applied_path=applied_path,
        candidate_path=candidate_path,
        checkpoint_records=checkpoint_records,
        checkpoints=checkpoints,
        bundle_root=bundle_root,
        root_manifest_path=root_manifest_path,
        verification_path=verification_path,
        root_assets=root_assets,
        example_root=example_root,
        example_manifest=example_manifest,
        example_manifest_path=example_manifest_path,
        example_receipt_path=example_receipt_path,
    )
    bundle_members.sort(key=lambda row: str(row["path"]))
    promotion = applied["promotion"]
    hair_expert = applied["hair_identity_count_expert"]
    proposal_source = capsule_sources[
        "model/assets/runtime/model_contract_proposal.json"
    ][0]
    _require(
        proposal_source is not None
        and _sha256_file(proposal_source) == promotion["proposal_file_sha256"],
        "release-example proposal differs from the applied contract promotion source",
    )
    portable_example_members = [
        dict(record)
        for record in bundle_members
        if str(record["path"]).startswith("model/examples/clean_install/")
    ]
    portable_manifest_member = next(
        record
        for record in portable_example_members
        if record["path"]
        == "model/examples/clean_install/release_example_manifest.json"
    )
    capsule_receipt_source = capsule_sources[
        "model/examples/clean_install/receipt.json"
    ]
    _require(
        capsule_receipt_source[1] is not None,
        "portable capsule receipt was not generated",
    )
    portable_capsule_receipt = json.loads(
        bytes(capsule_receipt_source[1]).decode("utf-8")
    )
    model_bundle_payload: dict[str, Any] = {
        "schema_version": MODEL_BUNDLE_SCHEMA,
        "status": "completed_final_immutable_bundle",
        "product": PRODUCT,
        "product_version": VERSION,
        "model_bundle_id": applied.get("model_bundle_id"),
        "model_contract_proposal_sha256": promotion["proposal_file_sha256"],
        "model_contract_proposal_identity_sha256": promotion[
            "proposal_identity_sha256"
        ],
        "root_expert_id": root_expert.get("expert_id"),
        "root_bundle_identity_sha256": root_bundle.get("bundle_identity_sha256"),
        "hair_identity_count_expert": hair_expert.get("expert_id"),
        "member_count": len(bundle_members),
        "members": bundle_members,
        "bundle_sha256": _sha256_json(bundle_members),
        "bundle_size_bytes": sum(int(record["bytes"]) for record in bundle_members),
        "applied_model_contract_sha256": _sha256_file(applied_path),
        "applied_model_contract_identity_sha256": applied[
            "model_contract_identity_sha256"
        ],
        "train399_candidate_manifest_sha256": _sha256_file(candidate_path),
        "candidate_bundle_identity_sha256": candidate[
            "candidate_bundle_identity_sha256"
        ],
        "stageb_checkpoints": checkpoint_records,
        "root_provider_bundle": {
            "manifest_sha256": _sha256_file(root_manifest_path),
            "verification_receipt_sha256": _sha256_file(verification_path),
            "bundle_id": root_bundle.get("bundle_id"),
            "bundle_identity_sha256": root_bundle.get("bundle_identity_sha256"),
            "files": [dict(record) for _source, record in root_assets],
        },
        "release_example": {
            "sample_input_suite_identity_sha256": example_identity,
            "source_projection_workflow_manifest_sha256": _sha256_file(
                example_manifest_path
            ),
            "source_projection_workflow_manifest_identity_sha256": example_manifest_identity,
            "workflow_manifest_sha256": portable_manifest_member["sha256"],
            "workflow_manifest_identity_sha256": portable_manifest[
                "manifest_identity_sha256"
            ],
            "portable_capsule_identity_sha256": portable_capsule_receipt[
                "portable_capsule_identity_sha256"
            ],
            "files": portable_example_members,
            "files_identity_sha256": _sha256_json(portable_example_members),
            "input_kind": "real_nonblind_release_example",
            "tasks": 1,
            "authoring_workspace_paths_required": False,
            "runtime_python_inherited_from_clean_environment": True,
            "blind_images_used": 0,
        },
        "release_attestation_sha256": _sha256_file(attestation_path),
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
        "historical_or_provisional_backfill_used": False,
    }
    model_bundle_payload["model_bundle_manifest_identity_sha256"] = _sha256_json(
        model_bundle_payload
    )
    bundle_encoded = _json_bytes(model_bundle_payload)
    bundle_hash = hashlib.sha256(bundle_encoded).hexdigest()
    provenance = (
        f"applied_model_contract={_project_relative(root, applied_path)};"
        f"candidate={_project_relative(root, candidate_path)};"
        f"root_bundle={_project_relative(root, root_manifest_path)};"
        f"release_example={_project_relative(root, example_receipt_path)};"
        f"release_attestation={_project_relative(root, attestation_path)}"
    )
    capsule_relative = _project_relative(root, capsule_destination)
    rows: list[dict[str, Any]] = []
    for record in bundle_members:
        role = str(record["role"])
        rows.append(
            {
                "source_path": f"{capsule_relative}/{record['path']}",
                "package_path": record["path"],
                "sha256": record["sha256"],
                "bytes": record["bytes"],
                "provenance": provenance,
                "notes": "self-contained hash-locked portable runtime capsule member",
                "release_authorized": "true",
                "asset_role": role,
                "member_index": record.get("member_index", ""),
                "seed": record.get("seed", ""),
            }
        )
    rows.append(
        {
            "source_path": f"{capsule_relative}/model/assets/MODEL_BUNDLE_MANIFEST.json",
            "package_path": "model/assets/MODEL_BUNDLE_MANIFEST.json",
            "sha256": bundle_hash,
            "bytes": len(bundle_encoded),
            "provenance": provenance,
            "notes": "sealed final PHAxis model-bundle authority",
            "release_authorized": "true",
            "asset_role": "model_bundle_manifest",
            "member_index": "",
            "seed": "",
        }
    )
    rows.sort(key=lambda row: str(row["package_path"]))
    _require(
        len({str(row["package_path"]).casefold() for row in rows}) == len(rows),
        "model asset package paths collide",
    )
    encoded = _manifest_bytes(rows, MODEL_ASSET_FIELDS)
    report = _materialisation_report(
        project_root=root,
        role="model_asset_manifest",
        output=destination,
        rows=rows,
        encoded=encoded,
        execute=execute,
        extra={
            "stageb_checkpoints": 5,
            "root_provider_assets": len(root_assets) + 2,
            "runtime_authorities": sum(
                record["role"] == "runtime_authority"
                for record in bundle_members
            ),
            "release_example_assets": len(portable_example_members),
            "release_example_suite_identity_sha256": example_identity,
            "portable_capsule_output": capsule_relative,
            "portable_capsule_identity_sha256": portable_capsule_receipt[
                "portable_capsule_identity_sha256"
            ],
            "model_bundle_manifest_output": _project_relative(root, bundle_destination),
            "model_bundle_manifest_sha256": bundle_hash,
            "model_bundle_manifest_identity_sha256": model_bundle_payload[
                "model_bundle_manifest_identity_sha256"
            ],
        },
    )
    if execute:
        expected_closure = {
            *capsule_sources,
            "model/assets/MODEL_BUNDLE_MANIFEST.json",
        }

        def verify_capsule(path: Path, *, role: str) -> None:
            observed_closure = {
                member.relative_to(path).as_posix()
                for member in path.rglob("*")
                if member.is_file()
            }
            _require(
                observed_closure == expected_closure,
                f"{role} closure differs from its sealed members",
            )
            for record in bundle_members:
                target = path / Path(str(record["path"]))
                _require(
                    not target.is_symlink()
                    and _sha256_file(target) == record["sha256"]
                    and target.stat().st_size == int(record["bytes"]),
                    f"{role} member differs: {record['path']}",
                )
            capsule_bundle = path / "model/assets/MODEL_BUNDLE_MANIFEST.json"
            _require(
                not capsule_bundle.is_symlink()
                and capsule_bundle.stat().st_size == len(bundle_encoded)
                and _sha256_file(capsule_bundle) == bundle_hash,
                f"{role} model-bundle authority differs",
            )

        # Validate every pre-existing atomic target before creating a missing
        # sibling.  Thus mismatched external data fail closed without extending
        # the prefix, while an exact hard-crash prefix can be safely completed.
        existing_outputs: list[str] = []
        if capsule_destination.exists():
            verify_capsule(capsule_destination, role="existing portable capsule retry prefix")
            existing_outputs.append(_project_relative(root, capsule_destination))
        for path, data, role in (
            (bundle_destination, bundle_encoded, "existing model-bundle retry prefix"),
            (destination, encoded, "existing model-asset manifest retry prefix"),
        ):
            if path.exists():
                _require(
                    not path.is_symlink()
                    and path.stat().st_size == len(data)
                    and _sha256_file(path) == hashlib.sha256(data).hexdigest(),
                    f"{role} differs from the recomputed authority",
                )
                existing_outputs.append(_project_relative(root, path))

        capsule_destination.parent.mkdir(parents=True, exist_ok=True)
        staging: Path | None = None
        published_this_invocation: list[Path] = []
        try:
            if not capsule_destination.exists():
                staging = Path(
                    tempfile.mkdtemp(
                        prefix=f".{capsule_destination.name}.",
                        dir=capsule_destination.parent,
                    )
                )
                for package, (source, data, _role) in sorted(capsule_sources.items()):
                    target = staging / Path(package)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if source is not None:
                        shutil.copyfile(source, target)
                    else:
                        assert data is not None
                        target.write_bytes(data)
                capsule_bundle = staging / "model/assets/MODEL_BUNDLE_MANIFEST.json"
                capsule_bundle.parent.mkdir(parents=True, exist_ok=True)
                capsule_bundle.write_bytes(bundle_encoded)
                verify_capsule(staging, role="portable capsule staging")
                os.replace(staging, capsule_destination)
                staging = None
                published_this_invocation.append(capsule_destination)
            if not bundle_destination.exists():
                _publish_no_overwrite(bundle_destination, bundle_encoded)
                published_this_invocation.append(bundle_destination)
            if not destination.exists():
                _publish_no_overwrite(destination, encoded)
                published_this_invocation.append(destination)
        except BaseException:
            if staging is not None and staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            for published in reversed(published_this_invocation):
                if published.is_dir():
                    shutil.rmtree(published, ignore_errors=True)
                else:
                    published.unlink(missing_ok=True)
            raise
        report.pop("plan_identity_sha256")
        report["publication_protocol"] = (
            "atomic_no_overwrite_targets_verify_exact_prefix_then_complete_missing"
        )
        report["recovered_existing_outputs"] = sorted(existing_outputs)
        report["outputs_created_this_invocation"] = sorted(
            _project_relative(root, path) for path in published_this_invocation
        )
        report["partial_publish_recovery_used"] = bool(existing_outputs)
        report["plan_identity_sha256"] = _sha256_json(report)
    return report


def _benchmark_json_record(
    *,
    project_root: Path,
    row: Mapping[str, str],
    role: str,
) -> tuple[Path, dict[str, Any]]:
    source = _resolve_project_file(
        project_root, row["source_path"], field=f"benchmark inventory {role}"
    )
    return source, _read_json(source, role=f"benchmark inventory {role}")


def _validate_benchmark_summary(
    payload: Mapping[str, Any], *, schema: str, mode: str | None, hardware: str, role: str
) -> None:
    _require(
        payload.get("schema_version") == schema
        and payload.get("status") == "completed_direct_full283"
        and payload.get("images") == 283
        and payload.get("hardware_identity_sha256") == hardware
        and payload.get("blind_images_used") == 0
        and payload.get("rootcap_region_metric") is False,
        f"{role} is not a blind-free same-hardware direct full283 summary",
    )
    if mode is not None:
        _require(payload.get("benchmark_mode") == mode, f"{role} benchmark mode is invalid")
    _sealed_identity(payload, "summary_identity_sha256", role=role)


def build_benchmark_manifest(
    *,
    project_root: str | Path,
    same_hardware_receipt: str | Path,
    artifact_inventory: str | Path,
    release_attestation: str | Path,
    output: str | Path,
    execute: bool = False,
) -> dict[str, Any]:
    """Validate and project the complete same-hardware benchmark evidence set."""

    root = Path(project_root).resolve()
    destination = _planned_project_output(root, output, field="benchmark manifest output")
    receipt_path = _resolve_project_file(
        root, same_hardware_receipt, field="same-hardware benchmark receipt"
    )
    inventory_path = _resolve_project_file(
        root, artifact_inventory, field="benchmark artifact inventory"
    )
    attestation_path, _attestation = validate_release_attestation(
        root, release_attestation, required_role="benchmark_manifest"
    )
    receipt = _read_json(receipt_path, role="same-hardware benchmark receipt")
    _require(
        receipt.get("schema_version") == "PHAxis-same-hardware-benchmark-receipt-1.0"
        and receipt.get("status") == "passed"
        and receipt.get("product") == PRODUCT
        and receipt.get("product_version") == VERSION
        and receipt.get("images") == 283
        and receipt.get("blind_images_used") == 0,
        "same-hardware benchmark receipt is not passed exact283",
    )
    _sealed_identity(receipt, "receipt_identity_sha256", role="same-hardware benchmark receipt")
    hardware = receipt.get("hardware_identity_sha256")
    runs = receipt.get("runs")
    _require(
        _is_sha256(hardware)
        and isinstance(runs, list)
        and len(runs) == 4
        and all(
            isinstance(run, Mapping)
            and run.get("hardware_identity_sha256") == hardware
            for run in runs
        ),
        "same-hardware benchmark receipt run/hardware closure is invalid",
    )
    inventory = _read_csv(
        inventory_path,
        required=(*COMMON_FIELDS, "artifact_role"),
        role="benchmark artifact inventory",
    )
    _require(bool(inventory), "benchmark artifact inventory is empty")
    rows: list[dict[str, Any]] = []
    by_role: dict[str, list[dict[str, Any]]] = {}
    package_paths: set[str] = set()
    for row_index, raw in enumerate(inventory, 2):
        source_value = _safe_relative(raw.get("source_path"), field="benchmark source_path")
        package = _safe_relative(raw.get("package_path"), field="benchmark package_path")
        _require(package.startswith("model/benchmark/"), f"benchmark row {row_index} leaves model/benchmark")
        key = package.casefold()
        _require(key not in package_paths, f"benchmark package path collides: {package}")
        package_paths.add(key)
        role = str(raw.get("artifact_role", "")).strip()
        _require(role, f"benchmark row {row_index} lacks artifact_role")
        _require(
            str(raw.get("release_authorized", "")).strip().casefold() == "true",
            f"benchmark row {row_index} is not release-authorized",
        )
        _require(
            bool(str(raw.get("provenance", "")).strip())
            and bool(str(raw.get("notes", "")).strip()),
            f"benchmark row {row_index} lacks provenance/notes",
        )
        source = _resolve_project_file(root, source_value, field=f"benchmark artifact {role}")
        _validate_declared_file(
            path=source,
            expected_sha256=raw.get("sha256"),
            expected_bytes=raw.get("bytes"),
            role=f"benchmark artifact {role}",
        )
        row = {
            "source_path": source_value,
            "package_path": package,
            "sha256": str(raw["sha256"]),
            "bytes": int(raw["bytes"]),
            "provenance": str(raw["provenance"]),
            "notes": str(raw["notes"]),
            "release_authorized": "true",
            "artifact_role": role,
        }
        rows.append(row)
        by_role.setdefault(role, []).append(row)
    exact_roles = (
        "same_hardware_receipt",
        "phaxis_production_summary",
        "v1_production_summary",
        "phaxis_sequential_summary",
        "v1_sequential_summary",
        "production_comparison_receipt",
        "sequential_comparison_receipt",
    )
    for role in exact_roles:
        _require(len(by_role.get(role, ())) == 1, f"benchmark inventory must contain exactly one {role}")
    _require(len(by_role.get("per_image_latency_csv", ())) >= 2, "benchmark inventory lacks per-image latency CSVs")
    _require(bool(by_role.get("gpu_telemetry")), "benchmark inventory lacks GPU telemetry evidence")
    _require(bool(by_role.get("hardware_preflight")), "benchmark inventory lacks hardware preflight evidence")
    receipt_row = by_role["same_hardware_receipt"][0]
    _require(
        _resolve_project_file(root, receipt_row["source_path"], field="inventory same-hardware receipt")
        == receipt_path
        and receipt_row["sha256"] == _sha256_file(receipt_path),
        "benchmark inventory same-hardware receipt differs from the bound receipt",
    )
    json_payloads: dict[str, tuple[Path, dict[str, Any]]] = {
        role: _benchmark_json_record(project_root=root, row=by_role[role][0], role=role)
        for role in exact_roles
        if role != "same_hardware_receipt"
    }
    production_schema = "PHAxis-full-workflow-production-batch-benchmark-1.0"
    sequential_schema = "PHAxis-full-workflow-sequential-latency-benchmark-1.0"
    for role in ("phaxis_production_summary", "v1_production_summary"):
        _validate_benchmark_summary(
            json_payloads[role][1],
            schema=production_schema,
            mode="production_batch_full283",
            hardware=str(hardware),
            role=role,
        )
    sequential_modes: set[str] = set()
    latency_hashes = {row["sha256"] for row in by_role["per_image_latency_csv"]}
    for role in ("phaxis_sequential_summary", "v1_sequential_summary"):
        payload = json_payloads[role][1]
        _validate_benchmark_summary(
            payload,
            schema=sequential_schema,
            mode=None,
            hardware=str(hardware),
            role=role,
        )
        mode = str(payload.get("benchmark_mode", ""))
        _require(
            mode in {"sequential_persistent_full283", "sequential_cold_cli_full283"},
            f"{role} has an unsupported sequential mode",
        )
        sequential_modes.add(mode)
        _require(
            payload.get("per_image_csv_sha256") in latency_hashes,
            f"{role} per-image latency CSV is absent from inventory",
        )
    _require(len(sequential_modes) == 1, "PHAxis/v1 sequential benchmark modes differ")
    comparisons = (
        (
            "production_comparison_receipt",
            "phaxis_production_summary",
            "v1_production_summary",
        ),
        (
            "sequential_comparison_receipt",
            "phaxis_sequential_summary",
            "v1_sequential_summary",
        ),
    )
    for comparison_role, phaxis_role, v1_role in comparisons:
        payload = json_payloads[comparison_role][1]
        _require(
            payload.get("schema_version")
            == "PHAxis-full-workflow-benchmark-comparison-1.0"
            and payload.get("status") == "comparable_direct_full283"
            and payload.get("comparable") is True
            and payload.get("same_283_source_manifest_hardware_and_io_scope") is True
            and payload.get("phaxis_summary_sha256")
            == by_role[phaxis_role][0]["sha256"]
            and payload.get("baseline_summary_sha256")
            == by_role[v1_role][0]["sha256"]
            and payload.get("blind_images_used") == 0
            and payload.get("rootcap_region_metric") is False,
            f"{comparison_role} does not bind the corresponding PHAxis/v1 summaries",
        )
        _sealed_identity(payload, "comparison_identity_sha256", role=comparison_role)
    expected_run_bindings = {
        "phaxis_production": "phaxis_production_summary",
        "phaxis_sequential": "phaxis_sequential_summary",
        "frozen_v1_production": "v1_production_summary",
        "frozen_v1_sequential": "v1_sequential_summary",
    }
    receipt_runs = {
        str(run.get("role", "")): run
        for run in runs
        if isinstance(run, Mapping)
    }
    _require(
        set(receipt_runs) == set(expected_run_bindings),
        "same-hardware receipt does not bind exactly the four inventoried PHAxis/v1 runs",
    )
    for receipt_role, inventory_role in expected_run_bindings.items():
        run = receipt_runs[receipt_role]
        summary = json_payloads[inventory_role][1]
        _require(
            run.get("summary_sha256") == by_role[inventory_role][0]["sha256"]
            and run.get("summary_identity_sha256")
            == summary.get("summary_identity_sha256")
            and run.get("mode") == summary.get("benchmark_mode")
            and run.get("hardware_identity_sha256") == hardware
            and run.get("fresh_direct_run") is True
            and run.get("resume_or_cache_used") is False
            and run.get("full_workflow_io_included") is True,
            f"same-hardware receipt run binding differs from inventory: {receipt_role}",
        )
    receipt_comparisons = receipt.get("comparisons")
    _require(
        isinstance(receipt_comparisons, Mapping),
        "same-hardware receipt lacks comparison bindings",
    )
    for mode, inventory_role in (
        ("production", "production_comparison_receipt"),
        ("sequential", "sequential_comparison_receipt"),
    ):
        binding = receipt_comparisons.get(mode)
        comparison = json_payloads[inventory_role][1]
        _require(
            isinstance(binding, Mapping)
            and binding.get("comparison_sha256")
            == by_role[inventory_role][0]["sha256"]
            and binding.get("comparison_identity_sha256")
            == comparison.get("comparison_identity_sha256"),
            f"same-hardware receipt {mode} comparison differs from inventory",
        )
    _require(
        receipt.get("same_ordered_exact283_sources") is True
        and receipt.get("same_hardware_uuid_and_driver") is True
        and receipt.get("same_io_and_full_workflow_scope") is True
        and receipt.get("fresh_no_cache") is True
        and receipt.get("historical_98_47_min_component_receipt_used") is False
        and receipt.get("forward_only_runtime_used") is False
        and receipt.get("root_cap_region_statistics_included") is False,
        "same-hardware receipt boundary flags are not formal exact283",
    )
    rows.sort(key=lambda row: str(row["package_path"]))
    encoded = _manifest_bytes(rows, BENCHMARK_FIELDS)
    report = _materialisation_report(
        project_root=root,
        role="benchmark_manifest",
        output=destination,
        rows=rows,
        encoded=encoded,
        execute=execute,
        extra={
            "artifacts": len(rows),
            "hardware_identity_sha256": hardware,
            "sequential_mode": next(iter(sequential_modes)),
            "release_attestation": _project_relative(root, attestation_path),
        },
    )
    if execute:
        _publish_no_overwrite(destination, encoded)
    return report


def _contract_bytes(payload: Mapping[str, Any]) -> bytes:
    return _json_bytes(payload)


def assemble_handover_build_contract(
    *,
    project_root: str | Path,
    bindings: Mapping[str, str | Path],
    checkpoint_paths: Sequence[str | Path],
    release_attestation: str | Path,
    output: str | Path,
    execute: bool = False,
) -> dict[str, Any]:
    """Seal all 16 bindings and prove the result with the public inspector."""

    root = Path(project_root).resolve()
    destination = _planned_project_output(root, output, field="handover build contract output")
    _require(
        set(bindings) == set(handover.REQUIRED_BINDINGS),
        "handover build contract binding set is incomplete or expanded",
    )
    attestation_path, attestation = validate_release_attestation(root, release_attestation)
    binding_records: dict[str, dict[str, str]] = {}
    resolved_bindings: dict[str, Path] = {}
    for role in handover.REQUIRED_BINDINGS:
        path = _resolve_project_file(root, bindings[role], field=f"binding {role}")
        resolved_bindings[role] = path
        binding_records[role] = {
            "path": _project_relative(root, path),
            "sha256": _sha256_file(path),
        }
    _require(len(checkpoint_paths) == 5, "exactly five checkpoint paths are required")
    checkpoints = [
        _resolve_project_file(root, path, field=f"checkpoint member {index}")
        for index, path in enumerate(checkpoint_paths)
    ]
    checkpoint_hashes = [_sha256_file(path) for path in checkpoints]
    _require(
        len(set(checkpoints)) == 5
        and len(set(checkpoint_hashes)) == 5,
        "checkpoint files/hashes must be five distinct values",
    )
    asset_rows = _read_csv(
        resolved_bindings["model_asset_manifest"],
        required=(*COMMON_FIELDS, *MODEL_ASSET_FIELDS),
        role="model asset materialisation manifest",
    )
    try:
        asset_checkpoints = sorted(
            (
                row
                for row in asset_rows
                if row.get("asset_role") == "stageb_checkpoint"
            ),
            key=lambda row: int(row.get("member_index", "-1")),
        )
        member_indices = [int(row["member_index"]) for row in asset_checkpoints]
        member_seeds = tuple(int(row["seed"]) for row in asset_checkpoints)
    except (KeyError, TypeError, ValueError) as error:
        raise ProducerError(
            "model_asset_manifest checkpoint member_index/seed is invalid"
        ) from error
    _require(
        len(asset_checkpoints) == 5
        and member_indices == list(range(5))
        and member_seeds == FORMAL_TRAIN399_SEEDS
        and [row["sha256"] for row in asset_checkpoints] == checkpoint_hashes,
        "checkpoint order differs from model_asset_manifest",
    )
    contract: dict[str, Any] = {
        "schema_version": handover.CONTRACT_SCHEMA,
        "product": PRODUCT,
        "product_version": VERSION,
        "scope_attestation": {
            field: bool(attestation["scope_attestation"][field])
            for field in SCOPE_FIELDS
        },
        "license_attestation": dict(attestation),
        "bindings": binding_records,
        "train399_checkpoints": [
            {
                "member_index": index,
                "seed": FORMAL_TRAIN399_SEEDS[index],
                "sha256": checkpoint_hashes[index],
            }
            for index in range(5)
        ],
    }
    contract["contract_identity_sha256"] = _sha256_json(contract)
    encoded = _contract_bytes(contract)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".handover-contract-inspect-",
            suffix=".json",
            dir=root,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        inspection = handover.inspect_handover_contract(root, temporary)
    except handover.HandoverError as error:
        raise ProducerError(f"assembled contract failed handover inspector: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if execute:
        _publish_no_overwrite(destination, encoded)
        try:
            final_inspection = handover.inspect_handover_contract(root, destination)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise
        inspection = final_inspection
    report = {
        "schema_version": "PHAxis-handover-build-contract-assembly-report-1.0",
        "status": "created" if execute else "passed_check_only_not_written",
        "output": _project_relative(root, destination),
        "contract_identity_sha256": contract["contract_identity_sha256"],
        "contract_file_sha256": hashlib.sha256(encoded).hexdigest(),
        "bindings": len(binding_records),
        "checkpoint_sha256_in_member_order": checkpoint_hashes,
        "checks": inspection["checks"],
        "release_attestation_sha256": _sha256_file(attestation_path),
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
        "historical_or_provisional_backfill_used": False,
    }
    report["report_identity_sha256"] = _sha256_json(report)
    return report
