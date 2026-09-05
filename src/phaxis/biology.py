"""Leakage-safe biological cohort construction for PHAxis.

This module deliberately separates three concerns:

* every image digest is recomputed from bytes instead of trusting a manifest;
* HumanCurated443 overlap is used only to define a SHA-disjoint biological
  primary cohort, never as an accuracy or phenotype input;
* authoritative acquisition batches are audited without pretending that they
  are biological plates.

No annotation JSON, blind image, or final-validation image is opened here.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .io import atomic_write_json, read_json, sha256_file, sha256_json
from .model_contract_binding import (
    read_model_contract_authority,
    require_output_identity,
)


OVERLAP_SCHEMA = "PHAxis-biological-image-overlap-audit-1.0"
COHORT_SCHEMA = "PHAxis-biological-cohorts-1.0"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        return list(reader)


def _atomic_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(fields), extrasaction="ignore"
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        field: "" if row.get(field) is None else row.get(field, "")
                        for field in fields
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_text(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            for line in lines:
                handle.write(str(line).rstrip("\r\n") + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _require_empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _assert_nonblind_path(path: Path) -> None:
    forbidden = {"blind", "final-validation", "final_validation"}
    lowered = [part.casefold() for part in path.parts]
    if any(any(token in part for token in forbidden) for part in lowered):
        raise RuntimeError(f"refusing a blind/final-validation path: {path}")


def _require_columns(
    rows: Sequence[Mapping[str, str]], columns: Iterable[str], *, label: str
) -> None:
    if not rows:
        raise RuntimeError(f"{label} is empty")
    missing = sorted(set(columns) - set(rows[0]))
    if missing:
        raise RuntimeError(f"{label} is missing columns: {missing}")


def _unique_index(
    rows: Sequence[Mapping[str, str]], field: str, *, label: str
) -> dict[str, Mapping[str, str]]:
    returned: dict[str, Mapping[str, str]] = {}
    for row in rows:
        key = str(row.get(field, ""))
        if not key:
            raise RuntimeError(f"{label} has an empty {field}")
        if key in returned:
            raise RuntimeError(f"{label} has duplicate {field}: {key}")
        returned[key] = row
    return returned


def _hash_one(item: tuple[str, Path, str, Mapping[str, str]]) -> dict[str, Any]:
    task_id, path, declared, metadata = item
    if not path.is_file():
        raise FileNotFoundError(f"{task_id}: image is missing: {path}")
    observed = sha256_file(path)
    return {
        "task_id": task_id,
        "image_path": str(path.resolve()),
        "declared_image_sha256": declared.casefold(),
        "recomputed_image_sha256": observed,
        "declared_hash_matches_bytes": observed == declared.casefold(),
        "image_bytes": path.stat().st_size,
        **dict(metadata),
    }


def recompute_biological_overlap(
    *,
    human_dataset_root: str | Path,
    human_manifest: str | Path,
    biological_manifest: str | Path,
    output: str | Path,
    workers: int = 4,
    expected_human_images: int | None = 443,
    expected_biological_images: int | None = 283,
    expected_overlap_images: int | None = 22,
) -> dict[str, Any]:
    """Rehash both cohorts and derive overlap from observed byte digests."""

    human_dataset_root = Path(human_dataset_root).resolve()
    human_manifest = Path(human_manifest).resolve()
    biological_manifest = Path(biological_manifest).resolve()
    output = Path(output).resolve()
    for path in (human_dataset_root, human_manifest, biological_manifest, output):
        _assert_nonblind_path(path)
    if workers < 1 or workers > 16:
        raise ValueError("workers must be between 1 and 16")
    _require_empty_output(output)

    human_rows = _read_csv(human_manifest)
    biological_rows = _read_csv(biological_manifest)
    _require_columns(
        human_rows,
        ("task_id", "split", "image_relpath", "image_sha256", "family_key"),
        label="HumanCurated manifest",
    )
    _require_columns(
        biological_rows,
        ("task_id", "image_path", "image_sha256"),
        label="biological manifest",
    )
    _unique_index(human_rows, "task_id", label="HumanCurated manifest")
    _unique_index(biological_rows, "task_id", label="biological manifest")
    if expected_human_images is not None and len(human_rows) != expected_human_images:
        raise RuntimeError(
            f"HumanCurated count drift: {len(human_rows)} != {expected_human_images}"
        )
    if (
        expected_biological_images is not None
        and len(biological_rows) != expected_biological_images
    ):
        raise RuntimeError(
            "biological count drift: "
            f"{len(biological_rows)} != {expected_biological_images}"
        )

    human_jobs = [
        (
            row["task_id"],
            human_dataset_root / row["image_relpath"],
            row["image_sha256"],
            {
                "split": row["split"],
                "family_key": row["family_key"],
                "image_relpath": row["image_relpath"],
            },
        )
        for row in human_rows
    ]
    biological_jobs = [
        (
            row["task_id"],
            Path(row["image_path"]),
            row["image_sha256"],
            {},
        )
        for row in biological_rows
    ]
    for _task_id, path, _declared, _metadata in (*human_jobs, *biological_jobs):
        _assert_nonblind_path(path.resolve())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        human_audit = list(pool.map(_hash_one, human_jobs))
        biological_audit = list(pool.map(_hash_one, biological_jobs))
    if not all(bool(row["declared_hash_matches_bytes"]) for row in human_audit):
        bad = [
            row["task_id"]
            for row in human_audit
            if not row["declared_hash_matches_bytes"]
        ]
        raise RuntimeError(f"HumanCurated image hash mismatch: {bad[:10]}")
    if not all(
        bool(row["declared_hash_matches_bytes"]) for row in biological_audit
    ):
        bad = [
            row["task_id"]
            for row in biological_audit
            if not row["declared_hash_matches_bytes"]
        ]
        raise RuntimeError(f"biological image hash mismatch: {bad[:10]}")

    human_by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in human_audit:
        human_by_sha[row["recomputed_image_sha256"]].append(row)
    pairs: list[dict[str, Any]] = []
    overlap_biological_ids: set[str] = set()
    for biological in biological_audit:
        observed = biological["recomputed_image_sha256"]
        for human in human_by_sha.get(observed, ()):
            overlap_biological_ids.add(str(biological["task_id"]))
            pairs.append(
                {
                    "recomputed_image_sha256": observed,
                    "biological_task_id": biological["task_id"],
                    "human_task_id": human["task_id"],
                    "human_split": human["split"],
                    "human_family_key": human["family_key"],
                    "biological_image_path": biological["image_path"],
                    "human_image_path": human["image_path"],
                }
            )
    if (
        expected_overlap_images is not None
        and len(overlap_biological_ids) != expected_overlap_images
    ):
        raise RuntimeError(
            "recomputed overlap drift: "
            f"{len(overlap_biological_ids)} != {expected_overlap_images}"
        )

    human_audit_path = output / "human_curated443_hash_audit.csv"
    biological_audit_path = output / "biological283_hash_audit.csv"
    pairs_path = output / "overlap_pairs.csv"
    overlap_ids_path = output / "overlap_biological_task_ids.txt"
    clean_ids_path = output / "clean_biological_task_ids.txt"
    _atomic_csv(
        human_audit_path,
        human_audit,
        (
            "task_id",
            "split",
            "family_key",
            "image_relpath",
            "image_path",
            "declared_image_sha256",
            "recomputed_image_sha256",
            "declared_hash_matches_bytes",
            "image_bytes",
        ),
    )
    _atomic_csv(
        biological_audit_path,
        biological_audit,
        (
            "task_id",
            "image_path",
            "declared_image_sha256",
            "recomputed_image_sha256",
            "declared_hash_matches_bytes",
            "image_bytes",
        ),
    )
    _atomic_csv(
        pairs_path,
        pairs,
        (
            "recomputed_image_sha256",
            "biological_task_id",
            "human_task_id",
            "human_split",
            "human_family_key",
            "biological_image_path",
            "human_image_path",
        ),
    )
    _atomic_text(overlap_ids_path, sorted(overlap_biological_ids))
    clean_ids = sorted(
        str(row["task_id"])
        for row in biological_audit
        if row["task_id"] not in overlap_biological_ids
    )
    _atomic_text(clean_ids_path, clean_ids)
    split_counts = Counter(pair["human_split"] for pair in pairs)
    summary: dict[str, Any] = {
        "schema_version": OVERLAP_SCHEMA,
        "status": "completed_from_recomputed_image_bytes",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "human_images": len(human_audit),
        "biological_images": len(biological_audit),
        "human_unique_recomputed_sha256": len(human_by_sha),
        "biological_unique_recomputed_sha256": len(
            {row["recomputed_image_sha256"] for row in biological_audit}
        ),
        "overlap_biological_images": len(overlap_biological_ids),
        "overlap_pairs": len(pairs),
        "clean_biological_images": len(clean_ids),
        "overlap_human_split_counts": dict(sorted(split_counts.items())),
        "all_human_declared_hashes_match_recomputed_bytes": True,
        "all_biological_declared_hashes_match_recomputed_bytes": True,
        "overlap_source": "intersection_of_recomputed_sha256_not_legacy_id_list",
        "legacy_rhaxiscc_overlap_file_used": False,
        "human_manifest_sha256": sha256_file(human_manifest),
        "biological_manifest_sha256": sha256_file(biological_manifest),
        "output_sha256": {
            "human_hash_audit": sha256_file(human_audit_path),
            "biological_hash_audit": sha256_file(biological_audit_path),
            "overlap_pairs": sha256_file(pairs_path),
            "overlap_biological_task_ids": sha256_file(overlap_ids_path),
            "clean_biological_task_ids": sha256_file(clean_ids_path),
        },
        "human_annotations_read": False,
        "canonical_annotations_read": False,
        "condition_metadata_used_for_model_routing": False,
        "blind_images_used": 0,
    }
    summary["overlap_audit_identity_sha256"] = sha256_json(summary)
    atomic_write_json(output / "summary.json", summary)
    return summary


def _parse_bool(value: Any, *, field: str, task_id: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise RuntimeError(f"{task_id}: invalid boolean {field}={value!r}")


def _validate_trait_export(root: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    summary_path = root / "summary.json"
    summary = read_json(summary_path)
    if summary.get("status") != "completed" or summary.get("blind_images_used") != 0:
        raise RuntimeError("PHAxis trait export is incomplete or blind-tainted")
    guards = {
        "canonical_annotations_read": False,
        "root_cap_region_statistics_included": False,
        "whole_hair_zone_confirmatory_traits_allowed": False,
    }
    for field, expected in guards.items():
        if summary.get(field) is not expected:
            raise RuntimeError(f"PHAxis trait export guard failed: {field}")
    paths = {
        "traits": root / "traits.csv",
        "roots": root / "detailed_root_statistics.csv",
        "hairs": root / "hair_instances.csv",
        "image_traits": root / "image_traits.csv",
    }
    expected_hashes = {
        "traits": summary.get("traits_sha256"),
        "roots": summary.get("detailed_root_statistics_sha256"),
        "hairs": summary.get("hair_instances_sha256"),
        "image_traits": summary.get("image_traits_sha256"),
    }
    for name, path in paths.items():
        if not path.is_file() or sha256_file(path) != expected_hashes[name]:
            raise RuntimeError(f"PHAxis trait export hash mismatch: {name}")
    return summary, paths


def _subset_rows(
    rows: Sequence[Mapping[str, str]], task_ids: set[str]
) -> list[Mapping[str, str]]:
    return [row for row in rows if str(row.get("task_id")) in task_ids]


def _write_cohort_tables(
    *,
    output: Path,
    table_rows: Mapping[str, Sequence[Mapping[str, str]]],
) -> dict[str, str]:
    returned: dict[str, str] = {}
    for name, rows in table_rows.items():
        if not rows:
            raise RuntimeError(f"cohort table is empty: {name}")
        fields = tuple(rows[0].keys())
        path = output / f"{name}.csv"
        _atomic_csv(path, rows, fields)
        returned[name] = sha256_file(path)
    return returned


def _condition_table(
    membership: Sequence[Mapping[str, Any]], *, include_field: str
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in membership:
        if bool(row[include_field]):
            grouped[
                (
                    str(row["study_role"]),
                    str(row["experiment_key"]),
                    str(row["condition_code"]),
                )
            ].append(row)
    returned: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = grouped[key]
        returned.append(
            {
                "study_role": key[0],
                "experiment_key": key[1],
                "condition_code": key[2],
                "units": len(rows),
                "formal_statistics_eligible": sum(
                    bool(row["formal_statistics_eligible"]) for row in rows
                ),
                "recomputed_human443_overlap": sum(
                    bool(row["recomputed_human443_overlap"]) for row in rows
                ),
            }
        )
    return returned


def _batch_audit(
    membership: Sequence[Mapping[str, Any]], *, include_field: str, scope: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in membership:
        if bool(row[include_field]):
            grouped[str(row["acquisition_batch_id"])].append(row)
    returned: list[dict[str, Any]] = []
    for batch_id in sorted(grouped):
        rows = grouped[batch_id]
        conditions = sorted({str(row["condition_code"]) for row in rows})
        experiments = sorted({str(row["experiment_key"]) for row in rows})
        returned.append(
            {
                "scope": scope,
                "acquisition_batch_id": batch_id,
                "units": len(rows),
                "condition_count": len(conditions),
                "conditions": ";".join(conditions),
                "experiment_count": len(experiments),
                "experiments": ";".join(experiments),
                "spans_multiple_conditions": len(conditions) > 1,
                "is_biological_plate_id": False,
            }
        )
    return returned


def build_biological_cohorts(
    *,
    trait_export: str | Path,
    analysis_metadata: str | Path,
    design_manifest: str | Path,
    overlap_audit: str | Path,
    analysis_contract: str | Path,
    model_contract_proposal: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Build a clean SHA-disjoint primary and full sensitivity cohort.

    The function copies only PHAxis trait tables. It does not fit any model and
    therefore cannot inspect condition-specific effects.
    """

    trait_export = Path(trait_export).resolve()
    analysis_metadata = Path(analysis_metadata).resolve()
    design_manifest = Path(design_manifest).resolve()
    overlap_audit = Path(overlap_audit).resolve()
    analysis_contract = Path(analysis_contract).resolve()
    model_contract_proposal = Path(model_contract_proposal).resolve()
    output = Path(output).resolve()
    for path in (
        trait_export,
        analysis_metadata,
        design_manifest,
        overlap_audit,
        analysis_contract,
        model_contract_proposal,
        output,
    ):
        _assert_nonblind_path(path)
    _require_empty_output(output)

    proposal_binding = read_model_contract_authority(model_contract_proposal)
    proposal_fields = proposal_binding.receipt_fields()
    public_identity = proposal_binding.public_identity_fields()
    contract = read_json(analysis_contract)
    if contract.get("schema_version") != "PHAxis-biological-analysis-contract-1.0":
        raise RuntimeError("unexpected biological analysis contract")
    expected = contract["expected_cohort_counts"]
    trait_summary, trait_paths = _validate_trait_export(trait_export)
    require_output_identity(
        trait_summary,
        proposal_binding,
        role="PHAxis trait-export summary",
    )
    overlap_summary_path = overlap_audit / "summary.json"
    overlap_summary = read_json(overlap_summary_path)
    if (
        overlap_summary.get("schema_version") != OVERLAP_SCHEMA
        or overlap_summary.get("status")
        != "completed_from_recomputed_image_bytes"
        or overlap_summary.get("blind_images_used") != 0
    ):
        raise RuntimeError("overlap audit is invalid or blind-tainted")
    overlap_pairs_path = overlap_audit / "overlap_pairs.csv"
    if sha256_file(overlap_pairs_path) != overlap_summary["output_sha256"][
        "overlap_pairs"
    ]:
        raise RuntimeError("overlap pair table hash mismatch")

    traits = _read_csv(trait_paths["traits"])
    roots = _read_csv(trait_paths["roots"])
    hairs = _read_csv(trait_paths["hairs"])
    image_traits = _read_csv(trait_paths["image_traits"])
    metadata_rows = _read_csv(analysis_metadata)
    design_rows = _read_csv(design_manifest)
    pair_rows = _read_csv(overlap_pairs_path)
    trait_by_id = _unique_index(traits, "task_id", label="PHAxis traits")
    root_by_id = _unique_index(roots, "task_id", label="PHAxis root traits")
    image_by_id = _unique_index(
        image_traits, "task_id", label="PHAxis image traits"
    )
    metadata_by_id = _unique_index(
        metadata_rows, "task_id", label="analysis metadata"
    )
    design_by_id = _unique_index(
        design_rows, "biological_unit_id", label="canonical design manifest"
    )
    task_ids = set(trait_by_id)
    for label, indexed in (
        ("root traits", root_by_id),
        ("image traits", image_by_id),
        ("analysis metadata", metadata_by_id),
    ):
        if set(indexed) != task_ids:
            raise RuntimeError(f"task set mismatch: {label}")
    missing_design = sorted(task_ids - set(design_by_id))
    if missing_design:
        raise RuntimeError(f"tasks absent from canonical design manifest: {missing_design[:10]}")
    if len(task_ids) != int(trait_summary["tasks"]):
        raise RuntimeError("trait task count does not match trait summary")

    overlap_by_biological: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for pair in pair_rows:
        overlap_by_biological[pair["biological_task_id"]].append(pair)
    if set(overlap_by_biological) - task_ids:
        raise RuntimeError("overlap audit contains tasks absent from PHAxis export")

    membership: list[dict[str, Any]] = []
    for task_id in sorted(task_ids):
        trait = trait_by_id[task_id]
        metadata = metadata_by_id[task_id]
        design = design_by_id[task_id]
        source_sha = trait["source_image_sha256"].casefold()
        if source_sha != metadata["image_sha256"].casefold():
            raise RuntimeError(f"{task_id}: trait/metadata image hash mismatch")
        if source_sha != design["destination_sha256"].casefold():
            raise RuntimeError(f"{task_id}: trait/design image hash mismatch")
        pairs = overlap_by_biological.get(task_id, [])
        for pair in pairs:
            if source_sha != pair["recomputed_image_sha256"].casefold():
                raise RuntimeError(f"{task_id}: overlap digest mismatch")
        formal = _parse_bool(
            trait["formal_statistics_eligible"],
            field="formal_statistics_eligible",
            task_id=task_id,
        )
        membership.append(
            {
                "task_id": task_id,
                "source_image_sha256": source_sha,
                "primary_clean_sha_disjoint_include": not bool(pairs),
                "sensitivity_full_include": True,
                "recomputed_human443_overlap": bool(pairs),
                "overlap_human_task_ids": ";".join(
                    sorted(pair["human_task_id"] for pair in pairs)
                ),
                "overlap_human_splits": ";".join(
                    sorted({pair["human_split"] for pair in pairs})
                ),
                "formal_statistics_eligible": formal,
                "study_role": metadata["study_role"],
                "experiment_key": metadata["experiment_key"],
                "condition_code": metadata["condition_code"],
                "genotype_or_construct": metadata["genotype_or_construct"],
                "temperature_c": metadata["temperature_c"],
                "developmental_day": metadata["developmental_day"],
                "qc_disposition": metadata["qc_disposition"],
                "acquisition_batch_id": design["batch_id"],
                "acquisition_batch_folder": design["batch_folder"],
                "source_group_id": design["source_group"],
                "biological_plate_id": "",
                "biological_plate_id_status": (
                    "not_available_do_not_parse_from_filename"
                ),
            }
        )

    clean_ids = {
        str(row["task_id"])
        for row in membership
        if row["primary_clean_sha_disjoint_include"]
    }
    full_ids = {str(row["task_id"]) for row in membership}
    observed_counts = {
        "human_curated443": int(overlap_summary["human_images"]),
        "biological_full": len(full_ids),
        "human_curated_overlap": len(full_ids - clean_ids),
        "biological_clean": len(clean_ids),
    }
    for field, expected_value in expected.items():
        if int(expected_value) != observed_counts[field]:
            raise RuntimeError(
                f"cohort contract drift: {field}={observed_counts[field]} "
                f"!= {expected_value}"
            )

    clean_name = f"primary_clean{len(clean_ids)}"
    full_name = f"sensitivity_full{len(full_ids)}"
    tables = {
        "traits": traits,
        "detailed_root_statistics": roots,
        "hair_instances": hairs,
        "image_traits": image_traits,
    }
    clean_hashes = _write_cohort_tables(
        output=output / clean_name,
        table_rows={name: _subset_rows(rows, clean_ids) for name, rows in tables.items()},
    )
    full_hashes = _write_cohort_tables(
        output=output / full_name,
        table_rows={name: _subset_rows(rows, full_ids) for name, rows in tables.items()},
    )
    membership_path = output / "cohort_membership.csv"
    membership_fields = tuple(membership[0].keys())
    _atomic_csv(membership_path, membership, membership_fields)
    clean_condition = _condition_table(
        membership, include_field="primary_clean_sha_disjoint_include"
    )
    full_condition = _condition_table(
        membership, include_field="sensitivity_full_include"
    )
    condition_path = output / "cohort_condition_counts.csv"
    _atomic_csv(
        condition_path,
        [
            {"cohort": clean_name, **row} for row in clean_condition
        ]
        + [{"cohort": full_name, **row} for row in full_condition],
        (
            "cohort",
            "study_role",
            "experiment_key",
            "condition_code",
            "units",
            "formal_statistics_eligible",
            "recomputed_human443_overlap",
        ),
    )
    batch_rows = _batch_audit(
        membership,
        include_field="sensitivity_full_include",
        scope="full_biological_cohort",
    )
    primary_membership = [
        row
        for row in membership
        if row["study_role"] == contract["primary_model_scope"]["study_role"]
        and row["experiment_key"]
        == contract["primary_model_scope"]["experiment_key"]
    ]
    batch_rows.extend(
        _batch_audit(
            primary_membership,
            include_field="sensitivity_full_include",
            scope="D15_primary_full",
        )
    )
    batch_rows.extend(
        _batch_audit(
            primary_membership,
            include_field="primary_clean_sha_disjoint_include",
            scope="D15_primary_clean",
        )
    )
    batch_path = output / "acquisition_batch_condition_audit.csv"
    _atomic_csv(
        batch_path,
        batch_rows,
        (
            "scope",
            "acquisition_batch_id",
            "units",
            "condition_count",
            "conditions",
            "experiment_count",
            "experiments",
            "spans_multiple_conditions",
            "is_biological_plate_id",
        ),
    )

    full_batches = [
        row for row in batch_rows if row["scope"] == "full_biological_cohort"
    ]
    primary_full_batches = [
        row for row in batch_rows if row["scope"] == "D15_primary_full"
    ]
    lock: dict[str, Any] = {
        "schema_version": "PHAxis-biological-cohort-lock-1.0",
        "status": "postresult_software_transition_provenance_lock",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_claimed": False,
        "prior_predecessor_biological_effects_known": True,
        "phaxis_effects_inspected_before_lock": "not_asserted",
        "analysis_contract_sha256": sha256_file(analysis_contract),
        "trait_export_summary_sha256": sha256_file(trait_export / "summary.json"),
        "overlap_audit_summary_sha256": sha256_file(overlap_summary_path),
        "analysis_metadata_sha256": sha256_file(analysis_metadata),
        "canonical_design_manifest_sha256": sha256_file(design_manifest),
        "cohort_builder_implementation_sha256": sha256_file(Path(__file__)),
        "cohort_counts": observed_counts,
        "output_table_sha256": {
            clean_name: clean_hashes,
            full_name: full_hashes,
            "cohort_membership": sha256_file(membership_path),
            "cohort_condition_counts": sha256_file(condition_path),
            "acquisition_batch_condition_audit": sha256_file(batch_path),
        },
        "primary_clean_rule": (
            "exclude every biological image whose recomputed byte SHA-256 occurs "
            "anywhere in HumanCurated443, regardless of train/val split"
        ),
        "sensitivity_rule": "retain all 283 biological images and label overlap explicitly",
        "biological_plate_id_available": False,
        "filename_regex_plate_inference_used": False,
        "acquisition_batch_treated_as_biological_plate": False,
        "root_cap_region_statistics_allowed": False,
        "whole_hair_zone_confirmatory_traits_allowed": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        **proposal_fields,
        **public_identity,
    }
    lock["cohort_lock_identity_sha256"] = sha256_json(lock)
    lock_path = output / "analysis_contract_lock.json"
    atomic_write_json(lock_path, lock)

    primary_full_conditions = sorted(
        {
            str(row["condition_code"])
            for row in primary_membership
            if row["sensitivity_full_include"]
        }
    )
    summary: dict[str, Any] = {
        "schema_version": COHORT_SCHEMA,
        "status": "completed_without_fitting_biological_effect_models",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_directories": {
            "primary": clean_name,
            "sensitivity": full_name,
        },
        "counts": observed_counts,
        "primary_clean_formal_statistics_eligible": sum(
            bool(row["formal_statistics_eligible"])
            and bool(row["primary_clean_sha_disjoint_include"])
            for row in membership
        ),
        "sensitivity_full_formal_statistics_eligible": sum(
            bool(row["formal_statistics_eligible"]) for row in membership
        ),
        "primary_model_scope": {
            **contract["primary_model_scope"],
            "full_units": len(primary_membership),
            "clean_units": sum(
                bool(row["primary_clean_sha_disjoint_include"])
                for row in primary_membership
            ),
            "conditions": primary_full_conditions,
        },
        "design_identifiability": {
            "authoritative_acquisition_batch_ids_full": len(full_batches),
            "authoritative_acquisition_batches_spanning_multiple_conditions_full": sum(
                bool(row["spans_multiple_conditions"]) for row in full_batches
            ),
            "authoritative_acquisition_batch_ids_D15_primary": len(
                primary_full_batches
            ),
            "authoritative_acquisition_batches_spanning_multiple_conditions_D15_primary": sum(
                bool(row["spans_multiple_conditions"])
                for row in primary_full_batches
            ),
            "biological_plate_id_available": False,
            "plate_treatment_confounding_status": (
                "not_identifiable_from_authoritative_metadata; acquisition batch "
                "is not a biological plate and must not be relabelled as one"
            ),
            "legacy_rhaxiscc_filename_regex_plate_claim_reused": False,
            "six_group_genotype_effect_estimable": False,
            "six_group_reason": (
                "WT and RHD6 constructs occupy different experiment/study-role "
                "strata; six-group comparisons are descriptive only"
            ),
            "causal_treatment_claim_allowed": False,
            "D15_claim_scope": (
                "within-experiment exploratory association with model-based "
                "uncertainty; no plate-randomized causal generalization"
            ),
        },
        "statistical_transition": {
            "predecessor_model_code_may_be_reused": True,
            "required_changes": [
                "clean SHA-disjoint cohort is primary; full283 is sensitivity",
                "restrict factorial model to D15 RHD6-EV/RHD6-OE x 22/30C",
                "apply formal_statistics_eligible and endpoint observability gates",
                "do not infer a plate ID from image filenames",
                "report all-six-condition atlas descriptively",
                "label p-values exploratory/model-based, not confirmatory plate inference",
            ],
            "effect_model_fitted_by_this_step": False,
        },
        "input_sha256": {
            "trait_export_summary": sha256_file(trait_export / "summary.json"),
            "overlap_audit_summary": sha256_file(overlap_summary_path),
            "analysis_contract": sha256_file(analysis_contract),
            "analysis_metadata": sha256_file(analysis_metadata),
            "canonical_design_manifest": sha256_file(design_manifest),
        },
        "output_sha256": {
            "analysis_contract_lock": sha256_file(lock_path),
            "cohort_membership": sha256_file(membership_path),
            "cohort_condition_counts": sha256_file(condition_path),
            "acquisition_batch_condition_audit": sha256_file(batch_path),
            clean_name: clean_hashes,
            full_name: full_hashes,
        },
        "root_cap_region_statistics_included": False,
        "whole_hair_zone_confirmatory_traits_included": False,
        "biological_effect_models_fitted": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        **proposal_fields,
        **public_identity,
    }
    summary["cohort_build_identity_sha256"] = sha256_json(summary)
    atomic_write_json(output / "summary.json", summary)
    return summary
