"""Create-only, source-preserving PHAxis supplementary Table/Data S1--S10.

The publication figure stage is the formal owner of this bundle.  This module
contains no model execution and never discovers evidence from the workspace:
callers must provide every source file under an explicit, named role.  Each
reviewer-facing item has an independently sealed receipt, source-file hashes,
and an explicit denominator contract.  The bundle receipt seals the exact
ten-item/file closure and is reusable by the supplement, DOCX, evidence-graph,
and artifact-QA gates.
"""

from __future__ import annotations

from copy import deepcopy
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json
from phaxis.biological_analysis import (
    RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
    RAW_EFFECT_HC3_INTERVAL,
    RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL,
    RAW_EFFECT_OLS_MEAN_CONTRAST,
    raw_median_bootstrap_seed,
)
from phaxis.multitrait_atlas import (
    EFFECT_NAME_TO_KEY,
    EFFECT_KEYS,
    GROUP_ORDER,
    H11_ENDPOINT,
    H11_RAW_BOOTSTRAP_BASE_SEED,
    H11_RAW_BOOTSTRAP_REPLICATES,
    PRIMARY_ENDPOINTS,
)
from phaxis.publication_evidence import (
    validate_wt_secondary_analysis_binding,
    validate_wt_secondary_evidence,
)


BUNDLE_SCHEMA = "PHAxis-supplementary-table-data-bundle-1.0"
ITEM_SCHEMA = "PHAxis-supplementary-table-data-item-1.0"
SOURCE_MAP_SCHEMA = "PHAxis-supplementary-table-data-source-map-1.0"
BUNDLE_DIRECTORY = "supplementary_tables_and_data"
BUNDLE_RECEIPT = "bundle_receipt.json"
FINAL_STATUS = "final_sealed_reviewer_facing_table_data"
PROVISIONAL_STATUS = "provisional_not_for_submission"

TABLE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "number": "S1",
        "stem": "Table_S01_DOME_checklist",
        "title": "DOME data, optimization, model, and evaluation checklist",
        "source_roles": (
            "source/train399_candidate",
            "source/train399_selection",
            "proposal/model_contract_proposal",
            "receipt/train399_evaluation",
            "receipt/root_exact283",
            "receipt/stageb",
            "receipt/fusion",
            "receipt/traits",
            "receipt/cohorts",
            "receipt/analysis",
            "receipt/profiles",
        ),
    },
    {
        "number": "S2",
        "stem": "Table_S02_HumanCurated443_manifest",
        "title": "HumanCurated443 task, family, split, annotation, and SHA-256 manifest",
        "source_roles": ("source/dataset_manifest", "source/split_manifest"),
    },
    {
        "number": "S3",
        "stem": "Table_S03_five_member_model_identities",
        "title": "Five-member model configuration and checkpoint identities",
        "source_roles": (
            "source/train399_candidate",
            "source/training_receipt_seed_2026082801",
            "source/training_receipt_seed_2026082802",
            "source/training_receipt_seed_2026082803",
            "source/training_receipt_seed_2026082804",
            "source/training_receipt_seed_2026082805",
        ),
    },
    {
        "number": "S4",
        "stem": "Table_S04_trait_dictionary_and_export_schema",
        "title": "Complete 32-trait dictionary and 82-column export schema",
        "source_roles": ("resource/trait_contract", "source/image_traits_schema"),
    },
    {
        "number": "S5",
        "stem": "Table_S05_observability_null_truth_table",
        "title": "Observability and null-semantics truth table",
        "source_roles": ("resource/trait_contract", "source/image_traits_schema"),
    },
    {
        "number": "S6",
        "stem": "Table_S06_QCdevelopment44_per_image",
        "title": "Per-image QC-development metrics and paired legacy comparison",
        "source_roles": (
            "resource/development_per_image",
            "resource/development_tolerance",
            "resource/development_threshold",
            "source/historical_oof_per_image",
        ),
    },
    {
        "number": "S7",
        "stem": "Table_S07_hair_identity_attachment_geometry",
        "title": "Hair identity, formal attachment, and conditional-geometry support",
        "source_roles": (
            "resource/assurance_metrics",
            "resource/assurance_pairs",
            "resource/assurance_support",
        ),
    },
    {
        "number": "S8",
        "stem": "Table_S08_root_continuity_and_trait_agreement",
        "title": "Per-image primary-root continuity and per-trait primary-root agreement",
        "source_roles": (
            "resource/assurance_metrics",
            "resource/assurance_pairs",
            "source/assurance_topology",
        ),
    },
    {
        "number": "S9",
        "stem": "Table_S09_complete_multitrait_atlas",
        "title": (
            "Complete 32-trait D15 atlas and block/day-stratified WT temperature "
            "secondary analysis"
        ),
        "source_roles": (
            "resource/multitrait_atlas",
            "source/clean_traits",
            "source/full_traits",
            "source/full_image_traits",
            "source/analysis_primary_table",
            "source/analysis_sensitivity_table",
            "source/wt_temperature_qc_flow",
            "source/wt_within_experiment_contrasts",
            "source/wt_within_day_meta_analysis",
            "receipt/analysis",
        ),
    },
    {
        "number": "S10",
        "stem": "Table_S10_reproducibility_benchmark_ledger",
        "title": "Reproducibility, benchmark, and implementation-choice ledger",
        "source_roles": (
            "resource/workflow_stages",
            "resource/runtime_summary",
            "resource/runtime_per_image",
            "source/baseline_runtime_per_image",
            "source/runtime_latency_comparison",
            "source/runtime_production_comparison",
            "source/benchmark_same_hardware",
            "source/benchmark_artifact_inventory",
        ),
    },
)

TABLE_NUMBERS = tuple(record["number"] for record in TABLE_SPECS)
TABLE_STEMS = tuple(record["stem"] for record in TABLE_SPECS)
TABLE_DIRECTORIES = tuple(f"S{index:02d}" for index in range(1, 11))


class SupplementaryTableError(RuntimeError):
    """The table/data bundle is incomplete, denominator-open, or tampered."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SupplementaryTableError(message)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _safe_file(path: str | Path, role: str) -> Path:
    raw = Path(path)
    _require(not raw.is_symlink(), f"{role}: symlink source is forbidden")
    result = raw.resolve()
    _require(result.is_file(), f"{role}: source file is missing: {result}")
    _require("blind" not in str(result).casefold(), f"{role}: blind-labelled path refused")
    return result


def _json(path: Path, role: str) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SupplementaryTableError(f"{role}: invalid JSON: {error}") from error
    _require(isinstance(payload, Mapping), f"{role}: JSON root must be an object")
    return dict(payload)


def _csv_rows(path: Path, role: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _require(reader.fieldnames is not None, f"{role}: CSV header is missing")
            fieldnames = [str(value) for value in reader.fieldnames]
            rows = [dict(row) for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise SupplementaryTableError(f"{role}: invalid CSV: {error}") from error
    return fieldnames, rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _copy(path: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _require(not destination.exists(), f"refusing to overwrite supplementary file: {destination}")
    shutil.copyfile(path, destination)


def _source_filename(index: int, path: Path) -> str:
    # Keep the reviewer bundle comfortably below the legacy Windows MAX_PATH
    # boundary even when the run root and descriptive item stem are long.  The
    # receipt, not the filename, is the authority for the semantic source role.
    suffix = path.suffix.lower() or ".bin"
    return f"s{index:02d}{suffix}"


def _source_record(role: str, path: Path, identities: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    identity = identities.get(role, {})
    field = identity.get("identity_field")
    value = identity.get("identity_sha256")
    _require(
        (field is None and value is None)
        or (isinstance(field, str) and bool(field) and _is_sha256(value)),
        f"{role}: source logical-identity declaration is malformed",
    )
    return {
        "role": role,
        "source_file_sha256": sha256_file(path),
        "identity_field": field,
        "identity_sha256": value,
    }


def _assert_identity_binding(role: str, path: Path, record: Mapping[str, Any]) -> None:
    field = record.get("identity_field")
    identity = record.get("identity_sha256")
    if field is None:
        _require(identity is None, f"{role}: identity value lacks a field")
        return
    payload = _json(path, role)
    _require(payload.get(field) == identity, f"{role}: declared logical identity differs")
    unsigned = deepcopy(payload)
    unsigned.pop(str(field), None)
    _require(sha256_json(unsigned) == identity, f"{role}: logical identity does not seal source receipt")


def _copy_declared_sources(
    item_root: Path,
    roles: Sequence[str],
    sources: Mapping[str, Path],
    identities: Mapping[str, Mapping[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for index, role in enumerate(roles, start=1):
        path = sources[role]
        record = _source_record(role, path, identities)
        _assert_identity_binding(role, path, record)
        relative = Path("src") / _source_filename(index, path)
        target = item_root / relative
        _copy(path, target)
        observed = sha256_file(target)
        _require(observed == record["source_file_sha256"], f"{role}: copied bytes changed")
        record["copied_path"] = relative.as_posix()
        records.append(record)
        hashes[relative.as_posix()] = observed
    return records, hashes


def _dome_rows(sources: Mapping[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for role in TABLE_SPECS[0]["source_roles"]:
        path = sources[role]
        payload = _json(path, role)
        identity_fields = sorted(
            key for key, value in payload.items() if key.endswith("identity_sha256") and _is_sha256(value)
        )
        rows.append(
            {
                "authority_role": role,
                "schema_version": payload.get("schema_version"),
                "status": payload.get("status") or payload.get("formal_release_status"),
                "file_sha256": sha256_file(path),
                "logical_identity_fields": ";".join(identity_fields),
                "blind_images_used": payload.get("blind_images_used", 0),
            }
        )
    return rows


def _validate_dataset_manifest(dataset: Path, split: Path) -> dict[str, Any]:
    dataset_fields, dataset_rows = _csv_rows(dataset, "dataset_manifest")
    split_fields, split_rows = _csv_rows(split, "split_manifest")
    required_dataset = {
        "task_id",
        "split",
        "family_key",
        "image_sha256",
        "raw_annotation_sha256",
        "canonical_annotation_relpath",
    }
    _require(required_dataset <= set(dataset_fields), "dataset manifest omits task/family/annotation/SHA fields")
    _require({"task_id", "split", "family_key"} <= set(split_fields), "split manifest header is incomplete")
    _require(len(dataset_rows) == len(split_rows) == 443, "HumanCurated443 manifests must contain exactly 443 rows")
    dataset_ids = [row["task_id"] for row in dataset_rows]
    split_ids = [row["task_id"] for row in split_rows]
    _require(len(set(dataset_ids)) == 443 and set(dataset_ids) == set(split_ids), "HumanCurated443 task identity closure failed")
    dataset_task_split_family = {
        row["task_id"]: (row["split"], row["family_key"])
        for row in dataset_rows
    }
    split_task_split_family = {
        row["task_id"]: (row["split"], row["family_key"])
        for row in split_rows
    }
    _require(
        dataset_task_split_family == split_task_split_family,
        "dataset/split manifests disagree on per-task split or family_key",
    )
    _require(
        all(
            _is_sha256(row["image_sha256"])
            and _is_sha256(row["raw_annotation_sha256"])
            and isinstance(row["canonical_annotation_relpath"], str)
            and bool(row["canonical_annotation_relpath"])
            for row in dataset_rows
        ),
        "HumanCurated443 manifest contains an invalid image/annotation SHA-256 or canonical path",
    )
    split_counts: dict[str, int] = {}
    for row in split_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    _require(split_counts == {"train": 399, "val": 44}, "HumanCurated443 split must be 399 train / 44 val")
    train_families = {row["family_key"] for row in split_rows if row["split"] == "train"}
    val_families = {row["family_key"] for row in split_rows if row["split"] == "val"}
    _require(not train_families.intersection(val_families), "HumanCurated443 family_key overlap is nonzero")
    dataset_train_families = {
        row["family_key"] for row in dataset_rows if row["split"] == "train"
    }
    dataset_val_families = {
        row["family_key"] for row in dataset_rows if row["split"] == "val"
    }
    _require(
        not dataset_train_families.intersection(dataset_val_families),
        "dataset manifest family_key overlap is nonzero",
    )
    return {
        "row_unit": "one canonical HumanCurated443 task",
        "expected_rows": 443,
        "observed_rows": 443,
        "train_rows": 399,
        "validation_rows": 44,
        "family_key_overlap": 0,
        "closure_status": "closed_exact443_family_isolated",
    }


def _candidate_members(candidate_path: Path, training_paths: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate = _json(candidate_path, "train399_candidate")
    identity_payload = candidate.get("identity_payload")
    _require(isinstance(identity_payload, Mapping), "candidate identity payload is missing")
    members = identity_payload.get("members")
    _require(isinstance(members, list) and len(members) == 5, "candidate must contain exactly five members")
    training_lock = identity_payload.get("training_lock")
    _require(isinstance(training_lock, Mapping), "candidate training lock is missing")
    config_sha256 = training_lock.get("config_sha256")
    _require(_is_sha256(config_sha256), "candidate shared training config SHA-256 is missing")
    receipts_by_seed: dict[int, tuple[Path, dict[str, Any]]] = {}
    for path in training_paths:
        receipt = _json(path, f"training receipt {path.name}")
        seed = receipt.get("seed")
        _require(isinstance(seed, int), f"{path.name}: training seed is invalid")
        _require(seed not in receipts_by_seed, f"duplicate training receipt seed {seed}")
        receipts_by_seed[seed] = (path, receipt)
    rows: list[dict[str, Any]] = []
    expected_seeds = [2026082801, 2026082802, 2026082803, 2026082804, 2026082805]
    _require([member.get("seed") for member in members] == expected_seeds, "five-member seed order changed")
    _require(set(receipts_by_seed) == set(expected_seeds), "five completion receipts do not match fixed seeds")
    for member in members:
        seed = int(member["seed"])
        path, receipt = receipts_by_seed[seed]
        _require(receipt.get("status") == "completed", f"seed {seed}: training receipt is not completed")
        _require(receipt.get("blind_images_used") == 0, f"seed {seed}: blind guard changed")
        checkpoint_sha = member.get("checkpoint_sha256")
        _require(_is_sha256(checkpoint_sha), f"seed {seed}: checkpoint SHA-256 missing")
        _require(
            _is_sha256(member.get("initialization_sha256"))
            and _is_sha256(member.get("model_state_sha256")),
            f"seed {seed}: initialization/model-state identity is incomplete",
        )
        _require(receipt.get("checkpoint_sha256") == checkpoint_sha, f"seed {seed}: checkpoint/receipt hash mismatch")
        receipt_sha = sha256_file(path)
        _require(member.get("training_receipt_sha256") == receipt_sha, f"seed {seed}: completion receipt hash mismatch")
        rows.append(
            {
                "seed": seed,
                "member_id": member.get("member_id"),
                "config_sha256": config_sha256,
                "initialization_sha256": member.get("initialization_sha256"),
                "model_state_sha256": member.get("model_state_sha256"),
                "checkpoint_sha256": checkpoint_sha,
                "final_epoch": member.get("epoch", receipt.get("epochs")),
                "global_step": member.get("global_step", receipt.get("global_steps")),
                "peak_allocated_mib": receipt.get("peak_allocated_mib"),
                "peak_reserved_mib": receipt.get("peak_reserved_mib"),
                "cuda_visible_devices": receipt.get("cuda_visible_devices"),
                "internal_device": receipt.get("internal_device"),
                "gpu_name": receipt.get("gpu_name"),
                "training_receipt_sha256": receipt_sha,
            }
        )
    _require(
        len({row["member_id"] for row in rows}) == 5
        and len({row["checkpoint_sha256"] for row in rows}) == 5
        and len({row["model_state_sha256"] for row in rows}) == 5
        and len({row["initialization_sha256"] for row in rows}) == 5,
        "five-member ensemble identities are not unique",
    )
    return rows, {
        "row_unit": "one fixed-seed train399 ensemble member",
        "expected_rows": 5,
        "observed_rows": len(rows),
        "fixed_seed_order": expected_seeds,
        "closure_status": "closed_exact_five_completed_members",
    }


def _trait_rows(contract_path: Path, schema_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = _json(contract_path, "trait_contract")
    schema = _json(schema_path, "image_traits_schema")
    records = []
    for family in ("primary_root_traits", "root_hair_traits"):
        values = contract.get(family)
        _require(isinstance(values, list), f"trait contract {family} is missing")
        for value in values:
            _require(isinstance(value, Mapping), f"trait contract {family} record is malformed")
            records.append(
                {
                    "trait_id": value.get("id"),
                    "field": value.get("field"),
                    "display_name_en": value.get("display_name_en"),
                    "display_name_cn": value.get("display_name_cn"),
                    "unit": value.get("unit"),
                    "type": value.get("type"),
                    "source": value.get("source"),
                    "family": family,
                    "censoring": value.get("censoring") or value.get("x-censoring"),
                }
            )
    _require(
        len(records) == 32
        and len({row["trait_id"] for row in records}) == 32
        and len({row["field"] for row in records}) == 32,
        "trait dictionary is not exact32 unique traits/fields",
    )
    required = schema.get("required")
    properties = schema.get("properties")
    _require(
        isinstance(required, list)
        and len(required) == 82
        and len(set(required)) == 82
        and isinstance(properties, Mapping)
        and list(properties) == required
        and schema.get("additionalProperties") is False,
        "image-trait export schema is not exact82 closed ordered properties",
    )
    return records, {
        "row_unit": "one canonical biological descriptor",
        "expected_trait_rows": 32,
        "observed_trait_rows": 32,
        "expected_export_schema_columns": 82,
        "observed_export_schema_columns": 82,
        "closure_status": "closed_exact32_traits_exact82_schema_columns",
    }


def _null_rows(contract_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = _json(contract_path, "trait_contract")
    semantics = contract.get("null_semantics")
    _require(isinstance(semantics, Mapping) and bool(semantics), "trait null-semantics contract is missing")
    rows = [{"state": key, "canonical_semantics": value} for key, value in semantics.items()]
    return rows, {
        "row_unit": "one canonical observability/null state",
        "expected_rows": len(rows),
        "observed_rows": len(rows),
        "closure_status": "closed_from_static_trait_contract",
    }


def _qc_denominator(per_image: Path, tolerance: Path) -> dict[str, Any]:
    _, image_rows = _csv_rows(per_image, "development_per_image")
    _, tolerance_rows = _csv_rows(tolerance, "development_tolerance")
    source_column = "source_unit" if image_rows and "source_unit" in image_rows[0] else "task_id"
    units = {row.get(source_column) for row in image_rows}
    comparators = {row.get("comparator") for row in image_rows}
    tolerances = {row.get("tolerance_um") for row in tolerance_rows}
    _require(
        len(units) == 44 and all(isinstance(value, str) and bool(value) for value in units),
        "Table S6 must bind exactly 44 named QC-development source images",
    )
    _require(
        len(comparators) == 2
        and all(isinstance(value, str) and bool(value) for value in comparators),
        "Table S6 must retain two named paired comparators",
    )
    observed_pairs = [
        (row.get(source_column), row.get("comparator")) for row in image_rows
    ]
    _require(
        len(image_rows) == len(observed_pairs) == len(set(observed_pairs)) == 88
        and all(
            {row.get("comparator") for row in image_rows if row.get(source_column) == unit}
            == comparators
            for unit in units
        ),
        "Table S6 must contain one row for each exact44 x paired-comparator slot",
    )
    _require(tolerances == {"5", "10", "20"}, "Table S6 tolerance grid must be 5/10/20 um")
    tolerance_pairs = [
        (row.get("comparator"), row.get("tolerance_um")) for row in tolerance_rows
    ]
    _require(
        len(tolerance_rows) == len(tolerance_pairs) == len(set(tolerance_pairs)) == 6
        and {value[0] for value in tolerance_pairs} == comparators,
        "Table S6 aggregate grid must contain each comparator at 5/10/20 um exactly once",
    )
    return {
        "row_unit": "one QC-development source image per paired comparator plus aggregate tolerance rows",
        "expected_source_images": 44,
        "observed_source_images": 44,
        "paired_comparators": 2,
        "tolerance_grid_um": [5, 10, 20],
        "per_image_rows": len(image_rows),
        "aggregate_tolerance_rows": len(tolerance_rows),
        "closure_status": "closed_exact44_paired_comparison",
    }


def _assurance_denominator(paths: Sequence[Path], role: str) -> dict[str, Any]:
    tables = []
    for path in paths:
        _fields, rows = _csv_rows(path, f"{role}/{path.name}")
        tables.append({"file": path.name, "rows": len(rows)})
    _require(all(record["rows"] > 0 for record in tables), f"{role}: an assurance block is empty")
    return {
        "row_unit": "source-image, matched-association, support, metric, or trait row as declared by each source table",
        "source_table_rows": tables,
        "uncertainty_resampling_unit": "source_image",
        "second_base_only_matching_allowed": False,
        "closure_status": "closed_source_preserving_assurance_tables",
    }


def _flatten_atlas(atlas_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    atlas = _json(atlas_path, "multitrait_atlas")
    descriptors = atlas.get("descriptors")
    cohort_order = atlas.get("cohort_order")
    condition_order = atlas.get("condition_order")
    effect_order = atlas.get("effect_order")
    _require(isinstance(descriptors, list) and len(descriptors) == 32, "Table S9 atlas descriptor count is not 32")
    _require(isinstance(cohort_order, list) and len(cohort_order) == 2, "Table S9 cohort order changed")
    _require(isinstance(condition_order, list) and len(condition_order) == 4, "Table S9 condition order changed")
    _require(isinstance(effect_order, list) and len(effect_order) == 3, "Table S9 effect order changed")
    expected_trait_ids = [f"R{index:02d}" for index in range(1, 20)] + [
        f"H{index:02d}" for index in range(1, 14)
    ]
    _require(
        all(isinstance(descriptor, Mapping) for descriptor in descriptors),
        "Table S9 descriptor record is malformed",
    )
    observed_trait_ids = [descriptor.get("trait_id") for descriptor in descriptors]
    _require(
        observed_trait_ids == expected_trait_ids,
        "Table S9 canonical R01--R19/H01--H13 order changed",
    )
    conditions: list[dict[str, Any]] = []
    effects: list[dict[str, Any]] = []
    for descriptor in descriptors:
        _require(isinstance(descriptor, Mapping), "Table S9 descriptor record is malformed")
        cohorts = descriptor.get("cohorts")
        _require(isinstance(cohorts, Mapping), "Table S9 descriptor cohort map is missing")
        for cohort in cohort_order:
            cohort_record = cohorts.get(cohort)
            _require(isinstance(cohort_record, Mapping), f"Table S9 cohort missing: {cohort}")
            condition_map = cohort_record.get("condition_summaries")
            effect_map = cohort_record.get("effects")
            _require(isinstance(condition_map, Mapping), "Table S9 condition-summary map is missing")
            _require(isinstance(effect_map, Mapping), "Table S9 effect-status map is missing")
            for condition in condition_order:
                record = condition_map.get(condition)
                _require(isinstance(record, Mapping), f"Table S9 condition slot missing: {condition}")
                conditions.append(
                    {
                        "trait_id": descriptor.get("trait_id"),
                        "field": descriptor.get("field"),
                        "measurement_family": descriptor.get("measurement_family"),
                        "unit": descriptor.get("unit"),
                        "cohort": cohort,
                        "condition": condition,
                        **dict(record),
                    }
                )
            for effect in effect_order:
                record = effect_map.get(effect)
                _require(isinstance(record, Mapping), f"Table S9 effect slot missing: {effect}")
                effects.append(
                    {
                        "trait_id": descriptor.get("trait_id"),
                        "field": descriptor.get("field"),
                        "measurement_family": descriptor.get("measurement_family"),
                        "unit": descriptor.get("unit"),
                        "cohort": cohort,
                        "contrast": effect,
                        **dict(record),
                    }
                )
        _require(
            all(
                isinstance(descriptor.get(field), str) and bool(descriptor.get(field))
                for field in ("trait_id", "field", "measurement_family", "unit")
            ),
            f"Table S9 descriptor metadata incomplete: {descriptor.get('trait_id')}",
        )
    _require(len(conditions) == atlas.get("condition_summary_slot_count") == 256, "Table S9 Block A must contain exactly 256 slots")
    _require(len(effects) == atlas.get("effect_slot_count") == 192, "Table S9 Block B must contain exactly 192 slots")
    for record in conditions:
        total = record.get("source_unit_total")
        non_null = record.get("non_null_source_unit_n")
        observability = record.get("observability_fraction")
        if isinstance(total, int) and total > 0 and isinstance(non_null, int) and isinstance(observability, (int, float)):
            _require(math.isclose(float(observability), non_null / total, rel_tol=0, abs_tol=1e-12), "Table S9 observability denominator drift")
        q25, q75, iqr = record.get("q25"), record.get("q75"), record.get("iqr")
        if all(isinstance(value, (int, float)) for value in (q25, q75, iqr)):
            _require(math.isclose(float(iqr), float(q75) - float(q25), rel_tol=0, abs_tol=1e-12), "Table S9 IQR arithmetic drift")
    return conditions, effects, {
        "row_unit": "descriptor x cohort x condition (A) or descriptor x cohort x contrast (B)",
        "descriptor_rows": 32,
        "cohorts": 2,
        "conditions": 4,
        "contrasts": 3,
        "block_A_expected_slots": 256,
        "block_A_observed_slots": len(conditions),
        "block_B_expected_slots": 192,
        "block_B_observed_slots": len(effects),
        "estimated_effect_slots": atlas.get("estimated_effect_slot_count"),
        "not_estimated_effect_slots": atlas.get("not_estimated_effect_slot_count"),
        "closure_status": "closed_exact256_condition_exact192_effect_slots",
    }


def _cohort_ledger(clean_path: Path, full_path: Path) -> list[dict[str, Any]]:
    clean_fields, clean_rows = _csv_rows(clean_path, "clean_traits")
    full_fields, full_rows = _csv_rows(full_path, "full_traits")
    _require(len(clean_rows) == 261 and len(full_rows) == 283, "Table S9 clean/full cohort row counts must be 261/283")
    id_field = "task_id" if "task_id" in full_fields else "source_unit"
    _require(id_field in clean_fields, "Table S9 cohort source-unit identity is missing")
    clean_ids = [row.get(id_field) for row in clean_rows]
    full_ids = [row.get(id_field) for row in full_rows]
    _require(
        len(set(clean_ids)) == 261
        and len(set(full_ids)) == 283
        and all(isinstance(value, str) and bool(value) for value in full_ids)
        and set(clean_ids) <= set(full_ids),
        "Table S9 clean/full cohort source-unit closure is invalid",
    )
    selected = [
        field
        for field in (
            id_field,
            "source_image_sha256",
            "formal_statistics_eligible",
            "measurement_tier",
            "hair_count",
            "hair_length_measurement_hair_count",
            "hair_length_measurement_fraction",
            "distal_window_1_4mm_eligible",
            "local_hair_count_1_4mm",
            "local_total_hair_length_um_per_root_mm_1_4mm",
            "visible_hair_attachment_span_um_descriptive_right_censored",
        )
        if field in set(clean_fields).union(full_fields)
    ]
    rows: list[dict[str, Any]] = []
    for cohort, values in (("primary_clean261", clean_rows), ("sensitivity_full283", full_rows)):
        for value in values:
            rows.append({"cohort": cohort, **{field: value.get(field) for field in selected}})
    return rows


def _boolean_cell(value: Any, *, role: str, field: str) -> bool:
    normalized = str(value).strip().casefold()
    _require(
        normalized in {"true", "false"},
        f"{role}: {field} is not a typed boolean",
    )
    return normalized == "true"


def _finite_cell(value: Any, *, role: str, field: str) -> float:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise SupplementaryTableError(
            f"{role}: {field} is not numeric"
        ) from error
    _require(math.isfinite(number), f"{role}: {field} is not finite")
    return number


def _blank_cell(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"", "nan", "na", "null"}


def _integer_cell(value: Any, *, role: str, field: str) -> int:
    number = _finite_cell(value, role=role, field=field)
    _require(number.is_integer(), f"{role}: {field} is not an integer")
    return int(number)


def _optional_integer_cell(value: Any, *, role: str, field: str) -> int | None:
    if _blank_cell(value):
        return None
    return _integer_cell(value, role=role, field=field)


def _h11_raw_companion_block(
    *,
    atlas_path: Path,
    primary_analysis_path: Path,
    sensitivity_analysis_path: Path,
    analysis_summary_path: Path,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    """Bind the exact six H11 raw-median rows to their source bytes.

    The block is intentionally a reviewer-facing projection, not a second
    inferential family.  It checks all 30 fixed clean/full analysis rows against
    the atlas, then emits only H11 (two cohorts by three effects).  The primary
    ratio-scale estimates and their multiplicity family remain untouched.
    """

    atlas = _json(atlas_path, "Table S9 H11 multitrait atlas")
    atlas_identity = atlas.get("atlas_identity_sha256")
    _require(_is_sha256(atlas_identity), "Table S9 H11 atlas identity is missing")
    unsigned_atlas = deepcopy(atlas)
    unsigned_atlas.pop("atlas_identity_sha256", None)
    _require(
        sha256_json(unsigned_atlas) == atlas_identity,
        "Table S9 H11 atlas identity seal mismatch",
    )
    descriptors = atlas.get("descriptors")
    _require(isinstance(descriptors, list), "Table S9 H11 atlas descriptors missing")
    descriptor_by_field = {
        str(record.get("field")): record
        for record in descriptors
        if isinstance(record, Mapping)
    }
    _require(
        set(PRIMARY_ENDPOINTS) <= set(descriptor_by_field),
        "Table S9 H11 atlas omits a prespecified endpoint",
    )

    summary = _json(analysis_summary_path, "Table S9 H11 analysis receipt")
    output_hashes = summary.get("output_table_sha256")
    _require(
        isinstance(output_hashes, Mapping)
        and output_hashes.get("primary_tests") == sha256_file(primary_analysis_path)
        and output_hashes.get("sensitivity_tests")
        == sha256_file(sensitivity_analysis_path),
        "Table S9 H11 analysis receipt does not bind the actual primary/sensitivity source SHA-256",
    )

    required_columns = {
        "cohort",
        "endpoint",
        "effect",
        "n",
        "estimate",
        "ci95_low",
        "ci95_high",
        "effect_scale",
        "raw_effect_estimate",
        "raw_effect_ci95_low",
        "raw_effect_ci95_high",
        "raw_effect_estimand",
        "raw_effect_interval_method",
        "raw_effect_bootstrap_replicates",
        "raw_effect_bootstrap_seed",
        "standardized_effect",
        "standardized_ci95_low",
        "standardized_ci95_high",
    }
    expected_cells = {
        (endpoint, effect) for endpoint in PRIMARY_ENDPOINTS for effect in EFFECT_NAME_TO_KEY
    }
    atlas_sha = sha256_file(atlas_path)
    receipt_sha = sha256_file(analysis_summary_path)
    output_rows: list[dict[str, Any]] = []
    for cohort, source_path in (
        ("primary_clean261", primary_analysis_path),
        ("sensitivity_full283", sensitivity_analysis_path),
    ):
        fields, rows = _csv_rows(source_path, f"Table S9 H11 {cohort} analysis")
        _require(
            required_columns <= set(fields),
            f"Table S9 H11 {cohort} analysis companion schema is incomplete",
        )
        selected = [
            row
            for row in rows
            if row.get("endpoint") in PRIMARY_ENDPOINTS
            and row.get("effect") in EFFECT_NAME_TO_KEY
        ]
        observed_cells = [(str(row["endpoint"]), str(row["effect"])) for row in selected]
        _require(
            len(observed_cells) == 15
            and len(set(observed_cells)) == 15
            and set(observed_cells) == expected_cells
            and {str(row.get("cohort")) for row in selected} == {cohort},
            f"Table S9 H11 {cohort} fixed effect family is not exact 15",
        )
        source_sha = sha256_file(source_path)
        by_cell = {
            (str(row["endpoint"]), str(row["effect"])): row for row in selected
        }
        for endpoint in PRIMARY_ENDPOINTS:
            descriptor = descriptor_by_field[endpoint]
            cohorts = descriptor.get("cohorts")
            _require(isinstance(cohorts, Mapping), f"Table S9 H11 {endpoint}: atlas cohorts missing")
            atlas_cohort = cohorts.get(cohort)
            _require(isinstance(atlas_cohort, Mapping), f"Table S9 H11 {cohort}/{endpoint}: atlas cohort missing")
            atlas_effects = atlas_cohort.get("effects")
            _require(isinstance(atlas_effects, Mapping), f"Table S9 H11 {cohort}/{endpoint}: atlas effects missing")
            for raw_effect, effect_key in EFFECT_NAME_TO_KEY.items():
                role = f"Table S9 H11 {cohort}/{endpoint}/{raw_effect}"
                raw = by_cell[(endpoint, raw_effect)]
                effect = atlas_effects.get(effect_key)
                _require(isinstance(effect, Mapping), f"{role}: atlas effect missing")
                numeric_bindings = (
                    ("estimate", "estimate"),
                    ("ci95_low", "ci95_low"),
                    ("ci95_high", "ci95_high"),
                    ("n", "endpoint_n"),
                    ("raw_effect_estimate", "raw_effect_estimate"),
                    ("raw_effect_ci95_low", "raw_effect_ci95_low"),
                    ("raw_effect_ci95_high", "raw_effect_ci95_high"),
                    ("standardized_effect", "standardized_effect"),
                    ("standardized_ci95_low", "standardized_ci95_low"),
                    ("standardized_ci95_high", "standardized_ci95_high"),
                )
                for raw_key, atlas_key in numeric_bindings:
                    raw_value = _finite_cell(raw.get(raw_key), role=role, field=raw_key)
                    atlas_value = _finite_cell(effect.get(atlas_key), role=role, field=atlas_key)
                    _require(
                        math.isclose(raw_value, atlas_value, rel_tol=0, abs_tol=1e-12),
                        f"{role}: raw analysis and atlas companion differ",
                    )
                for key in (
                    "effect_scale",
                    "raw_effect_estimand",
                    "raw_effect_interval_method",
                ):
                    _require(
                        str(raw.get(key)) == str(effect.get(key)),
                        f"{role}: raw analysis and atlas semantic label differ",
                    )
                replicates = _integer_cell(
                    raw.get("raw_effect_bootstrap_replicates"),
                    role=role,
                    field="raw_effect_bootstrap_replicates",
                )
                atlas_replicates = _integer_cell(
                    effect.get("raw_effect_bootstrap_replicates"),
                    role=role,
                    field="atlas raw_effect_bootstrap_replicates",
                )
                seed = _optional_integer_cell(
                    raw.get("raw_effect_bootstrap_seed"),
                    role=role,
                    field="raw_effect_bootstrap_seed",
                )
                atlas_seed = _optional_integer_cell(
                    effect.get("raw_effect_bootstrap_seed"),
                    role=role,
                    field="atlas raw_effect_bootstrap_seed",
                )
                _require(
                    replicates == atlas_replicates and seed == atlas_seed,
                    f"{role}: raw analysis and atlas bootstrap metadata differ",
                )
                if endpoint == H11_ENDPOINT:
                    expected_seed = raw_median_bootstrap_seed(
                        seed=H11_RAW_BOOTSTRAP_BASE_SEED,
                        field=H11_ENDPOINT,
                        component="continuous",
                    )
                    _require(
                        raw.get("raw_effect_estimand")
                        == RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
                        and raw.get("raw_effect_interval_method")
                        == RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
                        and replicates == H11_RAW_BOOTSTRAP_REPLICATES
                        and seed == expected_seed,
                        f"{role}: H11 median/bootstrap/5000/seed contract changed",
                    )
                    summaries = atlas_cohort.get("condition_summaries")
                    _require(isinstance(summaries, Mapping), f"{role}: H11 condition summaries missing")
                    medians = []
                    for condition in GROUP_ORDER:
                        summary_record = summaries.get(condition)
                        _require(isinstance(summary_record, Mapping), f"{role}: H11 condition summary missing")
                        medians.append(
                            _finite_cell(
                                summary_record.get("median"),
                                role=role,
                                field=f"{condition} median",
                            )
                        )
                    ev22, ev30, oe22, oe30 = medians
                    expected_raw = {
                        "OE_vs_EV": 0.5 * ((oe22 - ev22) + (oe30 - ev30)),
                        "30C_vs_22C": 0.5 * ((ev30 - ev22) + (oe30 - oe22)),
                        "interaction": (oe30 - oe22) - (ev30 - ev22),
                    }[effect_key]
                    _require(
                        math.isclose(
                            _finite_cell(
                                raw.get("raw_effect_estimate"),
                                role=role,
                                field="raw_effect_estimate",
                            ),
                            expected_raw,
                            rel_tol=0,
                            abs_tol=1e-12,
                        ),
                        f"{role}: raw effect is not the four-cell median contrast",
                    )
                else:
                    _require(
                        raw.get("raw_effect_estimand") == RAW_EFFECT_OLS_MEAN_CONTRAST
                        and raw.get("raw_effect_interval_method") == RAW_EFFECT_HC3_INTERVAL
                        and replicates == 0
                        and seed is None,
                        f"{role}: non-H11 mean/HC3/0/blank-seed contract changed",
                    )

                raw_triplet = [
                    _finite_cell(raw.get(key), role=role, field=key)
                    for key in (
                        "raw_effect_estimate",
                        "raw_effect_ci95_low",
                        "raw_effect_ci95_high",
                    )
                ]
                standardized_triplet = [
                    _finite_cell(raw.get(key), role=role, field=key)
                    for key in (
                        "standardized_effect",
                        "standardized_ci95_low",
                        "standardized_ci95_high",
                    )
                ]
                implied_scales: list[float] = []
                for raw_value, standardized_value in zip(
                    raw_triplet, standardized_triplet, strict=True
                ):
                    if abs(standardized_value) <= 1e-12:
                        _require(
                            abs(raw_value) <= 1e-12,
                            f"{role}: standardized zero is inconsistent",
                        )
                    else:
                        implied_scales.append(raw_value / standardized_value)
                _require(
                    implied_scales
                    and all(value > 0 and math.isfinite(value) for value in implied_scales)
                    and all(
                        math.isclose(
                            value,
                            implied_scales[0],
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                        for value in implied_scales[1:]
                    ),
                    f"{role}: standardized companion does not use one positive source-unit SD",
                )

                if endpoint == H11_ENDPOINT:
                    output_rows.append(
                        {
                            "typed_block": "h11_raw_median_companion",
                            "trait_id": descriptor.get("trait_id"),
                            "endpoint": endpoint,
                            "cohort": cohort,
                            "effect": raw_effect,
                            "effect_key": effect_key,
                            "estimate": raw.get("estimate"),
                            "ci95_low": raw.get("ci95_low"),
                            "ci95_high": raw.get("ci95_high"),
                            "effect_scale": raw.get("effect_scale"),
                            "raw_effect_estimate": raw.get("raw_effect_estimate"),
                            "raw_effect_ci95_low": raw.get("raw_effect_ci95_low"),
                            "raw_effect_ci95_high": raw.get("raw_effect_ci95_high"),
                            "raw_effect_estimand": raw.get("raw_effect_estimand"),
                            "raw_effect_interval_method": raw.get("raw_effect_interval_method"),
                            "raw_effect_bootstrap_replicates": replicates,
                            "raw_effect_bootstrap_seed": seed,
                            "standardized_effect": raw.get("standardized_effect"),
                            "standardized_ci95_low": raw.get("standardized_ci95_low"),
                            "standardized_ci95_high": raw.get("standardized_ci95_high"),
                            "source_analysis_role": (
                                "analysis_primary_table"
                                if cohort == "primary_clean261"
                                else "analysis_sensitivity_table"
                            ),
                            "source_analysis_table_sha256": source_sha,
                            "source_multitrait_atlas_sha256": atlas_sha,
                            "source_analysis_receipt_sha256": receipt_sha,
                        }
                    )
    _require(
        len(output_rows) == 6
        and len({(row["cohort"], row["effect_key"]) for row in output_rows}) == 6,
        "Table S9 H11 reviewer block is not exact two cohorts by three effects",
    )
    fields = list(output_rows[0])
    denominator = {
        "block_G_typed_block": "h11_raw_median_companion",
        "block_G_expected_rows": 6,
        "block_G_observed_rows": len(output_rows),
        "block_G_endpoint": H11_ENDPOINT,
        "block_G_cohorts": ["primary_clean261", "sensitivity_full283"],
        "block_G_effect_order": list(EFFECT_KEYS),
        "block_G_raw_effect_estimand": RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
        "block_G_raw_effect_interval_method": RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL,
        "block_G_raw_effect_bootstrap_replicates": H11_RAW_BOOTSTRAP_REPLICATES,
        "block_G_raw_effect_bootstrap_seed": raw_median_bootstrap_seed(
            seed=H11_RAW_BOOTSTRAP_BASE_SEED,
            field=H11_ENDPOINT,
            component="continuous",
        ),
        "block_G_source_multitrait_atlas_sha256": atlas_sha,
        "block_G_source_primary_analysis_sha256": sha256_file(primary_analysis_path),
        "block_G_source_sensitivity_analysis_sha256": sha256_file(
            sensitivity_analysis_path
        ),
        "block_G_source_analysis_receipt_sha256": receipt_sha,
        "block_G_fixed_15_primary_ratio_family_changed": False,
        "block_G_closure_status": "closed_exact6_H11_raw_median_companions",
    }
    return fields, output_rows, denominator


def _wt_secondary_blocks(
    *,
    analysis_summary_path: Path,
    primary_analysis_path: Path,
    sensitivity_analysis_path: Path,
    gate_flow_path: Path,
    contrasts_path: Path,
    meta_path: Path,
) -> tuple[dict[str, tuple[list[str], list[dict[str, Any]]]], dict[str, Any]]:
    """Validate and type the separate WT secondary family for Table S9.

    The WT rows never enter Blocks A/B or the fixed D15 15-effect family.  The
    analysis receipt is the authority for all three source-table hashes and
    row-count/claim guards; the CSV checks below additionally enforce the
    experiment, developmental-day, and cohort boundaries at row level.
    """

    analysis = _json(analysis_summary_path, "Table S9 analysis receipt")
    _require(
        analysis.get("schema_version")
        == "PHAxis-exploratory-biological-analysis-1.0",
        "Table S9 WT analysis receipt schema changed",
    )
    _require(
        analysis.get("blind_images_used") == 0
        and analysis.get("root_cap_region_statistics_included") is False,
        "Table S9 WT analysis receipt violates a red-line guard",
    )
    _require(
        analysis.get("D15_fixed_effect_rows") == 15
        and analysis.get("D15_fixed_effect_family_changed_by_WT_secondary")
        is False,
        "Table S9 WT secondary analysis changed the D15 15-effect family",
    )
    for field in (
        "wt_secondary_cross_day_pooling_performed",
        "wt_secondary_unknown_day_meta_analysis_performed",
        "wt_secondary_clean_full_pooling_performed",
    ):
        _require(
            analysis.get(field) is False,
            f"Table S9 WT analysis boundary changed: {field}",
        )
    claim_status = analysis.get("wt_secondary_claim_status")
    _require(
        isinstance(claim_status, str)
        and "secondary exploratory" in claim_status.casefold(),
        "Table S9 WT secondary claim status is missing",
    )
    wt_contract = analysis.get("wt_secondary_analysis")
    _require(
        isinstance(wt_contract, Mapping)
        and wt_contract.get("schema_version")
        == "PHAxis-WT-temperature-secondary-1.0"
        and wt_contract.get("status")
        == "materialized_as_separate_secondary_family"
        and wt_contract.get("cross_day_pooling_performed") is False
        and wt_contract.get("unknown_day_meta_analysis_performed") is False
        and wt_contract.get("clean_full_pooling_performed") is False
        and wt_contract.get("D15_fixed_effect_family_changed") is False,
        "Table S9 WT secondary typed contract changed",
    )
    within_experiment_multiplicity = wt_contract.get(
        "within_experiment_multiplicity"
    )
    within_day_meta_multiplicity = wt_contract.get("within_day_meta_multiplicity")
    _require(
        isinstance(within_experiment_multiplicity, str)
        and "Benjamini-Hochberg" in within_experiment_multiplicity
        and isinstance(within_day_meta_multiplicity, str)
        and "Benjamini-Hochberg" in within_day_meta_multiplicity,
        "Table S9 WT multiplicity contract is missing",
    )

    table_hashes = analysis.get("output_table_sha256")
    _require(
        isinstance(table_hashes, Mapping),
        "Table S9 WT analysis receipt omits output-table hashes",
    )
    source_by_key = {
        "wt_temperature_qc_flow": gate_flow_path,
        "wt_within_experiment_contrasts": contrasts_path,
        "wt_within_day_meta_analysis": meta_path,
    }
    _require(
        all(
            table_hashes.get(key) == sha256_file(path)
            for key, path in source_by_key.items()
        ),
        "Table S9 WT source table differs from the analysis receipt",
    )

    primary_fields, primary_rows = _csv_rows(
        primary_analysis_path, "Table S9 D15 primary analysis"
    )
    sensitivity_fields, sensitivity_rows = _csv_rows(
        sensitivity_analysis_path, "Table S9 D15 sensitivity analysis"
    )
    _require(
        {"endpoint", "effect"} <= set(primary_fields)
        and {"endpoint", "effect"} <= set(sensitivity_fields)
        and len(primary_rows) == len(sensitivity_rows) == 15,
        "Table S9 D15 primary/sensitivity family is not exact 15 + 15",
    )

    gate_fields, gate_rows = _csv_rows(gate_flow_path, "Table S9 WT gate flow")
    contrast_fields, contrast_rows = _csv_rows(
        contrasts_path, "Table S9 WT experiment contrasts"
    )
    meta_fields, meta_rows = _csv_rows(meta_path, "Table S9 WT same-day meta-analysis")
    try:
        strict_evidence = validate_wt_secondary_evidence(
            contrasts=contrast_rows,
            meta=meta_rows,
            flow=gate_rows,
        )
        strict_binding = validate_wt_secondary_analysis_binding(
            analysis_summary=analysis,
            evidence_summary=strict_evidence,
            table_sha256={
                role: sha256_file(path) for role, path in source_by_key.items()
            },
        )
    except ValueError as error:
        raise SupplementaryTableError(
            f"Table S9 WT strict evidence validation failed: {error}"
        ) from error
    required_common = {
        "cohort",
        "cohort_role",
        "endpoint",
        "endpoint_label",
        "developmental_day",
    }
    _require(
        required_common
        | {
            "experiment_key",
            "analysis_status",
            "meta_eligible",
            "meta_exclusion_reason",
            "estimate_30C_over_22C",
            "ci95_low",
            "ci95_high",
            "p_value_model",
            "p_value_model_BH_FDR",
            "reject_model_BH_FDR_0p05",
            "multiplicity_family",
            "inference_status",
        }
        <= set(contrast_fields),
        "Table S9 WT experiment-contrast schema is incomplete",
    )
    _require(
        required_common
        | {
            "k_eligible_experiments",
            "analysis_status",
            "not_estimable_reason",
            "estimate_30C_over_22C",
            "ci95_low",
            "ci95_high",
            "p_value_hartung_knapp",
            "p_value_hartung_knapp_BH_FDR",
            "reject_hartung_knapp_BH_FDR_0p05",
            "multiplicity_family",
            "tau2_reml_log_scale",
            "Q",
            "I2",
            "cross_day_pooling_performed",
            "unknown_day_contrasts_included",
            "inference_status",
        }
        <= set(meta_fields),
        "Table S9 WT same-day meta-analysis schema is incomplete",
    )
    _require(
        required_common
        | {
            "experiment_key",
            "base_gate_pass",
            "endpoint_gate_pass",
            "model_status",
            "phenotype_outlier_filter_applied",
        }
        <= set(gate_fields),
        "Table S9 WT inventory/gate-flow schema is incomplete",
    )

    cohorts = {
        "primary_clean261": "primary_SHA_disjoint",
        "sensitivity_full283": "overlap_contaminated_sensitivity",
    }
    endpoints = {
        "local_hair_count_1_4mm",
        "local_median_hair_length_um_1_4mm",
        "first_hair_ge40um_distance_from_distal_point_um",
        "median_root_width_um",
        "visible_root_axis_length_um",
    }
    for role, rows in (
        ("Table S9 WT gate flow", gate_rows),
        ("Table S9 WT experiment contrasts", contrast_rows),
        ("Table S9 WT same-day meta-analysis", meta_rows),
    ):
        _require(
            all(
                row.get("cohort") in cohorts
                and row.get("cohort_role") == cohorts[row["cohort"]]
                and row.get("endpoint") in endpoints
                for row in rows
            ),
            f"{role}: clean/full cohort or five-endpoint boundary changed",
        )

    contrast_keys = [
        (row.get("cohort"), row.get("experiment_key"), row.get("endpoint"))
        for row in contrast_rows
    ]
    gate_keys = [
        (row.get("cohort"), row.get("experiment_key"), row.get("endpoint"))
        for row in gate_rows
    ]
    _require(
        len(contrast_keys) == len(set(contrast_keys))
        and set(contrast_keys) == set(gate_keys)
        and len(gate_keys) == len(set(gate_keys)),
        "Table S9 WT experiment inventory/contrast identity closure failed",
    )
    _require(
        all(row.get("experiment_key") != "D15_8d" for row in contrast_rows),
        "Table S9 WT secondary table contains the D15 factorial experiment",
    )
    for row in contrast_rows:
        status = row.get("analysis_status")
        _require(
            status in {"estimated", "not_estimable"},
            "Table S9 WT experiment contrast has an invalid analysis status",
        )
        day_unknown = _blank_cell(row.get("developmental_day"))
        meta_eligible = _boolean_cell(
            row.get("meta_eligible"),
            role="Table S9 WT experiment contrasts",
            field="meta_eligible",
        )
        if day_unknown:
            _require(
                not meta_eligible,
                "Table S9 unknown-day WT contrast entered meta-analysis",
            )
            if status == "estimated":
                _require(
                    row.get("meta_exclusion_reason") == "unknown_developmental_day",
                    "Table S9 estimated unknown-day contrast lacks its exclusion reason",
                )
        else:
            day = _finite_cell(
                row.get("developmental_day"),
                role="Table S9 WT experiment contrasts",
                field="developmental_day",
            )
            _require(
                day > 0 and math.isclose(day, round(day)),
                "Table S9 WT developmental day is not a positive integer",
            )
        _require(
            row.get("multiplicity_family")
            == (
                "within_cohort_all_estimated_WT_experiment_by_endpoint_"
                "contrasts_including_unknown_day"
            ),
            "Table S9 WT experiment-contrast multiplicity family changed",
        )

    meta_keys = [
        (row.get("cohort"), row.get("developmental_day"), row.get("endpoint"))
        for row in meta_rows
    ]
    _require(
        len(meta_keys) == len(set(meta_keys)),
        "Table S9 WT same-day meta-analysis contains a duplicate slot",
    )
    null_when_not_estimable = (
        "log_effect_30C_over_22C",
        "log_effect_standard_error_hartung_knapp",
        "estimate_30C_over_22C",
        "ci95_low",
        "ci95_high",
        "p_value_hartung_knapp",
        "p_value_hartung_knapp_BH_FDR",
        "tau2_reml_log_scale",
        "Q",
        "Q_df",
        "Q_p_value",
        "I2",
        "I2_percent",
        "hartung_knapp_scale",
    )
    for row in meta_rows:
        day = _finite_cell(
            row.get("developmental_day"),
            role="Table S9 WT same-day meta-analysis",
            field="developmental_day",
        )
        _require(
            day > 0 and math.isclose(day, round(day)),
            "Table S9 WT same-day synthesis has an invalid developmental day",
        )
        _require(
            not _boolean_cell(
                row.get("cross_day_pooling_performed"),
                role="Table S9 WT same-day meta-analysis",
                field="cross_day_pooling_performed",
            )
            and not _boolean_cell(
                row.get("unknown_day_contrasts_included"),
                role="Table S9 WT same-day meta-analysis",
                field="unknown_day_contrasts_included",
            ),
            "Table S9 WT same-day synthesis crossed a day boundary",
        )
        _require(
            row.get("multiplicity_family")
            == (
                "within_cohort_all_estimated_WT_developmental_day_by_endpoint_"
                "meta_analyses"
            ),
            "Table S9 WT same-day multiplicity family changed",
        )
        k_value = _finite_cell(
            row.get("k_eligible_experiments"),
            role="Table S9 WT same-day meta-analysis",
            field="k_eligible_experiments",
        )
        _require(
            k_value >= 0 and math.isclose(k_value, round(k_value)),
            "Table S9 WT same-day eligible-experiment count is not an integer",
        )
        k = int(round(k_value))
        status = row.get("analysis_status")
        _require(
            status in {"estimated", "not_estimable"},
            "Table S9 WT same-day meta-analysis has an invalid status",
        )
        if status == "estimated":
            _require(
                k >= 3,
                "Table S9 WT same-day estimate used fewer than three experiments",
            )
            estimate = _finite_cell(
                row.get("estimate_30C_over_22C"),
                role="Table S9 WT same-day meta-analysis",
                field="estimate_30C_over_22C",
            )
            low = _finite_cell(
                row.get("ci95_low"),
                role="Table S9 WT same-day meta-analysis",
                field="ci95_low",
            )
            high = _finite_cell(
                row.get("ci95_high"),
                role="Table S9 WT same-day meta-analysis",
                field="ci95_high",
            )
            _require(
                0 < low <= estimate <= high,
                "Table S9 WT same-day ratio/interval is invalid",
            )
        else:
            _require(
                k < 3
                or str(row.get("not_estimable_reason", "")).startswith(
                    "meta_model_failure:"
                ),
                "Table S9 WT not-estimable status lacks a replication/model reason",
            )
            _require(
                all(_blank_cell(row.get(field)) for field in null_when_not_estimable),
                "Table S9 WT not-estimable row contains a pooled estimate/statistic",
            )
            _require(
                not _boolean_cell(
                    row.get("reject_hartung_knapp_BH_FDR_0p05"),
                    role="Table S9 WT same-day meta-analysis",
                    field="reject_hartung_knapp_BH_FDR_0p05",
                ),
                "Table S9 WT not-estimable row contains a positive BH decision",
            )

    for row in gate_rows:
        _require(
            not _boolean_cell(
                row.get("phenotype_outlier_filter_applied"),
                role="Table S9 WT gate flow",
                field="phenotype_outlier_filter_applied",
            ),
            "Table S9 WT gate flow applied a phenotype outlier filter",
        )

    observed_counts = {
        "wt_secondary_within_experiment_rows": len(contrast_rows),
        "wt_secondary_estimable_within_experiment_rows": sum(
            row.get("analysis_status") == "estimated" for row in contrast_rows
        ),
        "wt_secondary_unknown_day_contrast_rows": sum(
            _blank_cell(row.get("developmental_day")) for row in contrast_rows
        ),
        "wt_secondary_within_day_meta_rows": len(meta_rows),
        "wt_secondary_estimable_within_day_meta_rows": sum(
            row.get("analysis_status") == "estimated" for row in meta_rows
        ),
        "wt_secondary_typed_not_estimable_meta_rows": sum(
            row.get("analysis_status") == "not_estimable" for row in meta_rows
        ),
    }
    _require(
        all(analysis.get(field) == value for field, value in observed_counts.items()),
        "Table S9 WT row counts differ from the analysis receipt",
    )
    _require(
        all(
            strict_evidence.get(evidence_field) == value
            for evidence_field, value in (
                ("within_experiment_rows", len(contrast_rows)),
                (
                    "estimated_within_experiment_rows",
                    observed_counts["wt_secondary_estimable_within_experiment_rows"],
                ),
                ("within_day_meta_rows", len(meta_rows)),
                (
                    "estimated_within_day_meta_rows",
                    observed_counts["wt_secondary_estimable_within_day_meta_rows"],
                ),
                (
                    "typed_not_estimable_meta_rows",
                    observed_counts["wt_secondary_typed_not_estimable_meta_rows"],
                ),
            )
        ),
        "Table S9 WT strict evidence counts differ from typed blocks",
    )
    _require(
        strict_binding.get("schema_version")
        == "PHAxis-WT-temperature-secondary-1.0"
        and strict_binding.get("D15_fixed_effect_family_changed") is False
        and strict_binding.get("cross_day_pooling_performed") is False
        and strict_binding.get("unknown_day_meta_analysis_performed") is False
        and strict_binding.get("clean_full_pooling_performed") is False,
        "Table S9 WT strict analysis binding changed",
    )

    block_rows: dict[str, tuple[list[str], list[dict[str, Any]]]] = {}
    for block_name, typed_block, fields, rows in (
        ("D", "wt_gate_flow", gate_fields, gate_rows),
        ("E", "wt_experiment_contrasts", contrast_fields, contrast_rows),
        ("F", "wt_same_day_meta", meta_fields, meta_rows),
    ):
        typed_rows = [{"typed_block": typed_block, **row} for row in rows]
        block_rows[block_name] = (["typed_block", *fields], typed_rows)

    denominator = {
        "D15_fixed_effect_rows": 15,
        "D15_fixed_effect_family_changed_by_WT_secondary": False,
        "block_D_typed_block": "wt_gate_flow",
        "block_D_observed_rows": len(gate_rows),
        "block_E_typed_block": "wt_experiment_contrasts",
        "block_E_observed_rows": len(contrast_rows),
        "block_F_typed_block": "wt_same_day_meta",
        "block_F_observed_rows": len(meta_rows),
        **observed_counts,
        "wt_secondary_cross_day_pooling_performed": False,
        "wt_secondary_unknown_day_meta_analysis_performed": False,
        "wt_secondary_clean_full_pooling_performed": False,
        "wt_secondary_claim_status": claim_status,
        "wt_secondary_schema_version": wt_contract["schema_version"],
        "wt_within_experiment_multiplicity": within_experiment_multiplicity,
        "wt_within_day_meta_multiplicity": within_day_meta_multiplicity,
        "wt_secondary_closure_status": (
            "closed_separate_WT_experiment_and_same_day_secondary_family"
        ),
    }
    return block_rows, denominator


def _runtime_denominator(
    runtime_summary: Path,
    runtime_per_image: Path,
    baseline_per_image: Path,
    same_hardware_receipt: Path,
    inventory: Path,
) -> dict[str, Any]:
    current_fields, current = _csv_rows(runtime_per_image, "runtime_per_image")
    baseline_fields, baseline = _csv_rows(
        baseline_per_image, "baseline_runtime_per_image"
    )
    inventory_fields, inventory_rows = _csv_rows(
        inventory, "benchmark_artifact_inventory"
    )
    _require(len(current) == len(baseline) == 283, "Table S10 sequential traces must both contain exact283 rows")
    _require(
        "source_unit" in current_fields and "source_unit" in baseline_fields,
        "Table S10 sequential traces omit source_unit",
    )
    current_order = [str(row["source_unit"]) for row in current]
    baseline_order = [str(row["source_unit"]) for row in baseline]
    _require(
        current_order == baseline_order and len(set(current_order)) == 283,
        "Table S10 sequential traces do not share one ordered unique exact283 source set",
    )
    ordered_identity = sha256_json(current_order)
    runtime = _json(runtime_summary, "runtime_summary")
    same_hardware = _json(same_hardware_receipt, "benchmark_same_hardware")
    required_same_hardware = {
        "status": "passed",
        "images": 283,
        "measurement_scope": "raw_image_to_final_traits_and_profiles_direct",
        "same_ordered_exact283_sources": True,
        "same_hardware_uuid_and_driver": True,
        "same_io_and_full_workflow_scope": True,
        "fresh_no_cache": True,
        "historical_98_47_min_component_receipt_used": False,
        "forward_only_runtime_used": False,
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }
    _require(
        all(same_hardware.get(field) == value for field, value in required_same_hardware.items()),
        "Table S10 same-hardware benchmark Gate is not passed exact283/fresh/full-workflow",
    )
    _require(
        same_hardware.get("source_unit_ordered_set_identity_sha256")
        == ordered_identity
        and runtime.get("source_unit_ordered_set_identity_sha256")
        == ordered_identity,
        "Table S10 ordered source-set identity differs across traces and receipts",
    )
    runs = same_hardware.get("runs")
    expected_run_roles = [
        "phaxis_production",
        "phaxis_sequential",
        "frozen_v1_production",
        "frozen_v1_sequential",
    ]
    _require(
        isinstance(runs, list)
        and len(runs) == 4
        and [run.get("role") if isinstance(run, Mapping) else None for run in runs]
        == expected_run_roles
        and all(
            isinstance(run, Mapping)
            and run.get("source_unit_ordered_set_identity_sha256")
            == ordered_identity
            and run.get("fresh_direct_run") is True
            and run.get("resume_or_cache_used") is False
            and run.get("full_workflow_io_included") is True
            for run in runs
        ),
        "Table S10 same-hardware run ledger is not four fresh identity-matched runs",
    )
    receipt_identity = same_hardware.get("receipt_identity_sha256")
    _require(_is_sha256(receipt_identity), "Table S10 same-hardware receipt identity is invalid")
    unsigned_same_hardware = deepcopy(same_hardware)
    unsigned_same_hardware.pop("receipt_identity_sha256", None)
    _require(
        sha256_json(unsigned_same_hardware) == receipt_identity,
        "Table S10 same-hardware receipt identity seal mismatch",
    )
    _require(
        "artifact_role" in inventory_fields,
        "Table S10 benchmark inventory omits artifact_role",
    )
    role_counts: dict[str, int] = {}
    for row in inventory_rows:
        role = str(row.get("artifact_role", ""))
        role_counts[role] = role_counts.get(role, 0) + 1
    exact_roles = (
        "same_hardware_receipt",
        "phaxis_production_summary",
        "v1_production_summary",
        "phaxis_sequential_summary",
        "v1_sequential_summary",
        "production_comparison_receipt",
        "sequential_comparison_receipt",
    )
    _require(
        all(role_counts.get(role) == 1 for role in exact_roles)
        and role_counts.get("per_image_latency_csv") == 2
        and role_counts.get("gpu_telemetry") == 4
        and role_counts.get("hardware_preflight") == 4
        and set(role_counts)
        == {*exact_roles, "per_image_latency_csv", "gpu_telemetry", "hardware_preflight"},
        "Table S10 benchmark inventory role closure differs from the formal stage",
    )
    return {
        "row_unit": "one source image for sequential traces; one explicit artifact for benchmark inventory",
        "phaxis_sequential_rows": 283,
        "frozen_v1_sequential_rows": 283,
        "benchmark_inventory_rows": len(inventory_rows),
        "source_unit_ordered_set_identity_sha256": ordered_identity,
        "same_hardware_receipt_identity_sha256": receipt_identity,
        "benchmark_inventory_role_counts": dict(sorted(role_counts.items())),
        "batch_wall_divided_into_pseudo_per_image_rows": False,
        "closure_status": "closed_exact283_same_io_scope_traces",
    }


def _derived_files(
    spec: Mapping[str, Any],
    item_root: Path,
    sources: Mapping[str, Path],
) -> tuple[dict[str, str], dict[str, Any]]:
    number = spec["number"]
    hashes: dict[str, str] = {}
    if number == "S1":
        rows = _dome_rows(sources)
        path = item_root / "DOME_checklist.csv"
        _write_csv(path, tuple(rows[0]), rows)
        denominator = {
            "row_unit": "one named publication authority",
            "expected_rows": len(spec["source_roles"]),
            "observed_rows": len(rows),
            "closure_status": "closed_all_named_authorities",
        }
    elif number == "S2":
        denominator = _validate_dataset_manifest(sources["source/dataset_manifest"], sources["source/split_manifest"])
        path = item_root / "dataset_manifest.csv"
        _copy(sources["source/dataset_manifest"], path)
    elif number == "S3":
        training = [sources[f"source/training_receipt_seed_{seed}"] for seed in range(2026082801, 2026082806)]
        rows, denominator = _candidate_members(sources["source/train399_candidate"], training)
        path = item_root / "five_member_identities.csv"
        _write_csv(path, tuple(rows[0]), rows)
    elif number == "S4":
        rows, denominator = _trait_rows(sources["resource/trait_contract"], sources["source/image_traits_schema"])
        path = item_root / "trait_dictionary.csv"
        _write_csv(path, tuple(rows[0]), rows)
        schema_copy = item_root / "image_traits.schema.json"
        _copy(sources["source/image_traits_schema"], schema_copy)
        hashes[schema_copy.relative_to(item_root).as_posix()] = sha256_file(schema_copy)
    elif number == "S5":
        rows, denominator = _null_rows(sources["resource/trait_contract"])
        path = item_root / "observability_null_truth_table.csv"
        _write_csv(path, tuple(rows[0]), rows)
    elif number == "S6":
        denominator = _qc_denominator(sources["resource/development_per_image"], sources["resource/development_tolerance"])
        path = item_root / "per_image_paired_metrics.csv"
        _copy(sources["resource/development_per_image"], path)
    elif number == "S7":
        paths = [sources[role] for role in spec["source_roles"]]
        denominator = _assurance_denominator(paths, "Table S7")
        path = item_root / "hair_association_rows.csv"
        _copy(sources["resource/assurance_pairs"], path)
    elif number == "S8":
        paths = [sources[role] for role in spec["source_roles"]]
        denominator = _assurance_denominator(paths, "Table S8")
        path = item_root / "root_continuity_and_trait_pairs.csv"
        _copy(sources["resource/assurance_pairs"], path)
    elif number == "S9":
        conditions, effects, denominator = _flatten_atlas(sources["resource/multitrait_atlas"])
        path = item_root / "block_A_condition_atlas.csv"
        fields_a = list(dict.fromkeys(key for row in conditions for key in row))
        _write_csv(path, fields_a, conditions)
        block_b = item_root / "block_B_effect_status_ledger.csv"
        fields_b = list(dict.fromkeys(key for row in effects for key in row))
        _write_csv(block_b, fields_b, effects)
        hashes[block_b.relative_to(item_root).as_posix()] = sha256_file(block_b)
        cohort_rows = _cohort_ledger(sources["source/clean_traits"], sources["source/full_traits"])
        block_c = item_root / "block_C_cohort_provenance.csv"
        _write_csv(block_c, tuple(cohort_rows[0]), cohort_rows)
        hashes[block_c.relative_to(item_root).as_posix()] = sha256_file(block_c)
        denominator["block_C_observed_rows"] = len(cohort_rows)
        denominator["block_C_expected_rows"] = 261 + 283
        wt_blocks, wt_denominator = _wt_secondary_blocks(
            analysis_summary_path=sources["receipt/analysis"],
            primary_analysis_path=sources["source/analysis_primary_table"],
            sensitivity_analysis_path=sources["source/analysis_sensitivity_table"],
            gate_flow_path=sources["source/wt_temperature_qc_flow"],
            contrasts_path=sources["source/wt_within_experiment_contrasts"],
            meta_path=sources["source/wt_within_day_meta_analysis"],
        )
        for block_name, filename in (
            ("D", "block_D_WT_inventory_and_gate_flow.csv"),
            ("E", "block_E_WT_within_experiment_temperature_contrasts.csv"),
            ("F", "block_F_WT_within_day_REML_Hartung_Knapp.csv"),
        ):
            fields, rows = wt_blocks[block_name]
            block_path = item_root / filename
            _write_csv(block_path, fields, rows)
            hashes[block_path.relative_to(item_root).as_posix()] = sha256_file(
                block_path
            )
        denominator.update(wt_denominator)
        h11_fields, h11_rows, h11_denominator = _h11_raw_companion_block(
            atlas_path=sources["resource/multitrait_atlas"],
            primary_analysis_path=sources["source/analysis_primary_table"],
            sensitivity_analysis_path=sources["source/analysis_sensitivity_table"],
            analysis_summary_path=sources["receipt/analysis"],
        )
        block_g = item_root / "block_G_H11_raw_median_companion.csv"
        _write_csv(block_g, h11_fields, h11_rows)
        hashes[block_g.relative_to(item_root).as_posix()] = sha256_file(block_g)
        denominator.update(h11_denominator)
    elif number == "S10":
        denominator = _runtime_denominator(
            sources["resource/runtime_summary"],
            sources["resource/runtime_per_image"],
            sources["source/baseline_runtime_per_image"],
            sources["source/benchmark_same_hardware"],
            sources["source/benchmark_artifact_inventory"],
        )
        path = item_root / "workflow_stages.csv"
        _copy(sources["resource/workflow_stages"], path)
    else:  # pragma: no cover - protected by the static exact-ten contract
        raise SupplementaryTableError(f"unknown supplementary item: {number}")
    hashes[path.relative_to(item_root).as_posix()] = sha256_file(path)
    return hashes, denominator


def _normalize_sources(
    source_paths: Mapping[str, str | Path],
) -> dict[str, Path]:
    required = {role for spec in TABLE_SPECS for role in spec["source_roles"]}
    _require(set(source_paths) == required, "supplementary source-role set is not exact")
    return {role: _safe_file(path, role) for role, path in source_paths.items()}


def materialize_supplementary_table_data_bundle(
    *,
    output: str | Path,
    status: str,
    source_paths: Mapping[str, str | Path],
    source_identities: Mapping[str, Mapping[str, str]] | None,
    figure_input_manifest_sha256: str,
    figure_input_assembly_identity_sha256: str,
    model_contract_proposal_identity_sha256: str,
) -> dict[str, Any]:
    """Atomically create the exact reviewer-facing S1--S10 bundle."""

    _require(status in {FINAL_STATUS, PROVISIONAL_STATUS}, "supplementary bundle status is invalid")
    for label, value in (
        ("figure-input manifest", figure_input_manifest_sha256),
        ("figure-input assembly", figure_input_assembly_identity_sha256),
        ("model-contract proposal", model_contract_proposal_identity_sha256),
    ):
        _require(_is_sha256(value), f"{label} identity is invalid")
    destination = Path(output).resolve()
    _require(not destination.exists(), f"refusing to overwrite supplementary bundle: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sources = _normalize_sources(source_paths)
    identities = dict(source_identities or {})
    _require(set(identities) <= set(sources), "supplementary source identity names an unknown role")
    staging = Path(tempfile.mkdtemp(prefix=".supp-tables-", dir=destination.parent)).resolve()
    try:
        item_records: dict[str, dict[str, Any]] = {}
        bundle_hashes: dict[str, str] = {}
        for spec, item_directory in zip(
            TABLE_SPECS, TABLE_DIRECTORIES, strict=True
        ):
            # Reviewer-facing semantic stems remain stable in receipts and
            # evidence maps, while short physical directories keep the exact
            # bundle usable under legacy Windows MAX_PATH behavior.
            item_root = staging / item_directory
            item_root.mkdir()
            source_records, source_hashes = _copy_declared_sources(
                item_root, spec["source_roles"], sources, identities
            )
            derived_hashes, denominator = _derived_files(spec, item_root, sources)
            file_hashes = {**source_hashes, **derived_hashes}
            item_receipt: dict[str, Any] = {
                "schema_version": ITEM_SCHEMA,
                "number": spec["number"],
                "directory": item_directory,
                "stem": spec["stem"],
                "title": spec["title"],
                "status": status,
                "submission_use_allowed": status == FINAL_STATUS,
                "source_records": source_records,
                "denominator_contract": denominator,
                "file_sha256": file_hashes,
                "blind_images_used": 0,
                "root_cap_region_statistics_included": False,
            }
            item_receipt["item_identity_sha256"] = sha256_json(item_receipt)
            item_receipt_path = item_root / "item_receipt.json"
            atomic_write_json(item_receipt_path, item_receipt)
            relative_receipt = item_receipt_path.relative_to(staging).as_posix()
            receipt_sha = sha256_file(item_receipt_path)
            bundle_hashes[relative_receipt] = receipt_sha
            for relative, digest in file_hashes.items():
                bundle_hashes[(Path(item_directory) / relative).as_posix()] = digest
            item_records[spec["stem"]] = {
                "number": spec["number"],
                "directory": item_directory,
                "title": spec["title"],
                "status": status,
                "item_receipt": relative_receipt,
                "item_receipt_sha256": receipt_sha,
                "item_identity_sha256": item_receipt["item_identity_sha256"],
                "source_roles": list(spec["source_roles"]),
                "denominator_contract": denominator,
                "file_sha256": file_hashes,
            }
        receipt: dict[str, Any] = {
            "schema_version": BUNDLE_SCHEMA,
            "status": status,
            "submission_use_allowed": status == FINAL_STATUS,
            "ordered_item_count": 10,
            "ordered_item_numbers": list(TABLE_NUMBERS),
            "ordered_item_directories": list(TABLE_DIRECTORIES),
            "ordered_item_stems": list(TABLE_STEMS),
            "items": item_records,
            "bundle_file_sha256": bundle_hashes,
            "source_authority_sha256": {
                role: sha256_file(path) for role, path in sources.items()
            },
            "source_authority_identity": deepcopy(identities),
            "figure_input_manifest_sha256": figure_input_manifest_sha256,
            "figure_input_assembly_identity_sha256": figure_input_assembly_identity_sha256,
            "model_contract_proposal_identity_sha256": model_contract_proposal_identity_sha256,
            "blind_images_used": 0,
            "root_cap_region_statistics_included": False,
        }
        receipt["bundle_identity_sha256"] = sha256_json(receipt)
        atomic_write_json(staging / BUNDLE_RECEIPT, receipt)
        _require(not destination.exists(), f"supplementary output appeared during assembly: {destination}")
        os.replace(staging, destination)
        return validate_supplementary_table_data_bundle(
            destination / BUNDLE_RECEIPT,
            require_final=status == FINAL_STATUS,
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def validate_supplementary_table_data_bundle(
    receipt_path: str | Path,
    *,
    require_final: bool,
) -> dict[str, Any]:
    """Validate receipt identities, exact file closure, slot counts, and hashes."""

    receipt_file = _safe_file(receipt_path, "supplementary bundle receipt")
    root = receipt_file.parent.resolve()
    _require(receipt_file.name == BUNDLE_RECEIPT, "supplementary bundle receipt filename changed")
    receipt = _json(receipt_file, "supplementary bundle receipt")
    _require(receipt.get("schema_version") == BUNDLE_SCHEMA, "supplementary bundle schema changed")
    expected_status = FINAL_STATUS if require_final else receipt.get("status")
    _require(expected_status in {FINAL_STATUS, PROVISIONAL_STATUS}, "supplementary bundle status changed")
    _require(receipt.get("status") == expected_status, "supplementary bundle final/provisional status mismatch")
    _require(receipt.get("submission_use_allowed") is (expected_status == FINAL_STATUS), "supplementary submission-use flag changed")
    _require(receipt.get("blind_images_used") == 0, "supplementary bundle blind guard changed")
    _require(receipt.get("root_cap_region_statistics_included") is False, "root-cap region entered supplementary bundle")
    for field in (
        "figure_input_manifest_sha256",
        "figure_input_assembly_identity_sha256",
        "model_contract_proposal_identity_sha256",
    ):
        _require(_is_sha256(receipt.get(field)), f"supplementary bundle {field} is invalid")
    _require(receipt.get("ordered_item_count") == 10, "supplementary item count is not ten")
    _require(receipt.get("ordered_item_numbers") == list(TABLE_NUMBERS), "supplementary item number order changed")
    _require(
        receipt.get("ordered_item_directories") == list(TABLE_DIRECTORIES),
        "supplementary item physical-directory order changed",
    )
    _require(receipt.get("ordered_item_stems") == list(TABLE_STEMS), "supplementary item stem order changed")
    identity = receipt.get("bundle_identity_sha256")
    _require(_is_sha256(identity), "supplementary bundle identity is invalid")
    unsigned = deepcopy(receipt)
    unsigned.pop("bundle_identity_sha256", None)
    _require(sha256_json(unsigned) == identity, "supplementary bundle identity seal mismatch")
    items = receipt.get("items")
    hashes = receipt.get("bundle_file_sha256")
    authorities = receipt.get("source_authority_sha256")
    authority_identities = receipt.get("source_authority_identity")
    _require(isinstance(items, Mapping) and list(items) == list(TABLE_STEMS), "supplementary ordered item map changed")
    _require(isinstance(hashes, Mapping) and bool(hashes), "supplementary bundle file hashes are missing")
    required_authorities = {
        role for spec in TABLE_SPECS for role in spec["source_roles"]
    }
    _require(
        isinstance(authorities, Mapping)
        and set(authorities) == required_authorities
        and all(_is_sha256(value) for value in authorities.values()),
        "supplementary source-authority hash closure is invalid",
    )
    _require(
        isinstance(authority_identities, Mapping)
        and set(authority_identities) <= required_authorities,
        "supplementary source-authority identity closure is invalid",
    )
    expected_paths = {BUNDLE_RECEIPT, *hashes.keys()}
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    _require(observed_paths == expected_paths, "supplementary bundle exact file closure failed")
    _require(not any(path.is_symlink() for path in root.rglob("*")), "supplementary bundle contains a symlink")
    for relative, digest in hashes.items():
        _require(_is_sha256(digest), f"supplementary bundle invalid SHA-256: {relative}")
        target = (root / relative).resolve()
        _require(target.is_relative_to(root), f"supplementary bundle path escapes root: {relative}")
        _require(sha256_file(target) == digest, f"supplementary bundle file tamper: {relative}")
    for spec, item_directory in zip(TABLE_SPECS, TABLE_DIRECTORIES, strict=True):
        record = items[spec["stem"]]
        _require(
            isinstance(record, Mapping)
            and record.get("number") == spec["number"]
            and record.get("directory") == item_directory
            and record.get("item_receipt")
            == (Path(item_directory) / "item_receipt.json").as_posix()
            and record.get("title") == spec["title"]
            and record.get("source_roles") == list(spec["source_roles"]),
            f"{spec['number']}: supplementary item contract changed",
        )
        item_path = (root / str(record.get("item_receipt"))).resolve()
        _require(item_path.is_relative_to(root), f"{spec['number']}: item receipt escapes bundle")
        _require(sha256_file(item_path) == record.get("item_receipt_sha256"), f"{spec['number']}: item receipt hash mismatch")
        item = _json(item_path, f"{spec['number']} item receipt")
        item_identity = item.get("item_identity_sha256")
        _require(_is_sha256(item_identity) and item_identity == record.get("item_identity_sha256"), f"{spec['number']}: item identity mismatch")
        unsigned_item = deepcopy(item)
        unsigned_item.pop("item_identity_sha256", None)
        _require(sha256_json(unsigned_item) == item_identity, f"{spec['number']}: item identity seal mismatch")
        _require(
            item.get("schema_version") == ITEM_SCHEMA
            and item.get("number") == spec["number"]
            and item.get("directory") == item_directory
            and item.get("stem") == spec["stem"]
            and item.get("title") == spec["title"]
            and item.get("status") == expected_status
            and item.get("submission_use_allowed")
            is (expected_status == FINAL_STATUS),
            f"{spec['number']}: item static/status contract changed",
        )
        _require(item.get("denominator_contract") == record.get("denominator_contract"), f"{spec['number']}: denominator contract differs")
        _require(item.get("file_sha256") == record.get("file_sha256"), f"{spec['number']}: item file map differs")
        source_records = item.get("source_records")
        _require(
            isinstance(source_records, list)
            and [source.get("role") for source in source_records]
            == list(spec["source_roles"]),
            f"{spec['number']}: source-record role/order changed",
        )
        copied_paths: set[str] = set()
        for source in source_records:
            _require(isinstance(source, Mapping), f"{spec['number']}: source record malformed")
            role = str(source["role"])
            copied = source.get("copied_path")
            _require(isinstance(copied, str) and bool(copied), f"{spec['number']}/{role}: copied path missing")
            copied_target = (item_path.parent / copied).resolve()
            _require(copied_target.is_relative_to(item_path.parent), f"{spec['number']}/{role}: copied path escapes item")
            _require(copied not in copied_paths, f"{spec['number']}: duplicate copied source path")
            copied_paths.add(copied)
            _require(
                source.get("source_file_sha256") == authorities[role]
                and item["file_sha256"].get(copied) == authorities[role],
                f"{spec['number']}/{role}: copied source authority differs",
            )
            expected_identity_record = authority_identities.get(role, {})
            _require(
                source.get("identity_field")
                == expected_identity_record.get("identity_field")
                and source.get("identity_sha256")
                == expected_identity_record.get("identity_sha256"),
                f"{spec['number']}/{role}: source logical identity differs",
            )
        _require(item.get("blind_images_used") == 0 and item.get("root_cap_region_statistics_included") is False, f"{spec['number']}: red-line guard changed")
    s9 = items["Table_S09_complete_multitrait_atlas"]["denominator_contract"]
    _require(s9.get("block_A_observed_slots") == 256, "Table S9 Block A slot count changed")
    _require(s9.get("block_B_observed_slots") == 192, "Table S9 Block B slot count changed")
    _require(s9.get("block_C_observed_rows") == 544, "Table S9 Block C cohort row count changed")
    _require(
        s9.get("D15_fixed_effect_rows") == 15
        and s9.get("D15_fixed_effect_family_changed_by_WT_secondary") is False,
        "Table S9 WT secondary block changed the D15 family",
    )
    _require(
        s9.get("block_D_typed_block") == "wt_gate_flow"
        and s9.get("block_E_typed_block") == "wt_experiment_contrasts"
        and s9.get("block_F_typed_block") == "wt_same_day_meta",
        "Table S9 WT typed-block contract changed",
    )
    _require(
        s9.get("wt_secondary_cross_day_pooling_performed") is False
        and s9.get("wt_secondary_unknown_day_meta_analysis_performed") is False
        and s9.get("wt_secondary_clean_full_pooling_performed") is False,
        "Table S9 WT analysis boundary changed",
    )
    return {
        "schema_version": BUNDLE_SCHEMA,
        "status": receipt["status"],
        "receipt_path": str(receipt_file),
        "receipt_sha256": sha256_file(receipt_file),
        "bundle_identity_sha256": identity,
        "ordered_item_count": 10,
        "items": deepcopy(dict(items)),
        "bundle_file_sha256": deepcopy(dict(hashes)),
        "source_authority_sha256": deepcopy(dict(authorities)),
        "source_authority_identity": deepcopy(dict(authority_identities)),
        "figure_input_manifest_sha256": receipt[
            "figure_input_manifest_sha256"
        ],
        "figure_input_assembly_identity_sha256": receipt[
            "figure_input_assembly_identity_sha256"
        ],
        "model_contract_proposal_identity_sha256": receipt[
            "model_contract_proposal_identity_sha256"
        ],
        "submission_use_allowed": receipt["submission_use_allowed"],
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }


__all__ = [
    "BUNDLE_DIRECTORY",
    "BUNDLE_RECEIPT",
    "BUNDLE_SCHEMA",
    "FINAL_STATUS",
    "ITEM_SCHEMA",
    "PROVISIONAL_STATUS",
    "SOURCE_MAP_SCHEMA",
    "SupplementaryTableError",
    "TABLE_NUMBERS",
    "TABLE_DIRECTORIES",
    "TABLE_SPECS",
    "TABLE_STEMS",
    "materialize_supplementary_table_data_bundle",
    "validate_supplementary_table_data_bundle",
]
