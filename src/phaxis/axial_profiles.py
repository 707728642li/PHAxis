"""Fixed distal-axis root-hair abundance and conditional-length profiles."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import ContractError
from .io import atomic_write_json, read_json, sha256_file, sha256_json
from .public_identity import MODEL_BUNDLE_PREFIX, ROOT_EXPERT_PREFIX


PROFILE_FIELDS = (
    "schema_version",
    "task_id",
    "source_image_sha256",
    "experiment_key",
    "condition_code",
    "study_role",
    "developmental_day",
    "genotype_or_construct",
    "temperature_c",
    "formal_statistics_eligible",
    "visible_root_axis_length_um",
    "attachment_axis_valid_fraction",
    "bin_index",
    "bin_start_um",
    "bin_end_um",
    "bin_width_mm",
    "bin_eligible",
    "bin_ineligibility_reason",
    "attached_identity_count",
    "attached_identity_density_per_mm",
    "endpoint_complete_length_count",
    "endpoint_complete_length_support_fraction",
    "conditional_mean_hair_length_um",
    "conditional_median_hair_length_um",
    "measured_total_hair_length_um",
    "measured_total_hair_length_per_root_mm",
    "interval_semantics",
    "root_cap_region_output",
    "stageb_two_point_vector_used_as_length",
    "blind_images_used",
)


COHORT_PROFILE_BINDING_SCHEMA = (
    "PHAxis-distal-axis-profile-cohort-binding-1.0.0"
)
COHORT_PROFILE_BUNDLE_SCHEMA = (
    "PHAxis-distal-axis-cohort-profile-bundle-1.0.0"
)
PRIMARY_COHORT_NAME = "primary_clean261"
SENSITIVITY_COHORT_NAME = "sensitivity_full283"
PRIMARY_COHORT_ROLE = "primary_SHA_disjoint"
SENSITIVITY_COHORT_ROLE = "overlap_contaminated_sensitivity"
_EXPECTED_COHORT_COUNTS = {
    "human_curated443": 443,
    "biological_full": 283,
    "human_curated_overlap": 22,
    "biological_clean": 261,
}
_COHORT_TABLES = (
    "traits",
    "detailed_root_statistics",
    "hair_instances",
    "image_traits",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ContractError(f"{field} must be an explicit boolean")


def _finite(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{field} must be numeric") from error
    if not math.isfinite(number):
        raise ContractError(f"{field} must be finite")
    return number


def _optional_finite(value: Any, *, field: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return _finite(value, field=field)


def _assert_optional_close(
    observed: float | None,
    expected: Any,
    *,
    field: str,
    absolute_tolerance: float = 1e-6,
) -> None:
    expected_number = _optional_finite(expected, field=field)
    if observed is None or expected_number is None:
        if observed is not None or expected_number is not None:
            raise ContractError(f"{field}: null/value semantics differ")
        return
    if not math.isclose(
        observed, expected_number, rel_tol=1e-10, abs_tol=absolute_tolerance
    ):
        raise ContractError(f"{field}: distal-profile/locked-trait values differ")


def _measured_total_or_null(
    lengths: np.ndarray, *, identity_count: int
) -> float | None:
    """Return zero only for a known zero identity denominator.

    A positive identity count with zero endpoint-complete curves is missing
    conditional geometry, not a measured length of zero.
    """

    length_count = int(len(lengths))
    if identity_count < 0 or length_count > identity_count:
        raise ContractError("endpoint-complete length support exceeds identity count")
    if identity_count == 0:
        return 0.0
    if length_count == 0:
        return None
    total = float(lengths.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ContractError("endpoint-complete measured total must be positive")
    return total


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=PROFILE_FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _validate_bins(values: Iterable[Iterable[Any]]) -> tuple[tuple[float, float], ...]:
    bins: list[tuple[float, float]] = []
    for index, value in enumerate(values):
        pair = tuple(value)
        if len(pair) != 2:
            raise ContractError(f"profile bin {index} must contain start/end")
        start = _finite(pair[0], field=f"bins_um[{index}].start")
        end = _finite(pair[1], field=f"bins_um[{index}].end")
        if start < 0.0 or end <= start:
            raise ContractError(f"invalid profile bin {index}")
        if bins and not math.isclose(start, bins[-1][1], rel_tol=0.0, abs_tol=1e-9):
            raise ContractError("profile bins must be contiguous and ordered")
        bins.append((start, end))
    if not bins:
        raise ContractError("at least one profile bin is required")
    return tuple(bins)


def _task_index(rows: Sequence[Mapping[str, str]], *, source: str) -> dict[str, Mapping[str, str]]:
    result: dict[str, Mapping[str, str]] = {}
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        if not task_id or task_id in result:
            raise ContractError(f"{source}: empty or duplicate task_id")
        result[task_id] = row
    return result


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value != value.casefold():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _cohort_membership_digest(task_ids: Iterable[str]) -> str:
    return sha256_json(sorted(str(task_id) for task_id in task_ids))


def _cohort_source_digest(source_hashes: Iterable[str]) -> str:
    return sha256_json(sorted(str(value).casefold() for value in source_hashes))


def _validate_cohort_binding(
    binding: Mapping[str, Any],
    *,
    traits_by_task: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "cohort_name",
        "cohort_role",
        "cohort_tasks",
        "cohort_build_summary_sha256",
        "cohort_build_identity_sha256",
        "cohort_membership_csv_sha256",
        "cohort_task_membership_sha256",
        "cohort_source_image_membership_sha256",
        "blind_images_used",
    }
    normalized = dict(binding)
    if set(normalized) != required:
        raise ContractError("profile export received an incomplete cohort binding")
    if normalized.get("schema_version") != COHORT_PROFILE_BINDING_SCHEMA:
        raise ContractError("unsupported distal-axis profile cohort binding")
    expected_roles = {
        PRIMARY_COHORT_NAME: (PRIMARY_COHORT_ROLE, 261),
        SENSITIVITY_COHORT_NAME: (SENSITIVITY_COHORT_ROLE, 283),
    }
    cohort_name = normalized.get("cohort_name")
    if cohort_name not in expected_roles:
        raise ContractError("profile export cohort name is not canonical")
    expected_role, expected_tasks = expected_roles[str(cohort_name)]
    if (
        normalized.get("cohort_role") != expected_role
        or normalized.get("cohort_tasks") != expected_tasks
        or normalized.get("blind_images_used") != 0
    ):
        raise ContractError("profile export cohort role/count is mislabelled")
    for field in (
        "cohort_build_summary_sha256",
        "cohort_build_identity_sha256",
        "cohort_membership_csv_sha256",
        "cohort_task_membership_sha256",
        "cohort_source_image_membership_sha256",
    ):
        if not _is_sha256(normalized.get(field)):
            raise ContractError(f"profile export cohort binding hash is invalid: {field}")
    task_ids = set(traits_by_task)
    if len(task_ids) != expected_tasks:
        raise ContractError("profile export cohort task count differs from its binding")
    source_hashes = {
        str(row.get("source_image_sha256", "")).casefold()
        for row in traits_by_task.values()
    }
    if len(source_hashes) != expected_tasks or not all(
        _is_sha256(value) for value in source_hashes
    ):
        raise ContractError("profile export cohort source-image identities are invalid")
    if normalized["cohort_task_membership_sha256"] != _cohort_membership_digest(
        task_ids
    ):
        raise ContractError("profile export cohort task membership is misbound")
    if normalized[
        "cohort_source_image_membership_sha256"
    ] != _cohort_source_digest(source_hashes):
        raise ContractError("profile export cohort source-image membership is misbound")
    return normalized


def export_distal_axis_profiles(
    *,
    traits_csv: str | Path,
    hair_instances_csv: str | Path,
    contract_json: str | Path,
    output: str | Path,
    model_contract_proposal: Mapping[str, str],
    model_contract_public_identity: Mapping[str, str],
    traits_summary_json: str | Path | None = None,
    cohort_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export fixed one-millimetre abundance and conditional-length bins.

    The function consumes only final trait tables.  It never reads images,
    annotations, experimental conditions for routing, or blind data.
    """

    traits_path = Path(traits_csv).resolve()
    hairs_path = Path(hair_instances_csv).resolve()
    contract_path = Path(contract_json).resolve()
    destination = Path(output).resolve()
    proposal_fields = dict(model_contract_proposal)
    public_identity = dict(model_contract_public_identity)
    if set(proposal_fields) != {
        "model_contract_proposal_sha256",
        "model_contract_proposal_identity_sha256",
    }:
        raise ContractError("profile export received an invalid model-contract binding")
    try:
        valid_proposal = all(
            len(value) == 64 and int(value, 16) >= 0
            for value in proposal_fields.values()
        )
    except (TypeError, ValueError):
        valid_proposal = False
    if not valid_proposal:
        raise ContractError("profile export model-contract hashes are invalid")
    if set(public_identity) != {
        "model_bundle_id",
        "root_expert_id",
    }:
        raise ContractError("profile export received an invalid public identity")
    if (
        not str(public_identity["model_bundle_id"]).startswith(MODEL_BUNDLE_PREFIX)
        or not str(public_identity["root_expert_id"]).startswith(ROOT_EXPERT_PREFIX)
    ):
        raise ContractError("profile export public identity is invalid")
    trait_summary_path = (
        Path(traits_summary_json).resolve()
        if traits_summary_json is not None
        else traits_path.parent / "summary.json"
    )
    trait_summary = read_json(trait_summary_path)
    for field, expected in {**proposal_fields, **public_identity}.items():
        if trait_summary.get(field) != expected:
            raise ContractError(
                f"profile export upstream traits identity mismatch: {field}"
            )
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {destination}")
    contract = read_json(contract_path)
    if contract.get("schema_version") != "PHAxis-distal-axis-profile-contract-1.0.0":
        raise ContractError("unsupported distal-axis profile contract")
    if contract.get("root_cap_region_output") is not False:
        raise ContractError("root-cap region output is forbidden")
    if contract.get("blind_images_used") != 0:
        raise ContractError("profile contract is blind-tainted")
    if contract.get("stageb_two_point_vector_used_as_length") is not False:
        raise ContractError("Stage-B two-point vectors cannot be length measurements")
    bins = _validate_bins(contract.get("bins_um", ()))
    trait_rows = _read_csv(traits_path)
    hair_rows = _read_csv(hairs_path)
    traits_by_task = _task_index(trait_rows, source="traits")
    validated_cohort_binding = (
        _validate_cohort_binding(cohort_binding, traits_by_task=traits_by_task)
        if cohort_binding is not None
        else None
    )
    hairs_by_task: dict[str, list[Mapping[str, str]]] = {task_id: [] for task_id in traits_by_task}
    hair_ids: set[tuple[str, str]] = set()
    for hair in hair_rows:
        task_id = str(hair.get("task_id", "")).strip()
        if task_id not in hairs_by_task:
            raise ContractError(f"hair_instances contains unknown task_id: {task_id}")
        hair_id = str(hair.get("hair_id", "")).strip()
        key = (task_id, hair_id)
        if not hair_id or key in hair_ids:
            raise ContractError(f"empty or duplicate hair identity: {key}")
        hair_ids.add(key)
        if str(hair.get("source_image_sha256", "")).casefold() != str(
            traits_by_task[task_id].get("source_image_sha256", "")
        ).casefold():
            raise ContractError(f"{task_id}: trait/hair source-image hash mismatch")
        hairs_by_task[task_id].append(hair)

    rows: list[dict[str, Any]] = []
    eligible_rows = 0
    attached_identities = 0
    length_identities = 0
    locked_window_crosschecks = 0
    for task_id in sorted(traits_by_task):
        trait = traits_by_task[task_id]
        formal = _bool(
            trait.get("formal_statistics_eligible"),
            field=f"{task_id}.formal_statistics_eligible",
        )
        root_length = _finite(
            trait.get("visible_root_axis_length_um"),
            field=f"{task_id}.visible_root_axis_length_um",
        )
        if root_length < 0.0:
            raise ContractError(f"{task_id}: visible root length cannot be negative")
        valid_fraction = _optional_finite(
            trait.get("attachment_axis_valid_fraction"),
            field=f"{task_id}.attachment_axis_valid_fraction",
        )
        identity_count = len(hairs_by_task[task_id])
        if identity_count == 0:
            if valid_fraction is not None:
                raise ContractError(
                    f"{task_id}: zero hair identities require null attachment fraction"
                )
        elif valid_fraction is None:
            raise ContractError(
                f"{task_id}: positive hair identity count requires attachment fraction"
            )
        elif not 0.0 <= valid_fraction <= 1.0:
            raise ContractError(f"{task_id}: invalid attachment fraction")
        valid_hairs: list[tuple[float, float | None]] = []
        for hair in hairs_by_task[task_id]:
            valid = _bool(
                hair.get("attachment_valid_within_40um"),
                field=f"{task_id}.{hair['hair_id']}.attachment_valid",
            )
            axis = _optional_finite(
                hair.get("attachment_distance_from_distal_point_um"),
                field=f"{task_id}.{hair['hair_id']}.axis_distance",
            )
            length_eligible = _bool(
                hair.get("complete_length_measurement_eligible"),
                field=f"{task_id}.{hair['hair_id']}.length_eligible",
            )
            length = _optional_finite(
                hair.get("length_um"), field=f"{task_id}.{hair['hair_id']}.length_um"
            )
            if valid != (axis is not None):
                raise ContractError(f"{task_id}.{hair['hair_id']}: attachment flag/value drift")
            if length_eligible != (length is not None):
                raise ContractError(f"{task_id}.{hair['hair_id']}: length flag/value drift")
            if length is not None and length < 0.0:
                raise ContractError(f"{task_id}.{hair['hair_id']}: negative length")
            if axis is not None:
                if axis < -1e-6 or axis > root_length + 1e-6:
                    raise ContractError(f"{task_id}.{hair['hair_id']}: attachment outside root axis")
                valid_hairs.append((max(axis, 0.0), length))
        observed_fraction = (
            len(valid_hairs) / identity_count if identity_count else None
        )
        if observed_fraction is not None and not math.isclose(
            observed_fraction, valid_fraction, rel_tol=0.0, abs_tol=5e-7
        ):
            raise ContractError(f"{task_id}: attachment fraction does not match hair table")

        # The fixed bins include the already locked 1--4 mm image-level window.
        # Prove that the new spatial table recomposes that existing trait exactly;
        # this prevents a second, subtly different phenotype definition.
        if "distal_window_1_4mm_eligible" in trait:
            window_eligible = formal and root_length >= 4000.0
            if _bool(
                trait.get("distal_window_1_4mm_eligible"),
                field=f"{task_id}.distal_window_1_4mm_eligible",
            ) != window_eligible:
                raise ContractError(f"{task_id}: locked 1--4 mm eligibility drift")
            window_pairs = [pair for pair in valid_hairs if 1000.0 <= pair[0] < 4000.0]
            window_lengths = np.asarray(
                [length for _axis, length in window_pairs if length is not None],
                dtype=np.float64,
            )
            expected_count = len(window_pairs) if window_eligible else None
            if expected_count is None:
                _assert_optional_close(
                    None,
                    trait.get("local_hair_count_1_4mm"),
                    field=f"{task_id}.local_hair_count_1_4mm",
                )
            elif int(_finite(
                trait.get("local_hair_count_1_4mm"),
                field=f"{task_id}.local_hair_count_1_4mm",
            )) != expected_count:
                raise ContractError(f"{task_id}: locked 1--4 mm count drift")
            _assert_optional_close(
                expected_count / 3.0 if expected_count is not None else None,
                trait.get("local_hair_density_per_mm_1_4mm"),
                field=f"{task_id}.local_hair_density_per_mm_1_4mm",
            )
            _assert_optional_close(
                float(window_lengths.mean())
                if window_eligible and len(window_lengths)
                else None,
                trait.get("local_mean_hair_length_um_1_4mm"),
                field=f"{task_id}.local_mean_hair_length_um_1_4mm",
            )
            _assert_optional_close(
                float(np.median(window_lengths))
                if window_eligible and len(window_lengths)
                else None,
                trait.get("local_median_hair_length_um_1_4mm"),
                field=f"{task_id}.local_median_hair_length_um_1_4mm",
            )
            window_total = (
                _measured_total_or_null(
                    window_lengths, identity_count=len(window_pairs)
                )
                if window_eligible
                else None
            )
            _assert_optional_close(
                window_total / 3.0 if window_total is not None else None,
                trait.get("local_total_hair_length_um_per_root_mm_1_4mm"),
                field=f"{task_id}.local_total_hair_length_um_per_root_mm_1_4mm",
            )
            locked_window_crosschecks += 1

        for bin_index, (start, end) in enumerate(bins):
            width_mm = (end - start) / 1000.0
            if not formal:
                eligible = False
                reason = "formal_statistics_ineligible"
            elif root_length + 1e-9 < end:
                eligible = False
                reason = "visible_root_axis_shorter_than_bin_end"
            else:
                eligible = True
                reason = None
            in_bin = [pair for pair in valid_hairs if start <= pair[0] < end] if eligible else []
            lengths = np.asarray(
                [length for _axis, length in in_bin if length is not None],
                dtype=np.float64,
            )
            count = len(in_bin)
            length_count = int(len(lengths))
            measured_total = (
                _measured_total_or_null(lengths, identity_count=count)
                if eligible
                else None
            )
            row = {
                "schema_version": "PHAxis-distal-axis-profile-row-1.0.0",
                "task_id": task_id,
                "source_image_sha256": trait["source_image_sha256"],
                "experiment_key": trait.get("experiment_key", ""),
                "condition_code": trait.get("condition_code", ""),
                "study_role": trait.get("study_role", ""),
                "developmental_day": trait.get("developmental_day", ""),
                "genotype_or_construct": trait.get("genotype_or_construct", ""),
                "temperature_c": trait.get("temperature_c", ""),
                "formal_statistics_eligible": formal,
                "visible_root_axis_length_um": root_length,
                "attachment_axis_valid_fraction": valid_fraction,
                "bin_index": bin_index,
                "bin_start_um": start,
                "bin_end_um": end,
                "bin_width_mm": width_mm,
                "bin_eligible": eligible,
                "bin_ineligibility_reason": reason,
                "attached_identity_count": count if eligible else None,
                "attached_identity_density_per_mm": count / width_mm if eligible else None,
                "endpoint_complete_length_count": length_count if eligible else None,
                "endpoint_complete_length_support_fraction": (
                    length_count / count if eligible and count else None
                ),
                "conditional_mean_hair_length_um": (
                    float(lengths.mean()) if eligible and length_count else None
                ),
                "conditional_median_hair_length_um": (
                    float(np.median(lengths)) if eligible and length_count else None
                ),
                "measured_total_hair_length_um": measured_total,
                "measured_total_hair_length_per_root_mm": (
                    measured_total / width_mm if measured_total is not None else None
                ),
                "interval_semantics": "left_closed_right_open",
                "root_cap_region_output": False,
                "stageb_two_point_vector_used_as_length": False,
                "blind_images_used": 0,
            }
            if set(row) != set(PROFILE_FIELDS):
                raise AssertionError("distal-axis profile field drift")
            rows.append(row)
            if eligible:
                eligible_rows += 1
                attached_identities += count
                length_identities += length_count

    destination.mkdir(parents=True, exist_ok=True)
    profile_path = destination / "distal_axis_profiles.csv"
    _atomic_csv(profile_path, rows)
    summary: dict[str, Any] = {
        "schema_version": "PHAxis-distal-axis-profile-export-1.0.0",
        "status": "completed",
        "tasks": len(traits_by_task),
        "bins_per_task": len(bins),
        "rows": len(rows),
        "eligible_rows": eligible_rows,
        "attached_identity_observations": attached_identities,
        "endpoint_complete_length_observations": length_identities,
        "locked_1_4mm_trait_crosscheck_tasks": locked_window_crosschecks,
        "locked_1_4mm_trait_crosscheck_mismatches": 0,
        "coordinate_origin": "distal_root_cap_point",
        "bins_um": [[start, end] for start, end in bins],
        "interval_semantics": "left_closed_right_open",
        "derived_profile_not_additional_image_level_trait_count": True,
        "traits_csv_sha256": sha256_file(traits_path),
        "hair_instances_csv_sha256": sha256_file(hairs_path),
        "profile_contract_sha256": sha256_file(contract_path),
        "profiles_csv_sha256": sha256_file(profile_path),
        "root_cap_region_output": False,
        "stageb_two_point_vector_used_as_length": False,
        "condition_metadata_used_for_model_routing": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        **proposal_fields,
        **public_identity,
    }
    if validated_cohort_binding is not None:
        summary["cohort_binding"] = validated_cohort_binding
    summary["export_identity_sha256"] = sha256_json(summary)
    atomic_write_json(destination / "summary.json", summary)
    return summary


def _verified_identity(
    payload: Mapping[str, Any], *, field: str, label: str
) -> str:
    identity = payload.get(field)
    if not _is_sha256(identity):
        raise ContractError(f"{label} identity is absent or invalid")
    unsigned = dict(payload)
    unsigned.pop(field, None)
    if sha256_json(unsigned) != identity:
        raise ContractError(f"{label} identity mismatch")
    return str(identity)


def _verified_hash(path: Path, expected: Any, *, label: str) -> str:
    if not path.is_file() or not _is_sha256(expected):
        raise ContractError(f"{label} is absent from the cohort receipt")
    observed = sha256_file(path)
    if observed != expected:
        raise ContractError(f"{label} hash mismatch")
    return observed


def _rows_by_task(
    path: Path,
    *,
    expected_tasks: set[str],
    membership_sources: Mapping[str, str],
    label: str,
) -> dict[str, dict[str, str]]:
    rows = _read_csv(path)
    indexed = {
        task_id: dict(row)
        for task_id, row in _task_index(rows, source=label).items()
    }
    if set(indexed) != expected_tasks:
        raise ContractError(f"{label} task membership differs from cohort_membership")
    for task_id, row in indexed.items():
        if str(row.get("source_image_sha256", "")).casefold() != membership_sources[
            task_id
        ]:
            raise ContractError(f"{label}/{task_id}: source-image identity mismatch")
    return indexed


def _hair_rows_by_identity(
    path: Path,
    *,
    expected_tasks: set[str],
    membership_sources: Mapping[str, str],
    label: str,
) -> dict[tuple[str, str], dict[str, str]]:
    rows = _read_csv(path)
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        hair_id = str(row.get("hair_id", "")).strip()
        key = (task_id, hair_id)
        if task_id not in expected_tasks or not hair_id or key in indexed:
            raise ContractError(f"{label}: unknown task or duplicate/empty hair identity")
        if str(row.get("source_image_sha256", "")).casefold() != membership_sources[
            task_id
        ]:
            raise ContractError(f"{label}/{task_id}/{hair_id}: source-image mismatch")
        indexed[key] = dict(row)
    return indexed


def _validate_cohort_export_authority(
    *,
    cohorts_root: Path,
    traits_summary_path: Path,
    proposal_fields: Mapping[str, str],
    public_identity: Mapping[str, str],
) -> dict[str, Any]:
    summary_path = cohorts_root / "summary.json"
    lock_path = cohorts_root / "analysis_contract_lock.json"
    membership_path = cohorts_root / "cohort_membership.csv"
    summary = read_json(summary_path)
    lock = read_json(lock_path)
    if (
        summary.get("schema_version") != "PHAxis-biological-cohorts-1.0"
        or summary.get("status")
        != "completed_without_fitting_biological_effect_models"
        or summary.get("blind_images_used") != 0
        or summary.get("canonical_annotations_read") is not False
        or summary.get("root_cap_region_statistics_included") is not False
        or summary.get("cohort_directories")
        != {
            "primary": PRIMARY_COHORT_NAME,
            "sensitivity": SENSITIVITY_COHORT_NAME,
        }
        or summary.get("counts") != _EXPECTED_COHORT_COUNTS
    ):
        raise ContractError("invalid, mislabelled, or blind-tainted cohort receipt")
    cohort_build_identity = _verified_identity(
        summary,
        field="cohort_build_identity_sha256",
        label="biological cohort receipt",
    )
    if (
        lock.get("schema_version") != "PHAxis-biological-cohort-lock-1.0"
        or lock.get("status") != "postresult_software_transition_provenance_lock"
        or lock.get("blind_images_used") != 0
        or lock.get("canonical_annotations_read") is not False
        or lock.get("cohort_counts") != _EXPECTED_COHORT_COUNTS
    ):
        raise ContractError("invalid or blind-tainted biological cohort lock")
    _verified_identity(
        lock,
        field="cohort_lock_identity_sha256",
        label="biological cohort lock",
    )
    expected_identity = {**dict(proposal_fields), **dict(public_identity)}
    for label, payload in (("cohort receipt", summary), ("cohort lock", lock)):
        for field, expected in expected_identity.items():
            if payload.get(field) != expected:
                raise ContractError(f"{label}/model-contract identity mismatch: {field}")
    if summary.get("input_sha256", {}).get(
        "trait_export_summary"
    ) != sha256_file(traits_summary_path):
        raise ContractError("cohort receipt is not bound to the supplied traits summary")
    output_hashes = summary.get("output_sha256")
    lock_hashes = lock.get("output_table_sha256")
    if not isinstance(output_hashes, Mapping) or not isinstance(lock_hashes, Mapping):
        raise ContractError("cohort table hash authority is incomplete")
    if output_hashes.get("analysis_contract_lock") != sha256_file(lock_path):
        raise ContractError("cohort lock file hash mismatch")
    expected_lock_hashes = {
        key: value
        for key, value in output_hashes.items()
        if key != "analysis_contract_lock"
    }
    if dict(lock_hashes) != expected_lock_hashes:
        raise ContractError("cohort receipt and lock disagree on table hashes")
    membership_sha = _verified_hash(
        membership_path,
        output_hashes.get("cohort_membership"),
        label="cohort membership table",
    )
    for name in ("cohort_condition_counts", "acquisition_batch_condition_audit"):
        _verified_hash(
            cohorts_root / f"{name}.csv",
            output_hashes.get(name),
            label=name,
        )

    membership_rows = _read_csv(membership_path)
    membership_index = _task_index(
        membership_rows, source="cohort_membership"
    )
    if len(membership_index) != 283:
        raise ContractError("cohort membership is not exact full283")
    membership_sources: dict[str, str] = {}
    clean_ids: set[str] = set()
    overlap_ids: set[str] = set()
    for task_id, row in membership_index.items():
        source_sha = str(row.get("source_image_sha256", "")).casefold()
        if not _is_sha256(source_sha):
            raise ContractError(f"{task_id}: invalid cohort source-image hash")
        include_clean = _bool(
            row.get("primary_clean_sha_disjoint_include"),
            field=f"{task_id}.primary_clean_sha_disjoint_include",
        )
        include_full = _bool(
            row.get("sensitivity_full_include"),
            field=f"{task_id}.sensitivity_full_include",
        )
        human_overlap = _bool(
            row.get("recomputed_human443_overlap"),
            field=f"{task_id}.recomputed_human443_overlap",
        )
        if not include_full or include_clean == human_overlap:
            raise ContractError(f"{task_id}: clean/full overlap membership is inconsistent")
        overlap_tasks = str(row.get("overlap_human_task_ids", "")).strip()
        overlap_splits = str(row.get("overlap_human_splits", "")).strip()
        if human_overlap != bool(overlap_tasks) or human_overlap != bool(overlap_splits):
            raise ContractError(f"{task_id}: recomputed HumanCurated443 overlap is mislabelled")
        membership_sources[task_id] = source_sha
        (clean_ids if include_clean else overlap_ids).add(task_id)
    if (
        len(set(membership_sources.values())) != 283
        or len(clean_ids) != 261
        or len(overlap_ids) != 22
        or clean_ids & overlap_ids
        or clean_ids | overlap_ids != set(membership_index)
    ):
        raise ContractError("clean261/full283 source membership does not close exactly")

    table_rows: dict[str, dict[str, Any]] = {}
    for cohort_name, expected_ids in (
        (PRIMARY_COHORT_NAME, clean_ids),
        (SENSITIVITY_COHORT_NAME, set(membership_index)),
    ):
        cohort_hashes = output_hashes.get(cohort_name)
        if not isinstance(cohort_hashes, Mapping) or set(cohort_hashes) != set(
            _COHORT_TABLES
        ):
            raise ContractError(f"{cohort_name}: cohort table hash set is incomplete")
        cohort_root = cohorts_root / cohort_name
        for table_name in _COHORT_TABLES:
            _verified_hash(
                cohort_root / f"{table_name}.csv",
                cohort_hashes.get(table_name),
                label=f"{cohort_name}/{table_name}",
            )
        table_rows[cohort_name] = {
            "traits": _rows_by_task(
                cohort_root / "traits.csv",
                expected_tasks=expected_ids,
                membership_sources=membership_sources,
                label=f"{cohort_name}/traits",
            ),
            "detailed_root_statistics": _rows_by_task(
                cohort_root / "detailed_root_statistics.csv",
                expected_tasks=expected_ids,
                membership_sources=membership_sources,
                label=f"{cohort_name}/detailed_root_statistics",
            ),
            "image_traits": _rows_by_task(
                cohort_root / "image_traits.csv",
                expected_tasks=expected_ids,
                membership_sources=membership_sources,
                label=f"{cohort_name}/image_traits",
            ),
            "hair_instances": _hair_rows_by_identity(
                cohort_root / "hair_instances.csv",
                expected_tasks=expected_ids,
                membership_sources=membership_sources,
                label=f"{cohort_name}/hair_instances",
            ),
        }

    primary_rows = table_rows[PRIMARY_COHORT_NAME]
    sensitivity_rows = table_rows[SENSITIVITY_COHORT_NAME]
    for table_name in ("traits", "detailed_root_statistics", "image_traits"):
        if any(
            primary_rows[table_name][task_id]
            != sensitivity_rows[table_name].get(task_id)
            for task_id in clean_ids
        ):
            raise ContractError(
                f"{table_name}: clean261 is not a byte-semantic subset of full283"
            )
    expected_primary_hairs = {
        key: row
        for key, row in sensitivity_rows["hair_instances"].items()
        if key[0] in clean_ids
    }
    if primary_rows["hair_instances"] != expected_primary_hairs:
        raise ContractError(
            "hair_instances: clean261 is not an identity-exact subset of full283"
        )
    return {
        "summary": summary,
        "summary_sha256": sha256_file(summary_path),
        "cohort_build_identity_sha256": cohort_build_identity,
        "lock_sha256": sha256_file(lock_path),
        "membership_sha256": membership_sha,
        "membership_sources": membership_sources,
        "primary_task_ids": clean_ids,
        "sensitivity_task_ids": set(membership_index),
        "overlap_task_ids": overlap_ids,
    }


def export_cohort_distal_axis_profiles(
    *,
    cohorts_root: str | Path,
    contract_json: str | Path,
    output: str | Path,
    model_contract_proposal: Mapping[str, str],
    model_contract_public_identity: Mapping[str, str],
    traits_summary_json: str | Path,
) -> dict[str, Any]:
    """Create exact clean261 and full283 profile exports in one transaction.

    The cohort receipt, its lock, all four cohort tables and the explicit
    membership table are rehashed before either profile is produced.  The two
    child summaries bind their biological cohort role and exact task/source
    membership inside their existing ``export_identity_sha256`` seal.
    """

    cohort_root = Path(cohorts_root).resolve()
    contract_path = Path(contract_json).resolve()
    destination = Path(output).resolve()
    traits_summary_path = Path(traits_summary_json).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {destination}")
    proposal_fields = dict(model_contract_proposal)
    public_identity = dict(model_contract_public_identity)
    # Reuse the legacy exporter as the single authority for validating the
    # proposal/public identity and computing each spatial phenotype table.
    authority = _validate_cohort_export_authority(
        cohorts_root=cohort_root,
        traits_summary_path=traits_summary_path,
        proposal_fields=proposal_fields,
        public_identity=public_identity,
    )
    summary_sha = authority["summary_sha256"]
    membership_sha = authority["membership_sha256"]
    build_identity = authority["cohort_build_identity_sha256"]
    bindings: dict[str, dict[str, Any]] = {}
    for cohort_name, cohort_role, task_key in (
        (PRIMARY_COHORT_NAME, PRIMARY_COHORT_ROLE, "primary_task_ids"),
        (
            SENSITIVITY_COHORT_NAME,
            SENSITIVITY_COHORT_ROLE,
            "sensitivity_task_ids",
        ),
    ):
        task_ids = set(authority[task_key])
        source_hashes = {
            authority["membership_sources"][task_id] for task_id in task_ids
        }
        bindings[cohort_name] = {
            "schema_version": COHORT_PROFILE_BINDING_SCHEMA,
            "cohort_name": cohort_name,
            "cohort_role": cohort_role,
            "cohort_tasks": len(task_ids),
            "cohort_build_summary_sha256": summary_sha,
            "cohort_build_identity_sha256": build_identity,
            "cohort_membership_csv_sha256": membership_sha,
            "cohort_task_membership_sha256": _cohort_membership_digest(task_ids),
            "cohort_source_image_membership_sha256": _cohort_source_digest(
                source_hashes
            ),
            "blind_images_used": 0,
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.cohort-profiles.",
            dir=destination.parent,
        )
    )
    try:
        child_summaries: dict[str, dict[str, Any]] = {}
        for cohort_name in (PRIMARY_COHORT_NAME, SENSITIVITY_COHORT_NAME):
            child_summaries[cohort_name] = export_distal_axis_profiles(
                traits_csv=cohort_root / cohort_name / "traits.csv",
                hair_instances_csv=cohort_root / cohort_name / "hair_instances.csv",
                contract_json=contract_path,
                output=staging / cohort_name,
                model_contract_proposal=proposal_fields,
                model_contract_public_identity=public_identity,
                traits_summary_json=traits_summary_path,
                cohort_binding=bindings[cohort_name],
            )
        primary_summary = child_summaries[PRIMARY_COHORT_NAME]
        sensitivity_summary = child_summaries[SENSITIVITY_COHORT_NAME]
        bundle: dict[str, Any] = {
            "schema_version": COHORT_PROFILE_BUNDLE_SCHEMA,
            "status": "completed",
            "cohort_directories": {
                "primary": PRIMARY_COHORT_NAME,
                "sensitivity": SENSITIVITY_COHORT_NAME,
            },
            "counts": dict(_EXPECTED_COHORT_COUNTS),
            "primary_is_strict_task_subset_of_sensitivity": True,
            "primary_sensitivity_task_overlap": 261,
            "sensitivity_only_human443_overlap_tasks": 22,
            "cohort_build_summary_sha256": summary_sha,
            "cohort_build_identity_sha256": build_identity,
            "cohort_lock_sha256": authority["lock_sha256"],
            "cohort_membership_csv_sha256": membership_sha,
            "traits_summary_sha256": sha256_file(traits_summary_path),
            "profile_contract_sha256": sha256_file(contract_path),
            "cohort_exports": {
                PRIMARY_COHORT_NAME: {
                    "summary_sha256": sha256_file(
                        staging / PRIMARY_COHORT_NAME / "summary.json"
                    ),
                    "profiles_csv_sha256": sha256_file(
                        staging
                        / PRIMARY_COHORT_NAME
                        / "distal_axis_profiles.csv"
                    ),
                    "export_identity_sha256": primary_summary[
                        "export_identity_sha256"
                    ],
                    "cohort_task_membership_sha256": bindings[
                        PRIMARY_COHORT_NAME
                    ]["cohort_task_membership_sha256"],
                },
                SENSITIVITY_COHORT_NAME: {
                    "summary_sha256": sha256_file(
                        staging / SENSITIVITY_COHORT_NAME / "summary.json"
                    ),
                    "profiles_csv_sha256": sha256_file(
                        staging
                        / SENSITIVITY_COHORT_NAME
                        / "distal_axis_profiles.csv"
                    ),
                    "export_identity_sha256": sensitivity_summary[
                        "export_identity_sha256"
                    ],
                    "cohort_task_membership_sha256": bindings[
                        SENSITIVITY_COHORT_NAME
                    ]["cohort_task_membership_sha256"],
                },
            },
            "root_cap_region_output": False,
            "stageb_two_point_vector_used_as_length": False,
            "canonical_annotations_read": False,
            "blind_images_used": 0,
            **proposal_fields,
            **public_identity,
        }
        bundle["cohort_profile_bundle_identity_sha256"] = sha256_json(bundle)
        atomic_write_json(staging / "summary.json", bundle)
        os.replace(staging, destination)
        return bundle
    finally:
        if staging.exists():
            shutil.rmtree(staging)
