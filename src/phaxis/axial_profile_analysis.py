"""Exploratory source-unit summaries for fixed distal-axis profiles."""

from __future__ import annotations

import csv
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .axial_profiles import (
    COHORT_PROFILE_BINDING_SCHEMA,
    PRIMARY_COHORT_NAME,
    PRIMARY_COHORT_ROLE,
    SENSITIVITY_COHORT_NAME,
    SENSITIVITY_COHORT_ROLE,
)
from .contracts import ContractError
from .io import atomic_write_json, read_json, sha256_file, sha256_json
from .public_identity import MODEL_BUNDLE_PREFIX, ROOT_EXPERT_PREFIX


SUMMARY_FIELDS = (
    "schema_version",
    "cohort",
    "cohort_role",
    "experiment_key",
    "study_role",
    "condition_code",
    "bin_index",
    "bin_start_um",
    "bin_end_um",
    "eligible_source_units",
    "attached_identity_count_sum",
    "mean_attached_identity_count",
    "mean_attached_identity_count_ci95_low",
    "mean_attached_identity_count_ci95_high",
    "mean_attached_identity_density_per_mm",
    "mean_attached_identity_density_per_mm_ci95_low",
    "mean_attached_identity_density_per_mm_ci95_high",
    "length_measurable_source_units",
    "endpoint_complete_length_count_sum",
    "endpoint_complete_support_fraction",
    "endpoint_complete_support_fraction_ci95_low",
    "endpoint_complete_support_fraction_ci95_high",
    "median_of_source_unit_conditional_median_length_um",
    "median_of_source_unit_conditional_median_length_um_ci95_low",
    "median_of_source_unit_conditional_median_length_um_ci95_high",
    "bootstrap_replicates_requested",
    "bootstrap_replicates_finite_abundance",
    "bootstrap_replicates_finite_length",
    "unit_of_analysis",
    "hypothesis_tests_performed",
    "causal_treatment_claim_allowed",
    "blind_images_used",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="raise")
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


def _bool(value: Any, *, field: str) -> bool:
    normalized = str(value).strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ContractError(f"{field} must be an explicit boolean")


def _number(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ContractError(f"{field} must be numeric") from error
    if not math.isfinite(result):
        raise ContractError(f"{field} must be finite")
    return result


def _optional_number(value: Any, *, field: str) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    return _number(value, field=field)


def _group_seed(base_seed: int, values: Sequence[Any]) -> int:
    encoded = "\x1f".join(map(str, values)).encode("utf-8")
    suffix = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return int(np.random.SeedSequence([int(base_seed), suffix]).generate_state(1)[0])


def _bootstrap(
    *,
    matrix: np.ndarray,
    replicates: int,
    seed: int,
    statistic: Callable[[np.ndarray], float],
) -> tuple[float | None, float | None, int]:
    if len(matrix) < 2:
        return None, None, 0
    generator = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = matrix[generator.integers(0, len(matrix), len(matrix))]
        values[index] = statistic(sample)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None, None, 0
    low, high = np.quantile(finite, (0.025, 0.975))
    return float(low), float(high), int(len(finite))


def _verify_profile_root(
    root: Path,
    *,
    expected_cohort: str,
    expected_role: str,
    expected_tasks: int,
) -> tuple[list[dict[str, str]], dict[str, Any], set[str]]:
    summary_path = root / "summary.json"
    profile_path = root / "distal_axis_profiles.csv"
    summary = read_json(summary_path)
    if (
        summary.get("schema_version")
        != "PHAxis-distal-axis-profile-export-1.0.0"
        or summary.get("status") != "completed"
        or summary.get("blind_images_used") != 0
        or summary.get("locked_1_4mm_trait_crosscheck_mismatches") != 0
    ):
        raise ContractError(f"invalid or blind-tainted profile export: {root}")
    if sha256_file(profile_path) != summary.get("profiles_csv_sha256"):
        raise ContractError(f"profile table hash mismatch: {root}")
    export_identity = summary.get("export_identity_sha256")
    unsigned = dict(summary)
    unsigned.pop("export_identity_sha256", None)
    if (
        not isinstance(export_identity, str)
        or len(export_identity) != 64
        or sha256_json(unsigned) != export_identity
    ):
        raise ContractError(f"profile export identity mismatch: {root}")
    binding = summary.get("cohort_binding")
    if not isinstance(binding, Mapping) or (
        binding.get("schema_version") != COHORT_PROFILE_BINDING_SCHEMA
        or binding.get("cohort_name") != expected_cohort
        or binding.get("cohort_role") != expected_role
        or binding.get("cohort_tasks") != expected_tasks
        or summary.get("tasks") != expected_tasks
        or binding.get("blind_images_used") != 0
    ):
        raise ContractError(f"profile export cohort binding is mislabelled: {root}")
    rows = _read_csv(profile_path)
    task_sources: dict[str, str] = {}
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        source_sha = str(row.get("source_image_sha256", "")).casefold()
        if not task_id or len(source_sha) != 64:
            raise ContractError(f"profile row task/source identity is invalid: {root}")
        try:
            int(source_sha, 16)
        except ValueError as error:
            raise ContractError(
                f"profile row source identity is invalid: {root}"
            ) from error
        if task_id in task_sources and task_sources[task_id] != source_sha:
            raise ContractError(f"profile task has multiple source identities: {task_id}")
        task_sources[task_id] = source_sha
        if (
            _bool(row.get("root_cap_region_output"), field="root_cap_region_output")
            or int(row.get("blind_images_used", -1)) != 0
        ):
            raise ContractError(f"profile row is root-cap-region/blind tainted: {root}")
    task_ids = set(task_sources)
    if (
        len(task_ids) != expected_tasks
        or sha256_json(sorted(task_ids))
        != binding.get("cohort_task_membership_sha256")
        or len(set(task_sources.values())) != expected_tasks
        or sha256_json(sorted(task_sources.values()))
        != binding.get("cohort_source_image_membership_sha256")
    ):
        raise ContractError(f"profile rows differ from sealed cohort membership: {root}")
    for field in (
        "cohort_build_summary_sha256",
        "cohort_build_identity_sha256",
        "cohort_membership_csv_sha256",
    ):
        value = binding.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ContractError(f"profile cohort authority hash is invalid: {field}")
        try:
            int(value, 16)
        except ValueError as error:
            raise ContractError(
                f"profile cohort authority hash is invalid: {field}"
            ) from error
    return rows, summary, task_ids


def _summarize(
    rows: Sequence[Mapping[str, str]],
    *,
    cohort: str,
    cohort_role: str,
    contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    scope = contract["primary_scope"]
    expected_conditions = set(scope["condition_codes"])
    selected = [
        row
        for row in rows
        if row.get("experiment_key") == scope["experiment_key"]
        and row.get("study_role") == scope["study_role"]
        and row.get("condition_code") in expected_conditions
        and _bool(row.get("formal_statistics_eligible"), field="formal_statistics_eligible")
        and _bool(row.get("bin_eligible"), field="bin_eligible")
    ]
    if not selected:
        raise ContractError(f"{cohort}: no eligible primary-scope profile rows")
    observed_conditions = {row["condition_code"] for row in selected}
    if observed_conditions != expected_conditions:
        raise ContractError(f"{cohort}: one or more primary conditions are absent")
    groups: dict[tuple[str, int], list[Mapping[str, str]]] = {}
    for row in selected:
        key = (row["condition_code"], int(row["bin_index"]))
        groups.setdefault(key, []).append(row)
    uncertainty = contract["uncertainty"]
    replicates = int(uncertainty["replicates"])
    base_seed = int(uncertainty["random_seed"])
    if replicates <= 0:
        raise ContractError("bootstrap replicate count must be positive")
    result: list[dict[str, Any]] = []
    for (condition, bin_index), group in sorted(groups.items()):
        task_ids = [row["task_id"] for row in group]
        if len(task_ids) != len(set(task_ids)):
            raise ContractError(f"{cohort}/{condition}/bin{bin_index}: duplicate source unit")
        matrix = np.asarray(
            [
                [
                    _number(row["attached_identity_count"], field="attached_identity_count"),
                    _number(
                        row["attached_identity_density_per_mm"],
                        field="attached_identity_density_per_mm",
                    ),
                    _number(
                        row["endpoint_complete_length_count"],
                        field="endpoint_complete_length_count",
                    ),
                    _optional_number(
                        row["conditional_median_hair_length_um"],
                        field="conditional_median_hair_length_um",
                    ),
                ]
                for row in group
            ],
            dtype=np.float64,
        )
        if np.any(matrix[:, :3] < 0.0) or np.any(matrix[:, 2] > matrix[:, 0]):
            raise ContractError(f"{cohort}/{condition}/bin{bin_index}: invalid counts")
        key = (cohort, condition, bin_index)
        count_low, count_high, count_finite = _bootstrap(
            matrix=matrix,
            replicates=replicates,
            seed=_group_seed(base_seed, (*key, "count")),
            statistic=lambda sample: float(sample[:, 0].mean()),
        )
        density_low, density_high, _density_finite = _bootstrap(
            matrix=matrix,
            replicates=replicates,
            seed=_group_seed(base_seed, (*key, "density")),
            statistic=lambda sample: float(sample[:, 1].mean()),
        )
        support_low, support_high, _support_finite = _bootstrap(
            matrix=matrix,
            replicates=replicates,
            seed=_group_seed(base_seed, (*key, "support")),
            statistic=lambda sample: (
                float(sample[:, 2].sum() / sample[:, 0].sum())
                if sample[:, 0].sum() > 0.0
                else float("nan")
            ),
        )
        length_low, length_high, length_finite = _bootstrap(
            matrix=matrix,
            replicates=replicates,
            seed=_group_seed(base_seed, (*key, "length")),
            statistic=lambda sample: (
                float(np.nanmedian(sample[:, 3]))
                if np.isfinite(sample[:, 3]).any()
                else float("nan")
            ),
        )
        identity_sum = float(matrix[:, 0].sum())
        length_sum = float(matrix[:, 2].sum())
        length_values = matrix[np.isfinite(matrix[:, 3]), 3]
        row: dict[str, Any] = {
            "schema_version": "PHAxis-distal-axis-profile-group-summary-1.0.0",
            "cohort": cohort,
            "cohort_role": cohort_role,
            "experiment_key": scope["experiment_key"],
            "study_role": scope["study_role"],
            "condition_code": condition,
            "bin_index": bin_index,
            "bin_start_um": _number(group[0]["bin_start_um"], field="bin_start_um"),
            "bin_end_um": _number(group[0]["bin_end_um"], field="bin_end_um"),
            "eligible_source_units": len(group),
            "attached_identity_count_sum": int(identity_sum),
            "mean_attached_identity_count": float(matrix[:, 0].mean()),
            "mean_attached_identity_count_ci95_low": count_low,
            "mean_attached_identity_count_ci95_high": count_high,
            "mean_attached_identity_density_per_mm": float(matrix[:, 1].mean()),
            "mean_attached_identity_density_per_mm_ci95_low": density_low,
            "mean_attached_identity_density_per_mm_ci95_high": density_high,
            "length_measurable_source_units": int(len(length_values)),
            "endpoint_complete_length_count_sum": int(length_sum),
            "endpoint_complete_support_fraction": (
                length_sum / identity_sum if identity_sum > 0.0 else None
            ),
            "endpoint_complete_support_fraction_ci95_low": support_low,
            "endpoint_complete_support_fraction_ci95_high": support_high,
            "median_of_source_unit_conditional_median_length_um": (
                float(np.median(length_values)) if len(length_values) else None
            ),
            "median_of_source_unit_conditional_median_length_um_ci95_low": length_low,
            "median_of_source_unit_conditional_median_length_um_ci95_high": length_high,
            "bootstrap_replicates_requested": replicates,
            "bootstrap_replicates_finite_abundance": count_finite,
            "bootstrap_replicates_finite_length": length_finite,
            "unit_of_analysis": "one_source_image_root_unit",
            "hypothesis_tests_performed": False,
            "causal_treatment_claim_allowed": False,
            "blind_images_used": 0,
        }
        if set(row) != set(SUMMARY_FIELDS):
            raise AssertionError("profile-analysis field drift")
        result.append(row)
    return result


def analyze_distal_axis_profiles(
    *,
    primary_profiles: str | Path,
    sensitivity_profiles: str | Path,
    contract_json: str | Path,
    output: str | Path,
    model_contract_proposal: Mapping[str, str],
    model_contract_public_identity: Mapping[str, str],
) -> dict[str, Any]:
    destination = Path(output).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {destination}")
    contract_path = Path(contract_json).resolve()
    contract = read_json(contract_path)
    if (
        contract.get("schema_version")
        != "PHAxis-distal-axis-profile-analysis-contract-1.0.0"
        or contract.get("reporting", {}).get("blind_images_used") != 0
        or contract.get("uncertainty", {}).get("hypothesis_tests_performed") is not False
        or contract.get("individual_hairs_treated_as_independent_replicates") is not False
    ):
        raise ContractError("invalid or blind-tainted axial-profile analysis contract")
    primary_root = Path(primary_profiles).resolve()
    sensitivity_root = Path(sensitivity_profiles).resolve()
    if primary_root == sensitivity_root:
        raise ContractError(
            "primary and sensitivity profiles must be distinct cohort-specific exports"
        )
    if (
        contract.get("primary_cohort") != PRIMARY_COHORT_NAME
        or contract.get("sensitivity_cohort") != SENSITIVITY_COHORT_NAME
    ):
        raise ContractError("profile analysis cohort labels are not canonical")
    primary_rows, primary_summary, primary_tasks = _verify_profile_root(
        primary_root,
        expected_cohort=PRIMARY_COHORT_NAME,
        expected_role=PRIMARY_COHORT_ROLE,
        expected_tasks=261,
    )
    sensitivity_rows, sensitivity_summary, sensitivity_tasks = _verify_profile_root(
        sensitivity_root,
        expected_cohort=SENSITIVITY_COHORT_NAME,
        expected_role=SENSITIVITY_COHORT_ROLE,
        expected_tasks=283,
    )
    primary_binding = primary_summary["cohort_binding"]
    sensitivity_binding = sensitivity_summary["cohort_binding"]
    if (
        primary_summary["profiles_csv_sha256"]
        == sensitivity_summary["profiles_csv_sha256"]
        or primary_summary["export_identity_sha256"]
        == sensitivity_summary["export_identity_sha256"]
    ):
        raise ContractError("primary/sensitivity profile bytes or identities are copied")
    shared_authority_fields = (
        "cohort_build_summary_sha256",
        "cohort_build_identity_sha256",
        "cohort_membership_csv_sha256",
    )
    if any(
        primary_binding[field] != sensitivity_binding[field]
        for field in shared_authority_fields
    ):
        raise ContractError("primary/sensitivity profiles use different cohort authorities")
    if not primary_tasks < sensitivity_tasks or len(sensitivity_tasks - primary_tasks) != 22:
        raise ContractError("clean261 is not the exact strict subset of full283")
    primary_by_key = {
        (row["task_id"], row["bin_index"]): dict(row) for row in primary_rows
    }
    sensitivity_by_key = {
        (row["task_id"], row["bin_index"]): dict(row) for row in sensitivity_rows
    }
    if (
        len(primary_by_key) != len(primary_rows)
        or len(sensitivity_by_key) != len(sensitivity_rows)
        or any(
            sensitivity_by_key.get(key) != row
            for key, row in primary_by_key.items()
        )
    ):
        raise ContractError("clean261 profile rows are not an exact subset of full283")
    proposal_fields = dict(model_contract_proposal)
    public_identity = dict(model_contract_public_identity)
    if set(proposal_fields) != {
        "model_contract_proposal_sha256",
        "model_contract_proposal_identity_sha256",
    }:
        raise ContractError("profile analysis received an invalid proposal binding")
    if set(public_identity) != {"model_bundle_id", "root_expert_id"}:
        raise ContractError("profile analysis received an invalid public identity")
    try:
        valid_hashes = all(
            len(value) == 64 and int(value, 16) >= 0
            for value in proposal_fields.values()
        )
    except (TypeError, ValueError):
        valid_hashes = False
    if not valid_hashes or not str(public_identity["model_bundle_id"]).startswith(
        MODEL_BUNDLE_PREFIX
    ) or not str(public_identity["root_expert_id"]).startswith(ROOT_EXPERT_PREFIX):
        raise ContractError("profile analysis model-contract identity is invalid")
    expected_identity = {**proposal_fields, **public_identity}
    for role, profile_summary in (
        ("primary", primary_summary),
        ("sensitivity", sensitivity_summary),
    ):
        for field, expected in expected_identity.items():
            if profile_summary.get(field) != expected:
                raise ContractError(
                    f"{role} profile/model-contract identity mismatch: {field}"
                )
    rows = [
        *_summarize(
            primary_rows,
            cohort=str(contract["primary_cohort"]),
            cohort_role="primary_SHA_disjoint",
            contract=contract,
        ),
        *_summarize(
            sensitivity_rows,
            cohort=str(contract["sensitivity_cohort"]),
            cohort_role="overlap_contaminated_sensitivity",
            contract=contract,
        ),
    ]
    destination.mkdir(parents=True, exist_ok=True)
    table_path = destination / "distal_axis_profile_group_summaries.csv"
    _atomic_csv(table_path, rows)
    summary: dict[str, Any] = {
        "schema_version": "PHAxis-distal-axis-profile-analysis-1.0.0",
        "status": "completed_exploratory_source_unit_profile_summaries",
        "rows": len(rows),
        "primary_rows": sum(row["cohort_role"] == "primary_SHA_disjoint" for row in rows),
        "sensitivity_rows": sum(
            row["cohort_role"] == "overlap_contaminated_sensitivity" for row in rows
        ),
        "analysis_contract_sha256": sha256_file(contract_path),
        "primary_profile_summary_sha256": sha256_file(primary_root / "summary.json"),
        "sensitivity_profile_summary_sha256": sha256_file(
            sensitivity_root / "summary.json"
        ),
        "primary_profile_identity_sha256": primary_summary["export_identity_sha256"],
        "sensitivity_profile_identity_sha256": sensitivity_summary[
            "export_identity_sha256"
        ],
        "output_table_sha256": sha256_file(table_path),
        "unit_of_analysis": "one_source_image_root_unit",
        "individual_hairs_treated_as_independent_replicates": False,
        "hypothesis_tests_performed": False,
        "prospective_experiment_preregistration_claimed": False,
        "causal_treatment_claim_allowed": False,
        "biological_plate_randomization_inference_performed": False,
        "root_cap_region_statistics_included": False,
        "stageb_two_point_vector_used_as_length": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        **proposal_fields,
        **public_identity,
    }
    summary["analysis_identity_sha256"] = sha256_json(summary)
    atomic_write_json(destination / "summary.json", summary)
    return summary
