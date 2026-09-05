from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "scripts/phaxis"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import build_paper_first_biology_artifacts as builder  # noqa: E402
from phaxis.biological_analysis import (  # noqa: E402
    RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST,
    RAW_EFFECT_HC3_INTERVAL,
    RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL,
    RAW_EFFECT_OLS_MEAN_CONTRAST,
    group_summaries,
    raw_median_bootstrap_seed,
)
from phaxis.io import atomic_write_json, sha256_file, sha256_json  # noqa: E402
from phaxis.public_identity import (  # noqa: E402
    PUBLIC_SYSTEM_DERIVATION,
    PUBLIC_SYSTEM_IDENTITY_SCHEMA,
    derive_public_identity,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _seal(payload: dict, field: str) -> dict:
    payload.pop(field, None)
    payload[field] = sha256_json(payload)
    return payload


def _proposal(path: Path) -> dict:
    checkpoints = [_hash(f"checkpoint-{index}") for index in range(5)]
    stageb = {
        "expert_id": builder.STAGEB_EXPERT_ID,
        "checkpoint_sha256": checkpoints,
        "selected_score_threshold": 0.42,
        "candidate_bundle_identity_sha256": _hash("candidate"),
        "selection_receipt_identity_sha256": _hash("selection"),
        "selected_model_metadata_identity_sha256": _hash("selected-metadata"),
    }
    root_bundle = _hash("current-root-bundle")
    root_pipeline = _hash("current-root-pipeline")
    root_audit = _hash("current-root-audit")
    public = derive_public_identity(
        stageb, root_bundle_identity_sha256=root_bundle
    )
    payload = {
        "schema_version": "PHAxis-model-contract-1.0.0",
        "product": "PHAxis",
        "product_version": "1.0.0",
        "formal_release_status": "passed_proposal_not_official",
        "model_bundle_id": public["model_bundle_id"],
        "public_system_identity": {
            "schema_version": PUBLIC_SYSTEM_IDENTITY_SCHEMA,
            "identity_sha256": public["public_system_identity_sha256"],
            "derivation": PUBLIC_SYSTEM_DERIVATION,
        },
        "expert_boundary": {
            "root_point_scale_continuity_statistics": public["root_expert_id"],
            "hair_identity_and_count": builder.STAGEB_EXPERT_ID,
        },
        "root_expert": {
            "provider_role": public["root_provider_role"],
            "expert_id": public["root_expert_id"],
            "bundle_identity_sha256": root_bundle,
            "pipeline_identity_sha256": root_pipeline,
            "fresh_exact283_audit_identity_sha256": root_audit,
            "root_bundle_authority": {
                "bundle_identity_sha256": root_bundle,
                "pipeline_identity_sha256": root_pipeline,
            },
        },
        "hair_identity_count_expert": {
            "expert_id": builder.STAGEB_EXPERT_ID,
            "checkpoint_policy": builder.STAGEB_CHECKPOINT_POLICY,
            "checkpoint_sha256_in_member_order": checkpoints,
        },
        "data_contract": {
            "train_images": 399,
            "development_images": 44,
            "validation_labels_used_for_gradient_or_early_stopping": False,
        },
        "red_lines": {
            "blind_images_used": 0,
            "canonical_annotations_read_during_inference": False,
            "condition_metadata_used_for_routing": False,
            "root_cap_region_statistics_included": False,
            "legacy_v1_runtime_dependency": False,
            "rhaxiscc_runtime_dependency": False,
        },
        "promotion": {
            "schema_version": "PHAxis-model-contract-promotion-1.0",
            "status": "validated_proposal_not_applied",
            "official_apply_performed": False,
            "stageb_binding": stageb,
            "checkpoint_file_sha256_in_member_order": checkpoints,
            "formal_gate_source_sha256": {
                "train399_candidate": _hash("candidate-file"),
                "train399_selection": _hash("selection-file"),
                "train399_evaluation": _hash("evaluation-file"),
                "root_exact283": _hash("root-audit-file"),
            },
            "formal_gate_identity_sha256": {
                "candidate_bundle_identity_sha256": stageb[
                    "candidate_bundle_identity_sha256"
                ],
                "selection_receipt_identity_sha256": stageb[
                    "selection_receipt_identity_sha256"
                ],
                "selected_model_metadata_identity_sha256": stageb[
                    "selected_model_metadata_identity_sha256"
                ],
                "root_exact283_audit_identity_sha256": root_audit,
            },
        },
    }
    _seal(payload, "model_contract_identity_sha256")
    atomic_write_json(path, payload)
    return payload


def _clean_tables(path: Path, proposal: dict) -> tuple[Path, Path, pd.DataFrame]:
    conditions = builder.GROUP_ORDER
    condition_values = {
        conditions[0]: (5.0, 100.0, 250.0, 52.0, 5200.0),
        conditions[1]: (6.0, 110.0, 240.0, 51.0, 5100.0),
        conditions[2]: (8.0, 120.0, 220.0, 55.0, 5500.0),
        conditions[3]: (10.0, 135.0, 200.0, 54.0, 5400.0),
    }
    rows: list[dict] = []
    images: list[dict] = []
    ordinal = 0
    for condition in conditions:
        for replicate in range(20):
            values = condition_values[condition]
            task_id = f"D15-{ordinal:03d}"
            image_sha = _hash(f"image-{ordinal}")
            jitter = (replicate - 9.5) * 0.05
            row = {
                "task_id": task_id,
                "source_image_sha256": image_sha,
                "experiment_key": "D15_8d",
                "condition_code": condition,
                "study_role": "rhd6_factorial_8d_primary",
                "formal_statistics_eligible": True,
                "hair_count": int(values[0] + replicate % 3),
                "hair_length_measurement_hair_count": 2,
                "local_hair_count_1_4mm": int(values[0] + replicate % 3),
                "local_median_hair_length_um_1_4mm": values[1] + jitter,
                "first_hair_ge40um_distance_from_distal_point_um": values[2] + jitter,
                "median_root_width_um": values[3] + jitter,
                "visible_root_axis_length_um": values[4] + jitter,
                "genotype_or_construct": "RHD6-EV" if "EV" in condition else "RHD6-OE",
                "temperature_c": 22 if condition.endswith("22C") else 30,
            }
            rows.append(row)
            images.append(
                {
                    "task_id": task_id,
                    "source_image_sha256": image_sha,
                    "model_bundle_id": proposal["model_bundle_id"],
                    "root_expert_id": proposal["root_expert"]["expert_id"],
                    "hair_identity_count_expert_id": builder.STAGEB_EXPERT_ID,
                    "visible_root_axis_length_um": row[
                        "visible_root_axis_length_um"
                    ],
                    "shootward_endpoint_border_visible": replicate % 4 != 0,
                    "root_cap_region_output": False,
                    "blind_images_used": 0,
                }
            )
            ordinal += 1
    while ordinal < 261:
        task_id = f"WT-{ordinal:03d}"
        image_sha = _hash(f"image-{ordinal}")
        row = {
            "task_id": task_id,
            "source_image_sha256": image_sha,
            "experiment_key": "WT_secondary",
            "condition_code": "WT_22C",
            "study_role": "wt_temperature_block",
            "formal_statistics_eligible": True,
            "hair_count": 4,
            "hair_length_measurement_hair_count": 1,
            "local_hair_count_1_4mm": 4,
            "local_median_hair_length_um_1_4mm": 90.0,
            "first_hair_ge40um_distance_from_distal_point_um": 300.0,
            "median_root_width_um": 50.0,
            "visible_root_axis_length_um": 5000.0,
            "genotype_or_construct": "WT",
            "temperature_c": 22,
        }
        rows.append(row)
        images.append(
            {
                "task_id": task_id,
                "source_image_sha256": image_sha,
                "model_bundle_id": proposal["model_bundle_id"],
                "root_expert_id": proposal["root_expert"]["expert_id"],
                "hair_identity_count_expert_id": builder.STAGEB_EXPERT_ID,
                "visible_root_axis_length_um": 5000.0,
                "shootward_endpoint_border_visible": True,
                "root_cap_region_output": False,
                "blind_images_used": 0,
            }
        )
        ordinal += 1
    clean = pd.DataFrame(rows)
    image = pd.DataFrame(images)
    clean_path = path / "clean_traits.csv"
    image_path = path / "clean_image_traits.csv"
    _write_csv(clean_path, clean)
    _write_csv(image_path, image)
    return clean_path, image_path, clean


def _cohort_condition_counts(path: Path, clean: pd.DataFrame) -> Path:
    rows: list[dict] = []
    grouped = clean.groupby(
        ["study_role", "experiment_key", "condition_code"],
        sort=True,
        dropna=False,
    )
    for keys, cell in grouped:
        clean_units = len(cell)
        clean_formal = int(cell["formal_statistics_eligible"].astype(bool).sum())
        is_d15 = keys[1] == "D15_8d"
        added_overlap = 2 if is_d15 else 14
        rows.extend(
            [
                {
                    "cohort": "primary_clean261",
                    "study_role": keys[0],
                    "experiment_key": keys[1],
                    "condition_code": keys[2],
                    "units": clean_units,
                    "formal_statistics_eligible": clean_formal,
                    "recomputed_human443_overlap": 0,
                },
                {
                    "cohort": "sensitivity_full283",
                    "study_role": keys[0],
                    "experiment_key": keys[1],
                    "condition_code": keys[2],
                    "units": clean_units + added_overlap,
                    "formal_statistics_eligible": clean_formal + added_overlap,
                    "recomputed_human443_overlap": added_overlap,
                },
            ]
        )
    table_path = path / "cohort_condition_counts.csv"
    _write_csv(table_path, pd.DataFrame(rows))
    return table_path


def _effect_tables(path: Path, clean: pd.DataFrame) -> tuple[Path, Path, Path]:
    primary_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    effects = tuple(builder.EFFECT_SOURCE.values())
    formal = clean[
        clean["study_role"].eq("rhd6_factorial_8d_primary")
        & clean["experiment_key"].eq("D15_8d")
    ]
    for endpoint_index, endpoint in enumerate(builder.ENDPOINT_ORDER):
        model_component = "count_rate" if endpoint_index == 0 else "continuous"
        for effect_index, effect in enumerate(effects):
            estimate = 0.9 + 0.08 * endpoint_index + 0.04 * effect_index
            low = max(0.2, estimate - 0.1)
            high = estimate + 0.1
            if endpoint == builder.ENDPOINT_ORDER[1]:
                medians = {
                    condition: float(
                        formal.loc[formal["condition_code"].eq(condition), endpoint].median()
                    )
                    for condition in builder.GROUP_ORDER
                }
                ev22, ev30, oe22, oe30 = (
                    medians[condition] for condition in builder.GROUP_ORDER
                )
                raw_by_effect = {
                    effects[0]: 0.5 * ((oe22 - ev22) + (oe30 - ev30)),
                    effects[1]: 0.5 * ((ev30 - ev22) + (oe30 - oe22)),
                    effects[2]: (oe30 - oe22) - (ev30 - ev22),
                }
                raw = raw_by_effect[effect]
                raw_estimand = RAW_EFFECT_FACTORIAL_MEDIAN_CONTRAST
                raw_interval = RAW_EFFECT_MEDIAN_BOOTSTRAP_INTERVAL
                raw_replicates = 5000
                raw_seed = raw_median_bootstrap_seed(
                    seed=20260823,
                    field=builder.ENDPOINT_ORDER[1],
                    component="continuous",
                )
            else:
                raw = (endpoint_index + 1) * (effect_index + 1) * 0.5
                raw_estimand = RAW_EFFECT_OLS_MEAN_CONTRAST
                raw_interval = RAW_EFFECT_HC3_INTERVAL
                raw_replicates = 0
                raw_seed = np.nan
            common = {
                "endpoint": endpoint,
                "effect": effect,
                "model_component": model_component,
                "n": 80,
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "effect_scale": "ratio",
                "raw_effect_estimate": raw,
                "raw_effect_ci95_low": raw - 0.2,
                "raw_effect_ci95_high": raw + 0.2,
                "raw_effect_estimand": raw_estimand,
                "raw_effect_interval_method": raw_interval,
                "raw_effect_bootstrap_replicates": raw_replicates,
                "raw_effect_bootstrap_seed": raw_seed,
                "standardized_effect": raw / 2.0,
                "standardized_ci95_low": (raw - 0.2) / 2.0,
                "standardized_ci95_high": (raw + 0.2) / 2.0,
                "causal_treatment_claim_allowed": False,
                "p_value_model_BH_FDR": 0.05,
            }
            primary_rows.append(
                {
                    "cohort": "primary_clean261",
                    "cohort_role": "primary_SHA_disjoint",
                    **common,
                }
            )
            sensitivity_rows.append(
                {
                    "cohort": "sensitivity_full283",
                    "cohort_role": "overlap_contaminated_sensitivity",
                    **{
                        **common,
                        "n": 88,
                        "estimate": estimate + 0.02,
                        "ci95_low": low + 0.02,
                        "ci95_high": high + 0.02,
                    },
                }
            )
    primary = pd.DataFrame(primary_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    stability_rows = []
    for clean_row, full_row in zip(
        primary.to_dict("records"), sensitivity.to_dict("records"), strict=True
    ):
        same = (
            clean_row["estimate"] >= 1 and full_row["estimate"] >= 1
        ) or (clean_row["estimate"] <= 1 and full_row["estimate"] <= 1)
        stability_rows.append(
            {
                "endpoint": clean_row["endpoint"],
                "model_component": clean_row["model_component"],
                "effect": clean_row["effect"],
                "n_clean": clean_row["n"],
                "estimate_clean": clean_row["estimate"],
                "ci95_low_clean": clean_row["ci95_low"],
                "ci95_high_clean": clean_row["ci95_high"],
                "n_full": full_row["n"],
                "estimate_full": full_row["estimate"],
                "ci95_low_full": full_row["ci95_low"],
                "ci95_high_full": full_row["ci95_high"],
                "same_direction_clean_vs_full": same,
            }
        )
    primary_path = path / "primary_tests.csv"
    sensitivity_path = path / "sensitivity_tests.csv"
    stability_path = path / "effect_stability.csv"
    _write_csv(primary_path, primary)
    _write_csv(sensitivity_path, sensitivity)
    _write_csv(stability_path, pd.DataFrame(stability_rows))
    return primary_path, sensitivity_path, stability_path


def _profiles(
    path: Path,
    *,
    proposal: dict,
    cohort_path: Path,
    cohort_summary: dict,
    clean_path: Path,
    clean: pd.DataFrame,
) -> tuple[Path, Path, dict, Path]:
    rows = []
    for cohort, role in (
        ("primary_clean261", "primary_SHA_disjoint"),
        ("sensitivity_full283", "overlap_contaminated_sensitivity"),
    ):
        for condition_index, condition in enumerate(builder.GROUP_ORDER):
            for bin_index in range(5):
                base = float(2 + condition_index + bin_index)
                rows.append(
                    {
                        "cohort": cohort,
                        "cohort_role": role,
                        "condition_code": condition,
                        "bin_start_um": bin_index * 1000.0,
                        "bin_end_um": (bin_index + 1) * 1000.0,
                        "eligible_source_units": 20,
                        "length_measurable_source_units": 16,
                        "mean_attached_identity_count": base,
                        "mean_attached_identity_count_ci95_low": base - 0.2,
                        "mean_attached_identity_count_ci95_high": base + 0.2,
                        "endpoint_complete_support_fraction": 0.8,
                        "endpoint_complete_support_fraction_ci95_low": 0.7,
                        "endpoint_complete_support_fraction_ci95_high": 0.9,
                        "median_of_source_unit_conditional_median_length_um": 100 + base,
                        "median_of_source_unit_conditional_median_length_um_ci95_low": 95 + base,
                        "median_of_source_unit_conditional_median_length_um_ci95_high": 105 + base,
                        "bootstrap_replicates_requested": 10000,
                        "unit_of_analysis": "one_source_image_root_unit",
                    }
                )
    table_path = path / "profile_table.csv"
    _write_csv(table_path, pd.DataFrame(rows))
    profile_export = {
        "schema_version": "PHAxis-distal-axis-profile-export-1.0.0",
        "status": "completed",
        "tasks": 261,
        "bins_per_task": 5,
        "rows": 1305,
        "locked_1_4mm_trait_crosscheck_tasks": 261,
        "locked_1_4mm_trait_crosscheck_mismatches": 0,
        "traits_csv_sha256": sha256_file(clean_path),
        "hair_instances_csv_sha256": _hash("clean-hair-instances"),
        "profile_contract_sha256": _hash("profile-contract"),
        "profiles_csv_sha256": _hash("clean-profile-table"),
        "root_cap_region_output": False,
        "stageb_two_point_vector_used_as_length": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        "cohort_binding": {
            "schema_version": "PHAxis-distal-axis-profile-cohort-binding-1.0.0",
            "cohort_name": "primary_clean261",
            "cohort_role": "primary_SHA_disjoint",
            "cohort_tasks": 261,
            "cohort_build_summary_sha256": sha256_file(cohort_path),
            "cohort_build_identity_sha256": cohort_summary[
                "cohort_build_identity_sha256"
            ],
            "cohort_membership_csv_sha256": cohort_summary["output_sha256"][
                "cohort_membership"
            ],
            "cohort_task_membership_sha256": sha256_json(
                sorted(clean["task_id"].astype(str))
            ),
            "cohort_source_image_membership_sha256": sha256_json(
                sorted(clean["source_image_sha256"].astype(str).str.casefold())
            ),
            "blind_images_used": 0,
        },
        "model_contract_proposal_sha256": sha256_file(path / "proposal.json"),
        "model_contract_proposal_identity_sha256": proposal[
            "model_contract_identity_sha256"
        ],
        "model_bundle_id": proposal["model_bundle_id"],
        "root_expert_id": proposal["root_expert"]["expert_id"],
    }
    _seal(profile_export, "export_identity_sha256")
    export_path = path / "primary_profile_export_summary.json"
    atomic_write_json(export_path, profile_export)
    summary = {
        "schema_version": "PHAxis-distal-axis-profile-analysis-1.0.0",
        "status": "completed_exploratory_source_unit_profile_summaries",
        "rows": 40,
        "primary_rows": 20,
        "sensitivity_rows": 20,
        "output_table_sha256": sha256_file(table_path),
        "unit_of_analysis": "one_source_image_root_unit",
        "individual_hairs_treated_as_independent_replicates": False,
        "hypothesis_tests_performed": False,
        "primary_profile_summary_sha256": sha256_file(export_path),
        "primary_profile_identity_sha256": profile_export[
            "export_identity_sha256"
        ],
        "root_cap_region_statistics_included": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
    }
    summary_path = path / "profile_summary.json"
    return table_path, summary_path, summary, export_path


def _dataset(root: Path) -> tuple[builder.Inputs, dict[str, str], dict]:
    root.mkdir(parents=True)
    proposal_path = root / "proposal.json"
    proposal = _proposal(proposal_path)
    clean_path, image_path, clean = _clean_tables(root, proposal)
    condition_counts_path = _cohort_condition_counts(root, clean)
    membership_path = root / "cohort_membership.csv"
    _write_csv(
        membership_path,
        pd.DataFrame(
            {
                "task_id": [*clean["task_id"].astype(str), *[f"OVERLAP-{i:02d}" for i in range(22)]],
                "sensitivity_full_include": True,
            }
        ),
    )

    d15 = clean[
        clean["study_role"].eq("rhd6_factorial_8d_primary")
        & clean["experiment_key"].eq("D15_8d")
    ]
    group = group_summaries(
        d15,
        [(endpoint, endpoint) for endpoint in builder.ENDPOINT_ORDER],
        scope="primary_clean261",
    )
    group_path = root / "primary_group_summaries.csv"
    _write_csv(group_path, group)
    primary_path, sensitivity_path, stability_path = _effect_tables(root, clean)

    cohort_summary = {
        "schema_version": "PHAxis-biological-cohorts-1.0",
        "status": "completed_without_fitting_biological_effect_models",
        "cohort_directories": {
            "primary": "primary_clean261",
            "sensitivity": "sensitivity_full283",
        },
        "counts": {
            "human_curated443": 443,
            "biological_clean": 261,
            "human_curated_overlap": 22,
            "biological_full": 283,
        },
        "primary_model_scope": {
            "study_role": "rhd6_factorial_8d_primary",
            "experiment_key": "D15_8d",
            "conditions": list(builder.GROUP_ORDER),
            "clean_units": 80,
            "full_units": 88,
        },
        "output_sha256": {
            "primary_clean261": {
                "traits": sha256_file(clean_path),
                "image_traits": sha256_file(image_path),
            },
            "cohort_membership": sha256_file(membership_path),
            "cohort_condition_counts": sha256_file(condition_counts_path),
        },
        "root_cap_region_statistics_included": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        "model_contract_proposal_sha256": sha256_file(proposal_path),
        "model_contract_proposal_identity_sha256": proposal[
            "model_contract_identity_sha256"
        ],
        "model_bundle_id": proposal["model_bundle_id"],
        "root_expert_id": proposal["root_expert"]["expert_id"],
    }
    _seal(cohort_summary, "cohort_build_identity_sha256")
    cohort_path = root / "cohorts_summary.json"
    atomic_write_json(cohort_path, cohort_summary)

    biology_summary = {
        "schema_version": "PHAxis-exploratory-biological-analysis-1.0",
        "status": "completed_exploratory_clean_primary_full_sensitivity",
        "primary_cohort": "primary_clean261",
        "sensitivity_cohort": "sensitivity_full283",
        "primary_result_rows": 15,
        "sensitivity_result_rows": 15,
        "D15_fixed_effect_family_changed_by_WT_secondary": False,
        "wt_secondary_cross_day_pooling_performed": False,
        "wt_secondary_clean_full_pooling_performed": False,
        "cohort_build_summary_sha256": sha256_file(cohort_path),
        "output_table_sha256": {
            "group_summaries": sha256_file(group_path),
            "primary_tests": sha256_file(primary_path),
            "sensitivity_tests": sha256_file(sensitivity_path),
            "clean_full_comparison": sha256_file(stability_path),
        },
        "root_cap_region_statistics_included": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        "model_contract_proposal_sha256": sha256_file(proposal_path),
        "model_contract_proposal_identity_sha256": proposal[
            "model_contract_identity_sha256"
        ],
        "model_bundle_id": proposal["model_bundle_id"],
        "root_expert_id": proposal["root_expert"]["expert_id"],
    }
    _seal(biology_summary, "analysis_identity_sha256")
    biology_path = root / "biology_summary.json"
    atomic_write_json(biology_path, biology_summary)

    (
        profile_table_path,
        profile_summary_path,
        profile_summary,
        primary_profile_export_path,
    ) = _profiles(
        root,
        proposal=proposal,
        cohort_path=cohort_path,
        cohort_summary=cohort_summary,
        clean_path=clean_path,
        clean=clean,
    )
    profile_summary.update(
        {
            "model_contract_proposal_sha256": sha256_file(proposal_path),
            "model_contract_proposal_identity_sha256": proposal[
                "model_contract_identity_sha256"
            ],
            "model_bundle_id": proposal["model_bundle_id"],
            "root_expert_id": proposal["root_expert"]["expert_id"],
        }
    )
    _seal(profile_summary, "analysis_identity_sha256")
    atomic_write_json(profile_summary_path, profile_summary)

    # Keep the synthetic evidence authority internally exact.  Production
    # validation deliberately recomputes these D15 totals from the trait table,
    # so a fixture must not use convenient but contradictory 40/50 constants.
    formal_d15 = clean[
        clean["formal_statistics_eligible"].astype(bool)
        & clean["experiment_key"].eq("D15_8d")
        & clean["study_role"].eq("rhd6_factorial_8d_primary")
    ]
    support_rows = []
    for condition in builder.GROUP_ORDER:
        cell = formal_d15[formal_d15["condition_code"].eq(condition)]
        identity_hairs = int(cell["hair_count"].sum())
        supported_hairs = int(cell["hair_length_measurement_hair_count"].sum())
        support_rows.append(
            {
                "condition_code": condition,
                "support_fraction": supported_hairs / identity_hairs,
                "supported_hairs": supported_hairs,
                "identity_hairs": identity_hairs,
                "source_units": int(len(cell)),
            }
        )
    support = pd.DataFrame(support_rows)
    support_path = root / "assurance_support.csv"
    _write_csv(support_path, support)
    stageb = proposal["promotion"]["stageb_binding"]
    assurance = {
        "schema_version": "PHAxis-measurement-assurance-receipt-1.0",
        "status": "completed_locked_qc_development_assurance",
        "scope": "QC-development measurement assurance; non-independent",
        "source_table_sha256": {"support": sha256_file(support_path)},
        "source_authority_sha256": {
            "clean_traits": sha256_file(clean_path),
            "cohorts_receipt": sha256_file(cohort_path),
        },
        "shared_stageb_authority": {
            **stageb,
            "checkpoint_policy": builder.STAGEB_CHECKPOINT_POLICY,
        },
        "independent_accuracy_claim_allowed": False,
        "root_cap_region_statistics_included": False,
        "blind_images_used": 0,
    }
    _seal(assurance, "measurement_assurance_identity_sha256")
    assurance_path = root / "assurance_receipt.json"
    atomic_write_json(assurance_path, assurance)

    inputs = builder.Inputs(
        model_contract_proposal=proposal_path,
        cohorts_summary=cohort_path,
        cohort_condition_counts=condition_counts_path,
        clean_traits=clean_path,
        clean_image_traits=image_path,
        biology_summary=biology_path,
        primary_group_summaries=group_path,
        primary_tests=primary_path,
        sensitivity_tests=sensitivity_path,
        effect_stability=stability_path,
        profile_summary=profile_summary_path,
        profile_table=profile_table_path,
        primary_profile_export_summary=primary_profile_export_path,
        assurance_receipt=assurance_path,
        assurance_support=support_path,
    )
    hashes = {role: sha256_file(path) for role, path in inputs.as_dict().items()}
    return inputs, hashes, proposal


def _build(root: Path, output: Path) -> tuple[dict, builder.Inputs, dict, dict]:
    inputs, hashes, proposal = _dataset(root)
    receipt = builder.build_paper_first_biology_artifacts(
        inputs=inputs,
        expected_sha256=hashes,
        expected_model_bundle_id=proposal["model_bundle_id"],
        expected_root_expert_id=proposal["root_expert"]["expert_id"],
        output=output,
    )
    return receipt, inputs, hashes, proposal


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_focused_contract_materializes_only_fig5_table3_and_values(tmp_path: Path) -> None:
    output = tmp_path / "paper_output"
    receipt, inputs, hashes, proposal = _build(tmp_path / "current_inputs", output)

    assert receipt["status"] == builder.STATUS
    assert receipt["figure_suite_generated"] is False
    assert receipt["benchmark_required"] is False
    assert receipt["release_packaging_required"] is False
    assert receipt["author_metadata_required"] is False
    assert receipt["gpu_program_started"] is False
    assert receipt["blind_images_used"] == 0
    assert set(path.name for path in output.iterdir()) == {
        "fig5_source_package",
        "input_locks.json",
        "manuscript_values.json",
        "receipt.json",
        "table3.md",
        "table3_source.csv",
    }
    fig5 = json.loads(
        (output / "fig5_source_package/source_package.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(fig5["resources"]) == {
        "phenotype_points",
        "phenotype_effects",
        "assurance_support",
        "axial_profiles",
        "narrative_decision",
    }
    assert fig5["model_bundle_id"] == proposal["model_bundle_id"]
    assert fig5["blind_images_used"] == 0
    table = (output / "table3.md").read_text(encoding="utf-8")
    assert table.count("\n| **") == 5
    assert table.index("H08/N") < table.index("H11/L") < table.index("H07/F")
    assert table.index("H07/F") < table.index("R07/W") < table.index("R01/A")
    values = json.loads((output / "manuscript_values.json").read_text(encoding="utf-8"))
    assert values["scope"] == "Fig5_and_Table3_machine_values_only"
    assert values["claim_contract"]["current_train399_exact283_only"] is True
    assert "FINAL_D15_ABUNDANCE_PATTERN" in values["entries"]
    assert "FINAL_D15_AXIAL_SUPPORT_PATTERN" in values["entries"]
    expected_new_values = {
        "FINAL_D15_CLEAN_POOL_CELL_N": "20 / 20 / 20 / 20",
        "FINAL_D15_FULL_POOL_CELL_N": "22 / 22 / 22 / 22",
        "FINAL_PROFILE_CROSSCHECK_MATCH_N": 261,
        "FINAL_PROFILE_CROSSCHECK_TOTAL_N": 261,
    }
    for token, expected in expected_new_values.items():
        assert values["entries"][token]["value"] == expected
    for token in ("FINAL_D15_CLEAN_POOL_CELL_N", "FINAL_D15_FULL_POOL_CELL_N"):
        assert values["entries"][token]["derivation"]["source_file_sha256"] == {
            "cohort_condition_counts": hashes["cohort_condition_counts"]
        }
    for token in ("FINAL_PROFILE_CROSSCHECK_MATCH_N", "FINAL_PROFILE_CROSSCHECK_TOTAL_N"):
        assert values["entries"][token]["derivation"]["source_file_sha256"] == {
            "primary_profile_export_summary": hashes[
                "primary_profile_export_summary"
            ]
        }
    decision = json.loads(
        (output / "fig5_source_package/narrative_decision.json").read_text(
            encoding="utf-8"
        )
    )
    compact = values["narrative_decision"]
    assert len(compact["support_mask_bits"]) == 15
    assert compact["support_mask_bits"] == decision["support_mask_bits"]
    assert compact["clean_directions"] == [
        cell["clean_direction"] for cell in decision["cells"]
    ]
    assert compact["cell_order"] == [
        {"endpoint_key": endpoint, "effect_key": effect}
        for endpoint in builder.ENDPOINT_ORDER
        for effect in builder.EFFECT_ORDER
    ]
    assert compact["source_sha256"] == {
        role: hashes[role]
        for role in ("primary_tests", "sensitivity_tests", "effect_stability")
    }
    assert values["model_contract_proposal_identity_sha256"] == proposal[
        "model_contract_identity_sha256"
    ]
    assert set(receipt["explicit_input_roles"]) == set(builder.INPUT_ROLES)
    assert receipt["input_file_sha256"] == hashes

    with pytest.raises(builder.PaperFirstBuildError, match="overwrite"):
        builder.build_paper_first_biology_artifacts(
            inputs=inputs,
            expected_sha256=hashes,
            expected_model_bundle_id=proposal["model_bundle_id"],
            expected_root_expert_id=proposal["root_expert"]["expert_id"],
            output=output,
        )


def test_rejects_wrong_lineage_and_forbidden_path_markers(tmp_path: Path) -> None:
    inputs, hashes, proposal = _dataset(tmp_path / "current_inputs")
    biology = json.loads(inputs.biology_summary.read_text(encoding="utf-8"))
    biology["model_contract_proposal_identity_sha256"] = "f" * 64
    _seal(biology, "analysis_identity_sha256")
    atomic_write_json(inputs.biology_summary, biology)
    hashes["biology_summary"] = sha256_file(inputs.biology_summary)
    with pytest.raises(builder.PaperFirstBuildError, match="current-lineage mismatch"):
        builder.build_paper_first_biology_artifacts(
            inputs=inputs,
            expected_sha256=hashes,
            expected_model_bundle_id=proposal["model_bundle_id"],
            expected_root_expert_id=proposal["root_expert"]["expert_id"],
            output=tmp_path / "wrong_lineage_output",
        )

    # Restore a valid fixture, then prove that an old-lineage-labelled path is
    # rejected even when its bytes and explicit digest are otherwise valid.
    inputs, hashes, proposal = _dataset(tmp_path / "second_current_inputs")
    old_path = tmp_path / "legacy_443cv" / "primary_tests.csv"
    old_path.parent.mkdir()
    shutil.copyfile(inputs.primary_tests, old_path)
    replaced = builder.Inputs(
        **{
            **inputs.as_dict(),
            "primary_tests": old_path,
        }
    )
    hashes["primary_tests"] = sha256_file(old_path)
    with pytest.raises(builder.PaperFirstBuildError, match="labelled path refused"):
        builder.build_paper_first_biology_artifacts(
            inputs=replaced,
            expected_sha256=hashes,
            expected_model_bundle_id=proposal["model_bundle_id"],
            expected_root_expert_id=proposal["root_expert"]["expert_id"],
            output=tmp_path / "old_path_output",
        )


def test_build_is_byte_deterministic_and_hash_lock_is_fail_closed(tmp_path: Path) -> None:
    inputs, hashes, proposal = _dataset(tmp_path / "current_inputs")
    first = tmp_path / "paper_run_a"
    second = tmp_path / "paper_run_b"
    for output in (first, second):
        builder.build_paper_first_biology_artifacts(
            inputs=inputs,
            expected_sha256=hashes,
            expected_model_bundle_id=proposal["model_bundle_id"],
            expected_root_expert_id=proposal["root_expert"]["expert_id"],
            output=output,
        )
    assert _tree_hashes(first) == _tree_hashes(second)

    tampered = deepcopy(hashes)
    tampered["profile_table"] = "0" * 64
    with pytest.raises(builder.PaperFirstBuildError, match="explicit lock"):
        builder.build_paper_first_biology_artifacts(
            inputs=inputs,
            expected_sha256=tampered,
            expected_model_bundle_id=proposal["model_bundle_id"],
            expected_root_expert_id=proposal["root_expert"]["expert_id"],
            output=tmp_path / "hash_mismatch_output",
        )


def test_new_counts_are_exact_authorities_not_balanced_or_success_inferences(
    tmp_path: Path,
) -> None:
    inputs, hashes, proposal = _dataset(tmp_path / "redistributed_current_inputs")
    condition_counts = pd.read_csv(inputs.cohort_condition_counts)
    first, second = builder.GROUP_ORDER[:2]
    first_mask = (
        condition_counts["cohort"].eq("sensitivity_full283")
        & condition_counts["experiment_key"].eq("D15_8d")
        & condition_counts["condition_code"].eq(first)
    )
    second_mask = (
        condition_counts["cohort"].eq("sensitivity_full283")
        & condition_counts["experiment_key"].eq("D15_8d")
        & condition_counts["condition_code"].eq(second)
    )
    for field in ("units", "formal_statistics_eligible", "recomputed_human443_overlap"):
        condition_counts.loc[first_mask, field] += 1
        condition_counts.loc[second_mask, field] -= 1
    _write_csv(inputs.cohort_condition_counts, condition_counts)

    cohorts = json.loads(inputs.cohorts_summary.read_text(encoding="utf-8"))
    cohorts["output_sha256"]["cohort_condition_counts"] = sha256_file(
        inputs.cohort_condition_counts
    )
    _seal(cohorts, "cohort_build_identity_sha256")
    atomic_write_json(inputs.cohorts_summary, cohorts)

    biology = json.loads(inputs.biology_summary.read_text(encoding="utf-8"))
    biology["cohort_build_summary_sha256"] = sha256_file(inputs.cohorts_summary)
    _seal(biology, "analysis_identity_sha256")
    atomic_write_json(inputs.biology_summary, biology)

    profile_export = json.loads(
        inputs.primary_profile_export_summary.read_text(encoding="utf-8")
    )
    profile_export["cohort_binding"]["cohort_build_summary_sha256"] = sha256_file(
        inputs.cohorts_summary
    )
    profile_export["cohort_binding"]["cohort_build_identity_sha256"] = cohorts[
        "cohort_build_identity_sha256"
    ]
    _seal(profile_export, "export_identity_sha256")
    atomic_write_json(inputs.primary_profile_export_summary, profile_export)

    profile = json.loads(inputs.profile_summary.read_text(encoding="utf-8"))
    profile["primary_profile_summary_sha256"] = sha256_file(
        inputs.primary_profile_export_summary
    )
    profile["primary_profile_identity_sha256"] = profile_export[
        "export_identity_sha256"
    ]
    _seal(profile, "analysis_identity_sha256")
    atomic_write_json(inputs.profile_summary, profile)

    assurance = json.loads(inputs.assurance_receipt.read_text(encoding="utf-8"))
    assurance["source_authority_sha256"]["cohorts_receipt"] = sha256_file(
        inputs.cohorts_summary
    )
    _seal(assurance, "measurement_assurance_identity_sha256")
    atomic_write_json(inputs.assurance_receipt, assurance)

    hashes = {role: sha256_file(path) for role, path in inputs.as_dict().items()}
    output = tmp_path / "redistributed_output"
    builder.build_paper_first_biology_artifacts(
        inputs=inputs,
        expected_sha256=hashes,
        expected_model_bundle_id=proposal["model_bundle_id"],
        expected_root_expert_id=proposal["root_expert"]["expert_id"],
        output=output,
    )
    values = json.loads((output / "manuscript_values.json").read_text(encoding="utf-8"))
    assert values["entries"]["FINAL_D15_FULL_POOL_CELL_N"]["value"] == (
        "23 / 21 / 22 / 22"
    )

    bad_inputs, bad_hashes, bad_proposal = _dataset(
        tmp_path / "bad_profile_crosscheck_inputs"
    )
    bad_export = json.loads(
        bad_inputs.primary_profile_export_summary.read_text(encoding="utf-8")
    )
    bad_export["locked_1_4mm_trait_crosscheck_mismatches"] = 1
    _seal(bad_export, "export_identity_sha256")
    atomic_write_json(bad_inputs.primary_profile_export_summary, bad_export)
    bad_profile = json.loads(bad_inputs.profile_summary.read_text(encoding="utf-8"))
    bad_profile["primary_profile_summary_sha256"] = sha256_file(
        bad_inputs.primary_profile_export_summary
    )
    bad_profile["primary_profile_identity_sha256"] = bad_export[
        "export_identity_sha256"
    ]
    _seal(bad_profile, "analysis_identity_sha256")
    atomic_write_json(bad_inputs.profile_summary, bad_profile)
    bad_hashes = {
        role: sha256_file(path) for role, path in bad_inputs.as_dict().items()
    }
    with pytest.raises(
        builder.PaperFirstBuildError,
        match="exact clean261 trait cross-check",
    ):
        builder.build_paper_first_biology_artifacts(
            inputs=bad_inputs,
            expected_sha256=bad_hashes,
            expected_model_bundle_id=bad_proposal["model_bundle_id"],
            expected_root_expert_id=bad_proposal["root_expert"]["expert_id"],
            output=tmp_path / "bad_profile_crosscheck_output",
        )
