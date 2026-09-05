from __future__ import annotations

from copy import deepcopy
import csv
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from phaxis.io import sha256_file, sha256_json
from phaxis.supplementary_tables import (
    BUNDLE_RECEIPT,
    FINAL_STATUS,
    SupplementaryTableError,
    TABLE_DIRECTORIES,
    TABLE_SPECS,
    materialize_supplementary_table_data_bundle,
    validate_supplementary_table_data_bundle,
)
from tests.phaxis.test_multitrait_atlas import _fixture as _atlas_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _dataset_tables(root: Path) -> tuple[Path, Path]:
    dataset_rows = []
    split_rows = []
    for index in range(443):
        split = "train" if index < 399 else "val"
        task_id = f"RHAUD-{index + 1:03d}"
        family = f"family-{split}-{index:03d}"
        split_rows.append({"task_id": task_id, "split": split, "family_key": family})
        dataset_rows.append(
            {
                "task_id": task_id,
                "split": split,
                "family_key": family,
                "image_sha256": sha256_json({"image": task_id}),
                "raw_annotation_sha256": sha256_json({"annotation": task_id}),
                "canonical_annotation_relpath": f"annotations/{task_id}.json",
            }
        )
    return (
        _write_csv(root / "dataset_manifest.csv", dataset_rows),
        _write_csv(root / "split_manifest.csv", split_rows),
    )


def _training_authorities(root: Path) -> tuple[Path, dict[int, Path]]:
    members = []
    receipts: dict[int, Path] = {}
    config_sha = sha256_json({"config": "fixed"})
    for index, seed in enumerate(range(2026082801, 2026082806)):
        checkpoint_sha = sha256_json({"checkpoint": seed})
        receipt = {
            "schema_version": "PHAxis-StageB-train399-training-receipt-1.0",
            "status": "completed",
            "seed": seed,
            "epochs": 60,
            "global_steps": 23940,
            "checkpoint_sha256": checkpoint_sha,
            "peak_allocated_mib": 1024.0 + index,
            "peak_reserved_mib": 1536.0 + index,
            "cuda_visible_devices": "1",
            "internal_device": "cuda:0",
            "gpu_name": "synthetic RTX 3090",
            "blind_images_used": 0,
        }
        receipt_path = _write_json(root / f"training_receipt_{seed}.json", receipt)
        receipts[seed] = receipt_path
        members.append(
            {
                "seed": seed,
                "member_id": f"seed_{seed}",
                "member_index": index,
                "epoch": 60,
                "global_step": 23940,
                "checkpoint_sha256": checkpoint_sha,
                "model_state_sha256": sha256_json({"model": seed}),
                "initialization_sha256": sha256_json({"init": seed}),
                "training_receipt_sha256": sha256_file(receipt_path),
            }
        )
    identity_payload = {
        "ensemble_members": 5,
        "training_lock": {"config_sha256": config_sha},
        "members": members,
    }
    candidate = {
        "schema_version": "PHAxis-StageB-train399-candidate-manifest-1.0",
        "status": "locked_candidate_pending_qcdevelopment44_selection",
        "identity_payload": identity_payload,
        "candidate_bundle_identity_sha256": sha256_json(identity_payload),
        "blind_images_used": 0,
    }
    candidate["candidate_manifest_identity_sha256"] = sha256_json(candidate)
    return _write_json(root / "candidate.json", candidate), receipts


def _wt_secondary_authorities(root: Path) -> tuple[dict[str, Path], dict]:
    endpoints = (
        "local_hair_count_1_4mm",
        "local_median_hair_length_um_1_4mm",
        "first_hair_ge40um_distance_from_distal_point_um",
        "median_root_width_um",
        "visible_root_axis_length_um",
    )
    cohorts = (
        ("primary_clean261", "primary_SHA_disjoint"),
        ("sensitivity_full283", "overlap_contaminated_sensitivity"),
    )
    experiments = (
        ("WT7_A", 7, True),
        ("WT7_B", 7, True),
        ("WT7_C", 7, True),
        ("WT8_A", 8, True),
        ("WT_unknown", None, False),
    )
    contrasts = []
    flow = []
    for cohort, cohort_role in cohorts:
        for experiment_index, (experiment, day, meta_eligible) in enumerate(
            experiments
        ):
            for endpoint_index, endpoint in enumerate(endpoints):
                estimate = 1.05 + experiment_index / 100 + endpoint_index / 100
                contrasts.append(
                    {
                        "cohort": cohort,
                        "cohort_role": cohort_role,
                        "endpoint": endpoint,
                        "endpoint_label": endpoint,
                        "model_component": (
                            "count_rate" if endpoint_index == 0 else "continuous_log"
                        ),
                        "experiment_key": experiment,
                        "developmental_day": day,
                        "developmental_day_status": (
                            "known_consistent" if day is not None else "unknown_all_rows"
                        ),
                        "n_total_22C": 4,
                        "n_total_30C": 4,
                        "n_formal_22C": 4,
                        "n_formal_30C": 4,
                        "n_endpoint_22C": 4,
                        "n_endpoint_30C": 4,
                        "mean_22C": 10.0,
                        "mean_30C": 10.0 * estimate,
                        "median_22C": 10.0,
                        "median_30C": 10.0 * estimate,
                        "raw_difference_30C_minus_22C": 10.0 * (estimate - 1.0),
                        "model": "synthetic_log_temperature",
                        "effect_scale": "ratio_30C_over_22C",
                        "log_effect_30C_over_22C": math.log(estimate),
                        "log_effect_standard_error": 0.02,
                        "sampling_variance": 0.0004,
                        "estimate_30C_over_22C": estimate,
                        "ci95_low": estimate - 0.05,
                        "ci95_high": estimate + 0.05,
                        "p_value_model": 0.2,
                        "p_value_model_BH_FDR": 0.3,
                        "reject_model_BH_FDR_0p05": False,
                        "multiplicity_family": (
                            "within_cohort_all_estimated_WT_experiment_by_endpoint_"
                            "contrasts_including_unknown_day"
                        ),
                        "nb2_alpha": None,
                        "poisson_pearson_dispersion_diagnostic": None,
                        "poisson_fallback_used": False,
                        "analysis_status": "estimated",
                        "not_estimable_reason": "",
                        "meta_eligible": meta_eligible,
                        "meta_exclusion_reason": (
                            "" if meta_eligible else "unknown_developmental_day"
                        ),
                        "inference_status": (
                            "secondary_exploratory_within_experiment_association"
                        ),
                    }
                )
                flow.append(
                    {
                        "cohort": cohort,
                        "cohort_role": cohort_role,
                        "experiment_key": experiment,
                        "developmental_day": day,
                        "developmental_day_status": (
                            "known_consistent" if day is not None else "unknown_all_rows"
                        ),
                        "endpoint": endpoint,
                        "endpoint_label": endpoint,
                        "n_total_22C": 4,
                        "n_total_30C": 4,
                        "n_formal_22C": 4,
                        "n_formal_30C": 4,
                        "n_endpoint_22C": 4,
                        "n_endpoint_30C": 4,
                        "base_gate_pass": True,
                        "endpoint_gate_pass": True,
                        "model_status": "estimated",
                        "not_estimable_reason": "",
                        "phenotype_outlier_filter_applied": False,
                    }
                )
    meta = []
    for cohort, cohort_role in cohorts:
        for day, k in ((7, 3), (8, 1)):
            for endpoint in endpoints:
                estimated = k >= 3
                meta.append(
                    {
                        "cohort": cohort,
                        "cohort_role": cohort_role,
                        "endpoint": endpoint,
                        "endpoint_label": endpoint,
                        "model_component": "continuous_log",
                        "developmental_day": day,
                        "k_eligible_experiments": k,
                        "eligible_experiments": (
                            "WT7_A;WT7_B;WT7_C" if day == 7 else "WT8_A"
                        ),
                        "excluded_experiments": "",
                        "model": "random_effects_REML_Hartung_Knapp",
                        "effect_scale": "ratio_30C_over_22C",
                        "log_effect_30C_over_22C": (
                            math.log(1.05) if estimated else None
                        ),
                        "log_effect_standard_error_hartung_knapp": (
                            0.02 if estimated else None
                        ),
                        "estimate_30C_over_22C": 1.05 if estimated else None,
                        "ci95_low": 1.01 if estimated else None,
                        "ci95_high": 1.09 if estimated else None,
                        "p_value_hartung_knapp": 0.2 if estimated else None,
                        "tau2_reml_log_scale": 0.0 if estimated else None,
                        "Q": 1.0 if estimated else None,
                        "Q_df": 2.0 if estimated else None,
                        "Q_p_value": 0.6 if estimated else None,
                        "I2": 0.0 if estimated else None,
                        "I2_percent": 0.0 if estimated else None,
                        "hartung_knapp_scale": 1.0 if estimated else None,
                        "analysis_status": "estimated" if estimated else "not_estimable",
                        "not_estimable_reason": (
                            ""
                            if estimated
                            else "fewer_than_3_estimable_same_day_experiments"
                        ),
                        "cross_day_pooling_performed": False,
                        "unknown_day_contrasts_included": False,
                        "p_value_hartung_knapp_BH_FDR": 0.3 if estimated else None,
                        "reject_hartung_knapp_BH_FDR_0p05": False,
                        "multiplicity_family": (
                            "within_cohort_all_estimated_WT_developmental_day_by_"
                            "endpoint_meta_analyses"
                        ),
                        "inference_status": (
                            "secondary_exploratory_same_day_experiment_replication"
                        ),
                    }
                )
    paths = {
        "source/wt_temperature_qc_flow": _write_csv(
            root / "wt_temperature_model_qc_flow.csv", flow
        ),
        "source/wt_within_experiment_contrasts": _write_csv(
            root / "wt_within_experiment_temperature_contrasts.csv", contrasts
        ),
        "source/wt_within_day_meta_analysis": _write_csv(
            root / "wt_within_day_REML_Hartung_Knapp.csv", meta
        ),
    }
    summary = {
        "schema_version": "PHAxis-exploratory-biological-analysis-1.0",
        "status": "completed_exploratory_clean_primary_full_sensitivity",
        "D15_fixed_effect_rows": 15,
        "D15_fixed_effect_family_changed_by_WT_secondary": False,
        "wt_secondary_analysis": {
            "schema_version": "PHAxis-WT-temperature-secondary-1.0",
            "status": "materialized_as_separate_secondary_family",
            "endpoint_count": 5,
            "minimum_per_temperature_base_and_endpoint": 3,
            "minimum_experiments_per_day_meta_analysis": 3,
            "within_experiment_multiplicity": (
                "Benjamini-Hochberg within each cohort across every estimated "
                "experiment-by-endpoint contrast, including unknown-day contrasts"
            ),
            "within_day_meta_multiplicity": (
                "Benjamini-Hochberg within each cohort across every estimated "
                "developmental-day-by-endpoint meta-analysis"
            ),
            "cross_day_pooling_performed": False,
            "unknown_day_meta_analysis_performed": False,
            "clean_full_pooling_performed": False,
            "D15_fixed_effect_family_changed": False,
        },
        "wt_secondary_within_experiment_rows": len(contrasts),
        "wt_secondary_estimable_within_experiment_rows": len(contrasts),
        "wt_secondary_unknown_day_contrast_rows": sum(
            row["developmental_day"] is None for row in contrasts
        ),
        "wt_secondary_within_day_meta_rows": len(meta),
        "wt_secondary_estimable_within_day_meta_rows": sum(
            row["analysis_status"] == "estimated" for row in meta
        ),
        "wt_secondary_typed_not_estimable_meta_rows": sum(
            row["analysis_status"] == "not_estimable" for row in meta
        ),
        "wt_secondary_cross_day_pooling_performed": False,
        "wt_secondary_unknown_day_meta_analysis_performed": False,
        "wt_secondary_clean_full_pooling_performed": False,
        "wt_secondary_claim_status": (
            "secondary exploratory blocked replication; pooled estimates require "
            "at least three estimable experiments within one developmental day"
        ),
        "output_table_sha256": {
            role.removeprefix("source/"): sha256_file(path)
            for role, path in paths.items()
        },
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }
    summary["analysis_identity_sha256"] = sha256_json(summary)
    return paths, summary


def source_fixture(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    required = {role for spec in TABLE_SPECS for role in spec["source_roles"]}
    dataset, split = _dataset_tables(root)
    candidate, training = _training_authorities(root)
    contract, clean, full, full_image, primary, sensitivity, _hashes, atlas = (
        _atlas_fixture()
    )

    paths: dict[str, Path] = {
        "source/dataset_manifest": dataset,
        "source/split_manifest": split,
        "source/train399_candidate": candidate,
        "source/image_traits_schema": PROJECT_ROOT
        / "configs/phaxis/v1_0/image_traits.schema.json",
        "resource/trait_contract": PROJECT_ROOT
        / "configs/phaxis/v1_0/trait_contract.json",
    }
    for seed, path in training.items():
        paths[f"source/training_receipt_seed_{seed}"] = path

    for role in (
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
    ):
        paths[role] = _write_json(
            root / f"{role.replace('/', '__')}.json",
            {
                "schema_version": f"synthetic-{role}",
                "status": "completed",
                "blind_images_used": 0,
            },
        )

    development = []
    for index in range(44):
        for comparator in ("stageb_train399", "legacy_hybrid"):
            development.append(
                {
                    "source_unit": f"qc-{index:02d}",
                    "comparator": comparator,
                    "gt_count": 10,
                    "predicted_count": 10,
                }
            )
    paths["resource/development_per_image"] = _write_csv(
        root / "development_per_image.csv", development
    )
    tolerance = [
        {"comparator": comparator, "tolerance_um": tolerance_um, "f1": 0.8}
        for comparator in ("stageb_train399", "legacy_hybrid")
        for tolerance_um in (5, 10, 20)
    ]
    paths["resource/development_tolerance"] = _write_csv(
        root / "development_tolerance.csv", tolerance
    )
    paths["resource/development_threshold"] = _write_csv(
        root / "development_threshold.csv", [{"threshold": 0.5, "f1": 0.8}]
    )
    paths["source/historical_oof_per_image"] = _write_csv(
        root / "historical_oof.csv", [{"source_unit": "oof-1", "f1": 0.7}]
    )
    paths["resource/assurance_metrics"] = _write_csv(
        root / "assurance_metrics.csv", [{"metric_key": "root_dice", "n": 44}]
    )
    paths["resource/assurance_pairs"] = _write_csv(
        root / "assurance_pairs.csv", [{"pair_type": "hair", "source_unit": "qc-00"}]
    )
    paths["resource/assurance_support"] = _write_csv(
        root / "assurance_support.csv", [{"source_unit": "qc-00", "support": 1.0}]
    )
    paths["source/assurance_topology"] = _write_csv(
        root / "assurance_topology.csv", [{"source_unit": "qc-00", "break_free": True}]
    )

    atlas_path = _write_json(root / "multitrait_atlas.json", atlas)
    paths["resource/multitrait_atlas"] = atlas_path
    for role, frame in (
        ("source/clean_traits", clean),
        ("source/full_traits", full),
        ("source/full_image_traits", full_image),
        ("source/analysis_primary_table", primary),
        ("source/analysis_sensitivity_table", sensitivity),
    ):
        path = root / f"{role.replace('/', '__')}.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        paths[role] = path

    wt_paths, analysis_summary = _wt_secondary_authorities(root)
    paths.update(wt_paths)
    analysis_summary["output_table_sha256"].update(
        {
            "primary_tests": sha256_file(paths["source/analysis_primary_table"]),
            "sensitivity_tests": sha256_file(
                paths["source/analysis_sensitivity_table"]
            ),
        }
    )
    analysis_summary.pop("analysis_identity_sha256", None)
    analysis_summary["analysis_identity_sha256"] = sha256_json(analysis_summary)
    paths["receipt/analysis"] = _write_json(
        root / "receipt__analysis.json", analysis_summary
    )

    paths["resource/workflow_stages"] = _write_csv(
        root / "workflow_stages.csv", [{"stage_order": 1, "stage_name": "input"}]
    )
    current_rows = [{"source_unit": f"app-{index:03d}", "wall_seconds": 1.0} for index in range(283)]
    baseline_rows = [{"source_unit": f"app-{index:03d}", "wall_seconds": 2.0} for index in range(283)]
    ordered_identity = sha256_json([row["source_unit"] for row in current_rows])
    paths["resource/runtime_summary"] = _write_json(
        root / "runtime_summary.json",
        {
            "status": "completed",
            "source_unit_ordered_set_identity_sha256": ordered_identity,
            "blind_images_used": 0,
        },
    )
    paths["resource/runtime_per_image"] = _write_csv(root / "runtime_per_image.csv", current_rows)
    paths["source/baseline_runtime_per_image"] = _write_csv(root / "baseline_runtime_per_image.csv", baseline_rows)
    paths["source/runtime_latency_comparison"] = _write_json(root / "latency_comparison.json", {"status": "completed", "blind_images_used": 0})
    paths["source/runtime_production_comparison"] = _write_json(root / "production_comparison.json", {"status": "completed", "blind_images_used": 0})
    same_hardware = {
        "schema_version": "PHAxis-same-hardware-benchmark-receipt-1.0",
        "status": "passed",
        "images": 283,
        "measurement_scope": "raw_image_to_final_traits_and_profiles_direct",
        "source_unit_ordered_set_identity_sha256": ordered_identity,
        "runs": [
            {
                "role": role,
                "source_unit_ordered_set_identity_sha256": ordered_identity,
                "fresh_direct_run": True,
                "resume_or_cache_used": False,
                "full_workflow_io_included": True,
            }
            for role in (
                "phaxis_production",
                "phaxis_sequential",
                "frozen_v1_production",
                "frozen_v1_sequential",
            )
        ],
        "same_ordered_exact283_sources": True,
        "same_hardware_uuid_and_driver": True,
        "same_io_and_full_workflow_scope": True,
        "fresh_no_cache": True,
        "historical_98_47_min_component_receipt_used": False,
        "forward_only_runtime_used": False,
        "blind_images_used": 0,
        "root_cap_region_statistics_included": False,
    }
    same_hardware["receipt_identity_sha256"] = sha256_json(same_hardware)
    paths["source/benchmark_same_hardware"] = _write_json(
        root / "same_hardware.json", same_hardware
    )
    inventory_roles = [
        "same_hardware_receipt",
        "phaxis_production_summary",
        "v1_production_summary",
        "phaxis_sequential_summary",
        "v1_sequential_summary",
        "production_comparison_receipt",
        "sequential_comparison_receipt",
        "per_image_latency_csv",
        "per_image_latency_csv",
        *(["gpu_telemetry"] * 4),
        *(["hardware_preflight"] * 4),
    ]
    paths["source/benchmark_artifact_inventory"] = _write_csv(
        root / "benchmark_inventory.csv",
        [
            {
                "artifact_role": role,
                "package_path": f"model/benchmark/item-{index:02d}.json",
                "sha256": sha256_json({"inventory": index}),
            }
            for index, role in enumerate(inventory_roles)
        ],
    )
    assert set(paths) == required
    return paths


def _materialize(root: Path, source_paths: dict[str, Path]):
    return materialize_supplementary_table_data_bundle(
        output=root,
        status=FINAL_STATUS,
        source_paths=source_paths,
        source_identities={},
        figure_input_manifest_sha256=sha256_json("figure-inputs"),
        figure_input_assembly_identity_sha256=sha256_json("assembly"),
        model_contract_proposal_identity_sha256=sha256_json("proposal"),
    )


def test_materializes_exact_ten_hash_closed_denominator_bound_resources(tmp_path: Path) -> None:
    sources = source_fixture(tmp_path / "sources")
    result = _materialize(tmp_path / "bundle", sources)
    assert result["ordered_item_count"] == 10
    assert list(result["items"]) == [record["stem"] for record in TABLE_SPECS]
    assert [record["directory"] for record in result["items"].values()] == list(
        TABLE_DIRECTORIES
    )
    assert all(
        record["item_receipt"] == f"{directory}/item_receipt.json"
        for directory, record in zip(
            TABLE_DIRECTORIES, result["items"].values(), strict=True
        )
    )
    s2 = result["items"]["Table_S02_HumanCurated443_manifest"][
        "denominator_contract"
    ]
    assert (s2["observed_rows"], s2["train_rows"], s2["validation_rows"]) == (
        443,
        399,
        44,
    )
    assert s2["family_key_overlap"] == 0
    s3 = result["items"]["Table_S03_five_member_model_identities"][
        "denominator_contract"
    ]
    assert s3["observed_rows"] == 5
    assert s3["fixed_seed_order"] == list(range(2026082801, 2026082806))
    s4 = result["items"]["Table_S04_trait_dictionary_and_export_schema"][
        "denominator_contract"
    ]
    assert (s4["observed_trait_rows"], s4["observed_export_schema_columns"]) == (
        32,
        82,
    )
    s6 = result["items"]["Table_S06_QCdevelopment44_per_image"][
        "denominator_contract"
    ]
    assert (s6["observed_source_images"], s6["per_image_rows"]) == (44, 88)
    s9 = result["items"]["Table_S09_complete_multitrait_atlas"]["denominator_contract"]
    assert (
        "block/day-stratified WT temperature secondary analysis"
        in result["items"]["Table_S09_complete_multitrait_atlas"]["title"]
    )
    assert s9["block_A_observed_slots"] == 256
    assert s9["block_B_observed_slots"] == 192
    assert s9["block_C_observed_rows"] == 544
    assert s9["D15_fixed_effect_rows"] == 15
    assert s9["D15_fixed_effect_family_changed_by_WT_secondary"] is False
    assert s9["block_D_typed_block"] == "wt_gate_flow"
    assert s9["block_E_typed_block"] == "wt_experiment_contrasts"
    assert s9["block_F_typed_block"] == "wt_same_day_meta"
    assert s9["block_D_observed_rows"] == 50
    assert s9["block_E_observed_rows"] == 50
    assert s9["block_F_observed_rows"] == 20
    assert s9["block_G_typed_block"] == "h11_raw_median_companion"
    assert s9["block_G_expected_rows"] == s9["block_G_observed_rows"] == 6
    assert s9["block_G_endpoint"] == "local_median_hair_length_um_1_4mm"
    assert s9["block_G_fixed_15_primary_ratio_family_changed"] is False
    assert s9["block_G_source_multitrait_atlas_sha256"] == sha256_file(
        sources["resource/multitrait_atlas"]
    )
    assert s9["block_G_source_primary_analysis_sha256"] == sha256_file(
        sources["source/analysis_primary_table"]
    )
    assert s9["block_G_source_sensitivity_analysis_sha256"] == sha256_file(
        sources["source/analysis_sensitivity_table"]
    )
    assert s9["wt_secondary_estimable_within_day_meta_rows"] == 10
    assert s9["wt_secondary_typed_not_estimable_meta_rows"] == 10
    assert s9["wt_secondary_cross_day_pooling_performed"] is False
    assert s9["wt_secondary_unknown_day_meta_analysis_performed"] is False
    assert s9["wt_secondary_clean_full_pooling_performed"] is False
    s9_root = tmp_path / "bundle" / "S09"
    for name, typed_block in (
        ("block_D_WT_inventory_and_gate_flow.csv", "wt_gate_flow"),
        (
            "block_E_WT_within_experiment_temperature_contrasts.csv",
            "wt_experiment_contrasts",
        ),
        ("block_F_WT_within_day_REML_Hartung_Knapp.csv", "wt_same_day_meta"),
    ):
        with (s9_root / name).open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows and {row["typed_block"] for row in rows} == {typed_block}
    h11_path = s9_root / "block_G_H11_raw_median_companion.csv"
    with h11_path.open(encoding="utf-8") as handle:
        h11_rows = list(csv.DictReader(handle))
    assert len(h11_rows) == 6
    assert {
        (row["cohort"], row["effect_key"]) for row in h11_rows
    } == {
        (cohort, effect)
        for cohort in ("primary_clean261", "sensitivity_full283")
        for effect in ("OE_vs_EV", "30C_vs_22C", "interaction")
    }
    for row in h11_rows:
        source_role = (
            "source/analysis_primary_table"
            if row["cohort"] == "primary_clean261"
            else "source/analysis_sensitivity_table"
        )
        assert row["source_analysis_table_sha256"] == sha256_file(
            sources[source_role]
        )
        assert row["source_multitrait_atlas_sha256"] == sha256_file(
            sources["resource/multitrait_atlas"]
        )
        assert row["source_analysis_receipt_sha256"] == sha256_file(
            sources["receipt/analysis"]
        )
        assert row["raw_effect_bootstrap_replicates"] == "5000"
        assert row["raw_effect_bootstrap_seed"] == str(
            s9["block_G_raw_effect_bootstrap_seed"]
        )
    s10 = result["items"]["Table_S10_reproducibility_benchmark_ledger"][
        "denominator_contract"
    ]
    assert (s10["phaxis_sequential_rows"], s10["frozen_v1_sequential_rows"]) == (
        283,
        283,
    )
    assert result["blind_images_used"] == 0
    assert result["root_cap_region_statistics_included"] is False
    validated = validate_supplementary_table_data_bundle(
        tmp_path / "bundle" / BUNDLE_RECEIPT, require_final=True
    )
    assert validated["bundle_identity_sha256"] == result["bundle_identity_sha256"]


def test_create_only_and_file_tamper_fail_closed(tmp_path: Path) -> None:
    sources = source_fixture(tmp_path / "sources")
    output = tmp_path / "bundle"
    result = _materialize(output, sources)
    with pytest.raises(SupplementaryTableError, match="overwrite"):
        _materialize(output, sources)
    relative = next(iter(result["bundle_file_sha256"]))
    (output / relative).write_bytes(b"tampered")
    with pytest.raises(SupplementaryTableError, match="tamper|hash"):
        validate_supplementary_table_data_bundle(output / BUNDLE_RECEIPT, require_final=True)


def test_s9_missing_slot_and_source_role_drift_are_rejected(tmp_path: Path) -> None:
    sources = source_fixture(tmp_path / "sources")
    broken = json.loads(sources["resource/multitrait_atlas"].read_text(encoding="utf-8"))
    first = broken["descriptors"][0]["cohorts"][broken["cohort_order"][0]]
    first["condition_summaries"].pop(broken["condition_order"][0])
    broken.pop("atlas_identity_sha256", None)
    broken["atlas_identity_sha256"] = sha256_json(broken)
    _write_json(sources["resource/multitrait_atlas"], broken)
    with pytest.raises(SupplementaryTableError, match="condition slot"):
        _materialize(tmp_path / "broken-slots", sources)

    repaired = source_fixture(tmp_path / "sources-2")
    repaired.pop("source/dataset_manifest")
    with pytest.raises(SupplementaryTableError, match="source-role set"):
        _materialize(tmp_path / "broken-roles", repaired)


def test_s9_h11_block_rejects_raw_atlas_drift_after_receipt_rebinding(
    tmp_path: Path,
) -> None:
    sources = source_fixture(tmp_path / "sources")
    primary_path = sources["source/analysis_primary_table"]
    primary = pd.read_csv(primary_path)
    target = primary.index[
        primary["endpoint"].eq("local_median_hair_length_um_1_4mm")
        & primary["effect"].eq("construct_OE_minus_EV")
    ]
    assert len(target) == 1
    primary.loc[target[0], "raw_effect_estimate"] += 1.0
    primary.to_csv(primary_path, index=False, lineterminator="\n")
    receipt_path = sources["receipt/analysis"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output_table_sha256"]["primary_tests"] = sha256_file(primary_path)
    receipt.pop("analysis_identity_sha256", None)
    receipt["analysis_identity_sha256"] = sha256_json(receipt)
    _write_json(receipt_path, receipt)
    with pytest.raises(
        SupplementaryTableError,
        match="raw analysis and atlas companion differ",
    ):
        _materialize(tmp_path / "broken-h11-value", sources)


@pytest.mark.parametrize(
    ("endpoint", "wrong_estimand", "message"),
    (
        (
            "local_median_hair_length_um_1_4mm",
            "OLS_raw_mean_difference",
            "H11 median/bootstrap/5000/seed contract changed",
        ),
        (
            "local_hair_count_1_4mm",
            "factorial_four_cell_median_contrast",
            "non-H11 mean/HC3/0/blank-seed contract changed",
        ),
    ),
)
def test_s9_raw_companion_semantics_fail_even_when_raw_and_atlas_agree(
    tmp_path: Path,
    endpoint: str,
    wrong_estimand: str,
    message: str,
) -> None:
    sources = source_fixture(tmp_path / f"sources-{endpoint}")
    primary_path = sources["source/analysis_primary_table"]
    primary = pd.read_csv(primary_path)
    target = primary.index[
        primary["endpoint"].eq(endpoint)
        & primary["effect"].eq("construct_OE_minus_EV")
    ]
    assert len(target) == 1
    primary.loc[target[0], "raw_effect_estimand"] = wrong_estimand
    primary.to_csv(primary_path, index=False, lineterminator="\n")

    atlas_path = sources["resource/multitrait_atlas"]
    atlas = json.loads(atlas_path.read_text(encoding="utf-8"))
    descriptor = next(row for row in atlas["descriptors"] if row["field"] == endpoint)
    descriptor["cohorts"]["primary_clean261"]["effects"]["OE_vs_EV"][
        "raw_effect_estimand"
    ] = wrong_estimand
    atlas.pop("atlas_identity_sha256", None)
    atlas["atlas_identity_sha256"] = sha256_json(atlas)
    _write_json(atlas_path, atlas)

    receipt_path = sources["receipt/analysis"]
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output_table_sha256"]["primary_tests"] = sha256_file(primary_path)
    receipt.pop("analysis_identity_sha256", None)
    receipt["analysis_identity_sha256"] = sha256_json(receipt)
    _write_json(receipt_path, receipt)
    with pytest.raises(SupplementaryTableError, match=message):
        _materialize(tmp_path / f"broken-semantics-{endpoint}", sources)


def test_s9_wt_secondary_rejects_cross_day_or_d15_family_drift(tmp_path: Path) -> None:
    sources = source_fixture(tmp_path / "sources")
    meta_path = sources["source/wt_within_day_meta_analysis"]
    meta = pd.read_csv(meta_path)
    meta.loc[0, "cross_day_pooling_performed"] = True
    meta.to_csv(meta_path, index=False, lineterminator="\n")
    analysis_path = sources["receipt/analysis"]
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["output_table_sha256"]["wt_within_day_meta_analysis"] = sha256_file(
        meta_path
    )
    analysis.pop("analysis_identity_sha256", None)
    analysis["analysis_identity_sha256"] = sha256_json(analysis)
    _write_json(analysis_path, analysis)
    with pytest.raises(
        SupplementaryTableError, match="cross-day pooling|crossed a day boundary"
    ):
        _materialize(tmp_path / "cross-day", sources)

    repaired = source_fixture(tmp_path / "sources-2")
    analysis_path = repaired["receipt/analysis"]
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["D15_fixed_effect_rows"] = 16
    analysis.pop("analysis_identity_sha256", None)
    analysis["analysis_identity_sha256"] = sha256_json(analysis)
    _write_json(analysis_path, analysis)
    with pytest.raises(SupplementaryTableError, match="D15 15-effect"):
        _materialize(tmp_path / "d15-drift", repaired)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        (
            "p_value_hartung_knapp_BH_FDR",
            0.4,
            "non-estimable value is populated|pooled estimate/statistic",
        ),
        (
            "hartung_knapp_scale",
            1.0,
            "non-estimable value is populated|pooled estimate/statistic",
        ),
        (
            "reject_hartung_knapp_BH_FDR_0p05",
            True,
            "non-estimable row promoted|positive BH decision",
        ),
        (
            "k_eligible_experiments",
            1.5,
            "non-negative integer|count is not an integer",
        ),
    ),
)
def test_s9_wt_not_estimable_meta_rows_cannot_carry_partial_inference(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    sources = source_fixture(tmp_path / f"sources-{field}")
    meta_path = sources["source/wt_within_day_meta_analysis"]
    meta = pd.read_csv(meta_path)
    not_estimable = meta.index[meta["analysis_status"].eq("not_estimable")].tolist()
    assert not_estimable
    row_index = not_estimable[0]
    if field == "k_eligible_experiments":
        meta[field] = meta[field].astype(float)
    meta.loc[row_index, field] = value
    meta.to_csv(meta_path, index=False, lineterminator="\n")
    analysis_path = sources["receipt/analysis"]
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["output_table_sha256"]["wt_within_day_meta_analysis"] = sha256_file(
        meta_path
    )
    analysis.pop("analysis_identity_sha256", None)
    analysis["analysis_identity_sha256"] = sha256_json(analysis)
    _write_json(analysis_path, analysis)
    with pytest.raises(SupplementaryTableError, match=message):
        _materialize(tmp_path / f"broken-{field}", sources)


def test_s3_config_and_checkpoint_identity_are_mandatory(tmp_path: Path) -> None:
    sources = source_fixture(tmp_path / "sources")
    candidate = json.loads(sources["source/train399_candidate"].read_text(encoding="utf-8"))
    candidate["identity_payload"]["training_lock"]["config_sha256"] = None
    _write_json(sources["source/train399_candidate"], candidate)
    with pytest.raises(SupplementaryTableError, match="config SHA-256"):
        _materialize(tmp_path / "broken-candidate", sources)


def test_s6_requires_complete_exact44_by_two_comparator_cartesian_pairing(
    tmp_path: Path,
) -> None:
    sources = source_fixture(tmp_path / "sources")
    path = sources["resource/development_per_image"]
    frame = pd.read_csv(path)
    # Preserve 44 global source units and both global comparator labels while
    # breaking one source's paired slot.  A loose set/count check would pass.
    first_source = frame.iloc[0]["source_unit"]
    first_rows = frame.index[frame["source_unit"] == first_source].tolist()
    frame = frame.drop(first_rows[0])
    duplicate = frame.iloc[[0]].copy()
    frame = pd.concat([frame, duplicate], ignore_index=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    with pytest.raises(SupplementaryTableError, match="exact44 x paired-comparator"):
        _materialize(tmp_path / "broken-pairing", sources)
