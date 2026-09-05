from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from phaxis.axial_profile_analysis import analyze_distal_axis_profiles
from phaxis.axial_profiles import COHORT_PROFILE_BINDING_SCHEMA
from phaxis.contracts import ContractError
from phaxis.io import sha256_file, sha256_json


PROPOSAL_BINDING = {
    "model_contract_proposal_sha256": "b" * 64,
    "model_contract_proposal_identity_sha256": "c" * 64,
}
PUBLIC_IDENTITY = {
    "model_bundle_id": "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC",
    "root_expert_id": "PHAxis-root-provider-SYNTHETIC",
}


def _write_profiles(root: Path, *, sensitivity: bool = False) -> None:
    root.mkdir(parents=True)
    rows = []
    clean_tasks = [f"C{index:03d}" for index in range(261)]
    overlap_tasks = [f"O{index:03d}" for index in range(22)]
    task_ids = clean_tasks + (overlap_tasks if sensitivity else [])
    conditions = (
        "RHD6_EV_22C",
        "RHD6_EV_30C",
        "RHD6_OE_22C",
        "RHD6_OE_30C",
    )
    for index, task_id in enumerate(task_ids):
        condition_index = index % len(conditions)
        count = index % 5
        rows.append(
            {
                "task_id": task_id,
                "source_image_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
                "experiment_key": "D15_8d",
                "study_role": "rhd6_factorial_8d_primary",
                "condition_code": conditions[condition_index],
                "formal_statistics_eligible": "True",
                "bin_eligible": "True",
                "bin_index": "0",
                "bin_start_um": "0",
                "bin_end_um": "1000",
                "attached_identity_count": str(count),
                "attached_identity_density_per_mm": str(count),
                "endpoint_complete_length_count": str(min(count, 1)),
                "conditional_median_hair_length_um": (
                    str(100 + condition_index * 10 + index % 7) if count else ""
                ),
                "root_cap_region_output": "False",
                "blind_images_used": "0",
            }
        )
    table = root / "distal_axis_profiles.csv"
    with table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cohort_name = "sensitivity_full283" if sensitivity else "primary_clean261"
    cohort_role = (
        "overlap_contaminated_sensitivity"
        if sensitivity
        else "primary_SHA_disjoint"
    )
    summary = {
        "schema_version": "PHAxis-distal-axis-profile-export-1.0.0",
        "status": "completed",
        "tasks": len(task_ids),
        "profiles_csv_sha256": sha256_file(table),
        "locked_1_4mm_trait_crosscheck_mismatches": 0,
        "cohort_binding": {
            "schema_version": COHORT_PROFILE_BINDING_SCHEMA,
            "cohort_name": cohort_name,
            "cohort_role": cohort_role,
            "cohort_tasks": len(task_ids),
            "cohort_build_summary_sha256": "d" * 64,
            "cohort_build_identity_sha256": "e" * 64,
            "cohort_membership_csv_sha256": "f" * 64,
            "cohort_task_membership_sha256": sha256_json(sorted(task_ids)),
            "cohort_source_image_membership_sha256": sha256_json(
                sorted(row["source_image_sha256"] for row in rows)
            ),
            "blind_images_used": 0,
        },
        "blind_images_used": 0,
        **PROPOSAL_BINDING,
        **PUBLIC_IDENTITY,
    }
    summary["export_identity_sha256"] = sha256_json(summary)
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _contract(path: Path, *, blind: int = 0) -> None:
    payload = {
        "schema_version": "PHAxis-distal-axis-profile-analysis-contract-1.0.0",
        "primary_cohort": "primary_clean261",
        "sensitivity_cohort": "sensitivity_full283",
        "individual_hairs_treated_as_independent_replicates": False,
        "primary_scope": {
            "experiment_key": "D15_8d",
            "study_role": "rhd6_factorial_8d_primary",
            "condition_codes": [
                "RHD6_EV_22C",
                "RHD6_EV_30C",
                "RHD6_OE_22C",
                "RHD6_OE_30C",
            ],
        },
        "uncertainty": {
            "replicates": 100,
            "random_seed": 7,
            "hypothesis_tests_performed": False,
        },
        "reporting": {
            "hypothesis_tests_performed": False,
            "blind_images_used": blind,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_source_unit_summary_is_deterministic(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    sensitivity = tmp_path / "sensitivity"
    _write_profiles(primary)
    _write_profiles(sensitivity, sensitivity=True)
    contract = tmp_path / "contract.json"
    _contract(contract)
    first = analyze_distal_axis_profiles(
        primary_profiles=primary,
        sensitivity_profiles=sensitivity,
        contract_json=contract,
        output=tmp_path / "out1",
        model_contract_proposal=PROPOSAL_BINDING,
        model_contract_public_identity=PUBLIC_IDENTITY,
    )
    second = analyze_distal_axis_profiles(
        primary_profiles=primary,
        sensitivity_profiles=sensitivity,
        contract_json=contract,
        output=tmp_path / "out2",
        model_contract_proposal=PROPOSAL_BINDING,
        model_contract_public_identity=PUBLIC_IDENTITY,
    )
    assert first["rows"] == 8
    assert first["output_table_sha256"] == second["output_table_sha256"]
    rows = list(
        csv.DictReader(
            (tmp_path / "out1" / "distal_axis_profile_group_summaries.csv").open(
                encoding="utf-8", newline=""
            )
        )
    )
    assert rows[0]["unit_of_analysis"] == "one_source_image_root_unit"
    assert rows[0]["hypothesis_tests_performed"] == "False"


def test_profile_hash_drift_fails_closed(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    sensitivity = tmp_path / "sensitivity"
    _write_profiles(primary)
    _write_profiles(sensitivity, sensitivity=True)
    contract = tmp_path / "contract.json"
    _contract(contract)
    with (primary / "distal_axis_profiles.csv").open("a", encoding="utf-8") as handle:
        handle.write("tamper\n")
    with pytest.raises(ContractError):
        analyze_distal_axis_profiles(
            primary_profiles=primary,
            sensitivity_profiles=sensitivity,
            contract_json=contract,
            output=tmp_path / "out",
            model_contract_proposal=PROPOSAL_BINDING,
            model_contract_public_identity=PUBLIC_IDENTITY,
        )


def test_profile_analysis_rejects_relabelled_single_export(tmp_path: Path) -> None:
    shared = tmp_path / "shared_full283"
    _write_profiles(shared)
    contract = tmp_path / "contract.json"
    _contract(contract)
    with pytest.raises(
        ContractError,
        match="distinct cohort-specific exports",
    ):
        analyze_distal_axis_profiles(
            primary_profiles=shared,
            sensitivity_profiles=shared,
            contract_json=contract,
            output=tmp_path / "out",
            model_contract_proposal=PROPOSAL_BINDING,
            model_contract_public_identity=PUBLIC_IDENTITY,
        )


def test_profile_analysis_rejects_distinct_directory_copy_of_full_export(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    sensitivity = tmp_path / "sensitivity"
    _write_profiles(sensitivity, sensitivity=True)
    shutil.copytree(sensitivity, primary)
    contract = tmp_path / "contract.json"
    _contract(contract)
    with pytest.raises(ContractError, match="cohort binding is mislabelled"):
        analyze_distal_axis_profiles(
            primary_profiles=primary,
            sensitivity_profiles=sensitivity,
            contract_json=contract,
            output=tmp_path / "out",
            model_contract_proposal=PROPOSAL_BINDING,
            model_contract_public_identity=PUBLIC_IDENTITY,
        )


def test_blind_contract_fails_closed(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    sensitivity = tmp_path / "sensitivity"
    _write_profiles(primary)
    _write_profiles(sensitivity, sensitivity=True)
    contract = tmp_path / "contract.json"
    _contract(contract, blind=1)
    with pytest.raises(ContractError):
        analyze_distal_axis_profiles(
            primary_profiles=primary,
            sensitivity_profiles=sensitivity,
            contract_json=contract,
            output=tmp_path / "out",
            model_contract_proposal=PROPOSAL_BINDING,
            model_contract_public_identity=PUBLIC_IDENTITY,
        )


def test_profile_analysis_rejects_cross_version_sensitivity_profile(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    sensitivity = tmp_path / "sensitivity"
    _write_profiles(primary)
    _write_profiles(sensitivity, sensitivity=True)
    summary_path = sensitivity / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["root_expert_id"] = "PHAxis-root-provider-DIFFERENT"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    contract = tmp_path / "contract.json"
    _contract(contract)
    with pytest.raises(ContractError):
        analyze_distal_axis_profiles(
            primary_profiles=primary,
            sensitivity_profiles=sensitivity,
            contract_json=contract,
            output=tmp_path / "out",
            model_contract_proposal=PROPOSAL_BINDING,
            model_contract_public_identity=PUBLIC_IDENTITY,
        )
