from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from phaxis.axial_profiles import (
    COHORT_PROFILE_BINDING_SCHEMA,
    COHORT_PROFILE_BUNDLE_SCHEMA,
    export_cohort_distal_axis_profiles,
)
from phaxis.axial_profile_analysis import analyze_distal_axis_profiles
from phaxis.contracts import ContractError
from phaxis.io import sha256_file, sha256_json


PROPOSAL_BINDING = {
    "model_contract_proposal_sha256": "1" * 64,
    "model_contract_proposal_identity_sha256": "2" * 64,
}
PUBLIC_IDENTITY = {
    "model_bundle_id": "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC",
    "root_expert_id": "PHAxis-root-provider-SYNTHETIC",
}
COUNTS = {
    "human_curated443": 443,
    "biological_full": 283,
    "human_curated_overlap": 22,
    "biological_clean": 261,
}


def _write_csv(
    path: Path, rows: list[dict[str, object]], *, fields: list[str] | None = None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _task_rows(task_ids: list[str]) -> dict[str, list[dict[str, object]]]:
    traits: list[dict[str, object]] = []
    roots: list[dict[str, object]] = []
    hairs: list[dict[str, object]] = []
    image_traits: list[dict[str, object]] = []
    for index, task_id in enumerate(task_ids):
        source_sha = hashlib.sha256(task_id.encode()).hexdigest()
        condition = (
            "RHD6_EV_22C",
            "RHD6_EV_30C",
            "RHD6_OE_22C",
            "RHD6_OE_30C",
        )[index % 4]
        traits.append(
            {
                "task_id": task_id,
                "source_image_sha256": source_sha,
                "experiment_key": "D15_8d",
                "condition_code": condition,
                "study_role": "rhd6_factorial_8d_primary",
                "developmental_day": "8",
                "genotype_or_construct": "RHD6",
                "temperature_c": "22",
                "formal_statistics_eligible": "True",
                "visible_root_axis_length_um": "2000",
                "attachment_axis_valid_fraction": "1.0",
            }
        )
        roots.append(
            {
                "task_id": task_id,
                "source_image_sha256": source_sha,
                "visible_root_axis_length_um": "2000",
            }
        )
        hairs.append(
            {
                "task_id": task_id,
                "source_image_sha256": source_sha,
                "hair_id": "H1",
                "attachment_valid_within_40um": "True",
                "attachment_distance_from_distal_point_um": "500",
                "complete_length_measurement_eligible": "True",
                "length_um": "100",
            }
        )
        image_traits.append(
            {
                "task_id": task_id,
                "source_image_sha256": source_sha,
                "visible_hair_count": "1",
            }
        )
    return {
        "traits": traits,
        "detailed_root_statistics": roots,
        "hair_instances": hairs,
        "image_traits": image_traits,
    }


def _reseal_cohort_receipts(root: Path) -> None:
    membership_sha = sha256_file(root / "cohort_membership.csv")
    lock_path = root / "analysis_contract_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["output_table_sha256"]["cohort_membership"] = membership_sha
    for cohort_name in ("primary_clean261", "sensitivity_full283"):
        for table_name in (
            "traits",
            "detailed_root_statistics",
            "hair_instances",
            "image_traits",
        ):
            lock["output_table_sha256"][cohort_name][table_name] = sha256_file(
                root / cohort_name / f"{table_name}.csv"
            )
    lock.pop("cohort_lock_identity_sha256", None)
    lock["cohort_lock_identity_sha256"] = sha256_json(lock)
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["output_sha256"]["cohort_membership"] = membership_sha
    summary["output_sha256"]["analysis_contract_lock"] = sha256_file(lock_path)
    for cohort_name in ("primary_clean261", "sensitivity_full283"):
        summary["output_sha256"][cohort_name] = dict(
            lock["output_table_sha256"][cohort_name]
        )
    summary.pop("cohort_build_identity_sha256", None)
    summary["cohort_build_identity_sha256"] = sha256_json(summary)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "cohorts"
    clean_ids = [f"C{index:03d}" for index in range(261)]
    overlap_ids = [f"O{index:03d}" for index in range(22)]
    full_ids = clean_ids + overlap_ids
    tables = {
        "primary_clean261": _task_rows(clean_ids),
        "sensitivity_full283": _task_rows(full_ids),
    }
    table_hashes: dict[str, dict[str, str]] = {}
    for cohort_name, cohort_tables in tables.items():
        table_hashes[cohort_name] = {}
        for table_name, rows in cohort_tables.items():
            path = root / cohort_name / f"{table_name}.csv"
            _write_csv(path, rows)
            table_hashes[cohort_name][table_name] = sha256_file(path)

    membership: list[dict[str, object]] = []
    for task_id in full_ids:
        human_overlap = task_id in overlap_ids
        membership.append(
            {
                "task_id": task_id,
                "source_image_sha256": hashlib.sha256(task_id.encode()).hexdigest(),
                "primary_clean_sha_disjoint_include": not human_overlap,
                "sensitivity_full_include": True,
                "recomputed_human443_overlap": human_overlap,
                "overlap_human_task_ids": "RHAUD-X" if human_overlap else "",
                "overlap_human_splits": "train" if human_overlap else "",
                "formal_statistics_eligible": True,
            }
        )
    _write_csv(root / "cohort_membership.csv", membership)
    _write_csv(
        root / "cohort_condition_counts.csv",
        [{"cohort": "primary_clean261", "units": 261}],
    )
    _write_csv(
        root / "acquisition_batch_condition_audit.csv",
        [{"scope": "full_biological_cohort", "units": 283}],
    )

    traits_summary = tmp_path / "traits_summary.json"
    traits_summary.write_text(
        json.dumps({**PROPOSAL_BINDING, **PUBLIC_IDENTITY}), encoding="utf-8"
    )
    output_table_hashes = {
        "cohort_membership": sha256_file(root / "cohort_membership.csv"),
        "cohort_condition_counts": sha256_file(root / "cohort_condition_counts.csv"),
        "acquisition_batch_condition_audit": sha256_file(
            root / "acquisition_batch_condition_audit.csv"
        ),
        **table_hashes,
    }
    lock = {
        "schema_version": "PHAxis-biological-cohort-lock-1.0",
        "status": "postresult_software_transition_provenance_lock",
        "blind_images_used": 0,
        "canonical_annotations_read": False,
        "cohort_counts": dict(COUNTS),
        "output_table_sha256": output_table_hashes,
        **PROPOSAL_BINDING,
        **PUBLIC_IDENTITY,
    }
    lock["cohort_lock_identity_sha256"] = sha256_json(lock)
    (root / "analysis_contract_lock.json").write_text(
        json.dumps(lock), encoding="utf-8"
    )
    summary = {
        "schema_version": "PHAxis-biological-cohorts-1.0",
        "status": "completed_without_fitting_biological_effect_models",
        "cohort_directories": {
            "primary": "primary_clean261",
            "sensitivity": "sensitivity_full283",
        },
        "counts": dict(COUNTS),
        "input_sha256": {"trait_export_summary": sha256_file(traits_summary)},
        "output_sha256": {
            "analysis_contract_lock": sha256_file(
                root / "analysis_contract_lock.json"
            ),
            **output_table_hashes,
        },
        "root_cap_region_statistics_included": False,
        "canonical_annotations_read": False,
        "blind_images_used": 0,
        **PROPOSAL_BINDING,
        **PUBLIC_IDENTITY,
    }
    summary["cohort_build_identity_sha256"] = sha256_json(summary)
    (root / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    contract = tmp_path / "profile_contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": "PHAxis-distal-axis-profile-contract-1.0.0",
                "bins_um": [[0, 1000], [1000, 2000]],
                "root_cap_region_output": False,
                "stageb_two_point_vector_used_as_length": False,
                "blind_images_used": 0,
            }
        ),
        encoding="utf-8",
    )
    return root, traits_summary, contract


def _export(root: Path, traits_summary: Path, contract: Path, output: Path) -> dict:
    return export_cohort_distal_axis_profiles(
        cohorts_root=root,
        contract_json=contract,
        output=output,
        model_contract_proposal=PROPOSAL_BINDING,
        model_contract_public_identity=PUBLIC_IDENTITY,
        traits_summary_json=traits_summary,
    )


def test_exact_clean261_and_full283_exports_are_distinct_and_sealed(
    tmp_path: Path,
) -> None:
    root, traits_summary, contract = _fixture(tmp_path)
    output = tmp_path / "profiles"
    bundle = _export(root, traits_summary, contract, output)
    assert bundle["schema_version"] == COHORT_PROFILE_BUNDLE_SCHEMA
    assert bundle["primary_sensitivity_task_overlap"] == 261
    assert bundle["sensitivity_only_human443_overlap_tasks"] == 22
    identity = bundle["cohort_profile_bundle_identity_sha256"]
    unsigned = dict(bundle)
    unsigned.pop("cohort_profile_bundle_identity_sha256")
    assert identity == sha256_json(unsigned)

    primary = json.loads(
        (output / "primary_clean261" / "summary.json").read_text(encoding="utf-8")
    )
    sensitivity = json.loads(
        (output / "sensitivity_full283" / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    assert primary["tasks"] == primary["cohort_binding"]["cohort_tasks"] == 261
    assert (
        sensitivity["tasks"]
        == sensitivity["cohort_binding"]["cohort_tasks"]
        == 283
    )
    assert primary["cohort_binding"]["schema_version"] == COHORT_PROFILE_BINDING_SCHEMA
    assert primary["export_identity_sha256"] != sensitivity["export_identity_sha256"]
    assert primary["blind_images_used"] == sensitivity["blind_images_used"] == 0


def test_cohort_exports_are_accepted_by_strict_dual_cohort_analysis(
    tmp_path: Path,
) -> None:
    root, traits_summary, contract = _fixture(tmp_path)
    output = tmp_path / "profiles"
    _export(root, traits_summary, contract, output)
    analysis_contract = tmp_path / "analysis_contract.json"
    analysis_contract.write_text(
        json.dumps(
            {
                "schema_version": (
                    "PHAxis-distal-axis-profile-analysis-contract-1.0.0"
                ),
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
                    "replicates": 25,
                    "random_seed": 7,
                    "hypothesis_tests_performed": False,
                },
                "reporting": {
                    "hypothesis_tests_performed": False,
                    "blind_images_used": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    result = analyze_distal_axis_profiles(
        primary_profiles=output / "primary_clean261",
        sensitivity_profiles=output / "sensitivity_full283",
        contract_json=analysis_contract,
        output=tmp_path / "analysis",
        model_contract_proposal=PROPOSAL_BINDING,
        model_contract_public_identity=PUBLIC_IDENTITY,
    )
    assert result["rows"] == 16
    assert result["blind_images_used"] == 0


def test_cohort_profile_export_rejects_table_hash_tamper(tmp_path: Path) -> None:
    root, traits_summary, contract = _fixture(tmp_path)
    with (root / "primary_clean261" / "traits.csv").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("tamper\n")
    with pytest.raises(ContractError, match="hash mismatch"):
        _export(root, traits_summary, contract, tmp_path / "profiles")


def test_cohort_profile_export_rejects_resealed_overlap_semantic_drift(
    tmp_path: Path,
) -> None:
    root, traits_summary, contract = _fixture(tmp_path)
    membership_path = root / "cohort_membership.csv"
    with membership_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["primary_clean_sha_disjoint_include"] = "False"
    _write_csv(membership_path, rows)
    _reseal_cohort_receipts(root)
    with pytest.raises(ContractError, match="overlap membership is inconsistent"):
        _export(root, traits_summary, contract, tmp_path / "profiles")


def test_cohort_profile_export_rejects_resealed_full_table_mislabelled_as_clean(
    tmp_path: Path,
) -> None:
    root, traits_summary, contract = _fixture(tmp_path)
    shutil.copyfile(
        root / "sensitivity_full283" / "traits.csv",
        root / "primary_clean261" / "traits.csv",
    )
    _reseal_cohort_receipts(root)
    with pytest.raises(ContractError, match="task membership differs"):
        _export(root, traits_summary, contract, tmp_path / "profiles")


def test_cohort_profile_export_is_create_only(tmp_path: Path) -> None:
    root, traits_summary, contract = _fixture(tmp_path)
    output = tmp_path / "profiles"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite existing output"):
        _export(root, traits_summary, contract, output)
