from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

from phaxis.axial_profiles import export_distal_axis_profiles
from phaxis.fusion import fuse_hybrid_root_with_stageb_hairs
from phaxis.io import atomic_write_json, sha256_json
from phaxis.traits import HAIR_TRAIT_FIELDS, IMAGE_TRAIT_FIELDS, ROOT_TRAIT_FIELDS, export_traits


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROPOSAL_BINDING = {
    "model_contract_proposal_sha256": "b" * 64,
    "model_contract_proposal_identity_sha256": "c" * 64,
}
PUBLIC_IDENTITY = {
    "model_bundle_id": "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC",
    "root_expert_id": "PHAxis-root-provider-SYNTHETIC",
}


def _write_metadata(path: Path, *, um_per_px: float = 1.0) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task_id",
                "image_sha256",
                "um_per_px",
                "experiment_key",
                "condition_code",
                "study_role",
                "developmental_day",
                "genotype_or_construct",
                "temperature_c",
                "qc_disposition",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "T1",
                "image_sha256": "a" * 64,
                "um_per_px": um_per_px,
                "experiment_key": "synthetic",
                "condition_code": "zero-hair" if um_per_px == 1.0 else "scale-audit",
                "study_role": "unit_test",
                "developmental_day": "",
                "genotype_or_construct": "",
                "temperature_c": "",
                "qc_disposition": "eligible",
            }
        )


def _fused_prediction(
    phaxis_case,
    *,
    zero_hairs: bool = False,
    prediction_scale_fail_closed: bool = False,
) -> tuple[dict, Path]:
    root_prediction, hair_detection, artifact_root = phaxis_case
    root_prediction = deepcopy(root_prediction)
    hair_detection = deepcopy(hair_detection)
    root_prediction["detailed_root_statistics"] = {
        **{field: 1.0 for field in ROOT_TRAIT_FIELDS},
        "visible_root_axis_length_um": 5000.0,
        "median_root_width_um": 12.0,
    }
    if prediction_scale_fail_closed:
        root_prediction["scale"] = {"um_per_px": 1.0, "fail_closed": True}
    if zero_hairs:
        hair_detection["detections"] = []
        hair_detection["n"] = 0
    hair_detection.update({**PROPOSAL_BINDING, **PUBLIC_IDENTITY})
    hair_detection.pop("detection_identity_sha256", None)
    hair_detection["detection_identity_sha256"] = sha256_json(hair_detection)
    fused = fuse_hybrid_root_with_stageb_hairs(
        root_prediction,
        hair_detection,
        hybrid_artifact_root=artifact_root,
        model_contract_proposal=PROPOSAL_BINDING,
        model_contract_public_identity=PUBLIC_IDENTITY,
    )
    return fused, artifact_root


def _export(
    phaxis_case,
    tmp_path: Path,
    *,
    zero_hairs: bool = False,
    prediction_scale_fail_closed: bool = False,
    metadata_um_per_px: float = 1.0,
) -> tuple[dict, Path]:
    fused, artifact_root = _fused_prediction(
        phaxis_case,
        zero_hairs=zero_hairs,
        prediction_scale_fail_closed=prediction_scale_fail_closed,
    )
    prediction_root = artifact_root / "predictions"
    atomic_write_json(prediction_root / "T1.json", fused)
    metadata = artifact_root / "metadata.csv"
    _write_metadata(metadata, um_per_px=metadata_um_per_px)
    output = tmp_path / "traits"
    summary = export_traits(
        prediction_root=prediction_root,
        metadata_csv=metadata,
        output=output,
        model_contract_proposal=PROPOSAL_BINDING,
        model_contract_public_identity=PUBLIC_IDENTITY,
    )
    return summary, output


def _row(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return next(csv.DictReader(handle))


def test_zero_hair_batch_exports_header_only_instances_and_profiles(
    phaxis_case, tmp_path: Path
) -> None:
    summary, output = _export(phaxis_case, tmp_path, zero_hairs=True)

    with (output / "hair_instances.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        hair_reader = csv.DictReader(handle)
        assert hair_reader.fieldnames is not None
        assert list(hair_reader) == []
    canonical = _row(output / "image_traits.csv")
    assert tuple(canonical) == IMAGE_TRAIT_FIELDS
    assert len(canonical) == 82
    assert summary["hair_identities"] == 0
    assert summary["endpoint_complete_length_identities"] == 0
    assert canonical["hair_count"] == "0"
    assert canonical["mean_hair_length_um"] == ""
    assert canonical["median_hair_length_um"] == ""
    assert canonical["total_hair_length_um"] == "0.0"
    assert canonical["hair_density_per_mm_visible_root"] == "0.0"
    assert canonical["first_hair_distance_from_distal_point_um"] == ""
    assert canonical["first_hair_ge40um_distance_from_distal_point_um"] == ""
    assert canonical["local_hair_count_1_4mm"] == "0"
    assert canonical["local_hair_density_per_mm_1_4mm"] == "0.0"
    assert canonical["local_mean_hair_length_um_1_4mm"] == ""
    assert canonical["local_median_hair_length_um_1_4mm"] == ""
    assert canonical["local_total_hair_length_um_per_root_mm_1_4mm"] == "0.0"
    assert canonical["visible_hair_attachment_span_um_descriptive_right_censored"] == ""

    profile_summary = export_distal_axis_profiles(
        traits_csv=output / "traits.csv",
        hair_instances_csv=output / "hair_instances.csv",
        contract_json=PROJECT_ROOT / "configs/phaxis/v1_0/axial_profile_contract.json",
        output=tmp_path / "profiles",
        model_contract_proposal=PROPOSAL_BINDING,
        model_contract_public_identity=PUBLIC_IDENTITY,
    )
    assert profile_summary["rows"] == 5
    with (tmp_path / "profiles" / "distal_axis_profiles.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["attached_identity_count"] == "0" for row in rows)
    assert all(row["endpoint_complete_length_support_fraction"] == "" for row in rows)
    assert all(row["measured_total_hair_length_um"] == "0.0" for row in rows)


def test_scale_failure_and_metadata_mismatch_null_all_physical_traits(
    phaxis_case, tmp_path: Path
) -> None:
    _summary, failed_output = _export(
        phaxis_case,
        tmp_path / "prediction-failure",
        prediction_scale_fail_closed=True,
    )
    failed = _row(failed_output / "image_traits.csv")
    assert failed["scale_status"] == "fail_closed"
    assert failed["physical_units_valid"] == "False"
    assert failed["formal_statistics_eligible"] == "False"
    assert failed["automatic_measurement_fail_closed"] == "True"
    assert failed["exclusion_reason"] == "scale_fail_closed"
    assert failed["hair_count"] == "2"
    assert all(failed[field] == "" for field in ROOT_TRAIT_FIELDS)
    assert all(failed[field] == "" for field in HAIR_TRAIT_FIELDS if field != "hair_count")
    assert failed["root_cap_point_to_axis_bridge_um"] == ""
    assert failed["width_end_exclusion_um"] == ""

    _summary, mismatch_output = _export(
        phaxis_case,
        tmp_path / "metadata-mismatch",
        metadata_um_per_px=2.0,
    )
    mismatch = _row(mismatch_output / "image_traits.csv")
    assert mismatch["scale_status"] == "fail_closed"
    assert mismatch["physical_units_valid"] == "False"
    assert mismatch["formal_statistics_eligible"] == "False"
    assert mismatch["automatic_measurement_fail_closed"] == "True"
    assert mismatch["exclusion_reason"] == "scale_metadata_mismatch"
    assert mismatch["um_per_px"] == ""
    assert mismatch["hair_count"] == "2"
    assert all(mismatch[field] == "" for field in ROOT_TRAIT_FIELDS)
    assert all(mismatch[field] == "" for field in HAIR_TRAIT_FIELDS if field != "hair_count")
    assert mismatch["root_cap_point_to_axis_bridge_um"] == ""
    assert mismatch["width_end_exclusion_um"] == ""


def test_exported_length_provenance_uses_only_public_phaxis_nomenclature(
    phaxis_case, tmp_path: Path
) -> None:
    summary, output = _export(phaxis_case, tmp_path)
    canonical = _row(output / "image_traits.csv")
    hair = _row(output / "hair_instances.csv")
    traits = _row(output / "traits.csv")
    public_values = (
        summary["hair_length_expert"],
        canonical["hair_length_expert_id"],
        hair["length_expert"],
        traits["hair_length_semantics"],
    )
    assert all("PHAxis" in value for value in public_values)
    assert all("Hybrid-Max" not in value for value in public_values)
