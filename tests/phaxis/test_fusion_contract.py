from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from phaxis.contracts import ContractError, root_lock_sha256
from phaxis.fusion import _stageb_hairs, fuse_hybrid_root_with_stageb_hairs
from phaxis.hair_stageb.serialization import make_detection_payload
from phaxis.io import sha256_json


PROPOSAL_BINDING = {
    "model_contract_proposal_sha256": "b" * 64,
    "model_contract_proposal_identity_sha256": "c" * 64,
}
PUBLIC_IDENTITY = {
    "model_bundle_id": "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC",
    "root_expert_id": "PHAxis-root-provider-SYNTHETIC",
}


def _fusion_kwargs() -> dict[str, object]:
    return {
        "model_contract_proposal": PROPOSAL_BINDING,
        "model_contract_public_identity": PUBLIC_IDENTITY,
    }


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    masks = tmp_path / "masks"
    axes = tmp_path / "axis_geometry"
    masks.mkdir()
    axes.mkdir()
    mask_path = masks / "T1.root.png"
    mask_path.write_bytes(b"immutable-mask-fixture")
    axis_path = axes / "T1.npz"
    np.savez_compressed(
        axis_path,
        path_xy=np.asarray([[10.0, 10.0], [10.0, 20.0], [10.0, 30.0]], np.float32),
        distance_from_tip_px=np.asarray([0.0, 10.0, 20.0], np.float32),
        radius_px=np.asarray([3.0, 3.0, 3.0], np.float32),
        reference_radius_px=np.asarray(3.0, np.float32),
        point_xy=np.asarray([10.0, 10.0], np.float32),
        source_image_sha256=np.asarray("a" * 64),
    )
    hybrid = {
        "schema_version": "Hybrid-fixture",
        "task_id": "T1",
        "source_image_sha256": "a" * 64,
        "root_mask_relpath": "masks/T1.root.png",
        "root_mask_sha256": _digest(mask_path),
        "root_axis_geometry_relpath": "axis_geometry/T1.npz",
        "root_axis_geometry_sha256": _digest(axis_path),
        "root_source": "hybrid",
        "root_axis_source": "hybrid",
        "root_cap_region_output": False,
        "root_cap_point_xy": [10.0, 10.0],
        "root_cap_point_source": "hybrid",
        "formal_phenotype_eligible": True,
        "detailed_root_statistics": {"visible_root_axis_length_um": 40.0},
        "identity_hairs": [{"points_xy": [[1, 1], [2, 2]]}],
        "count_hairs": [{"points_xy": [[1, 1], [2, 2]]}],
        "length_hairs": [
            {"points_xy": [[7, 20], [-7, 20]], "complete_length_measurement_eligible": True}
        ],
        "length_hair_variant": "hybrid_endpoint_complete",
        "scale": {"um_per_px": 1.0, "fail_closed": False},
        "phenotypes": {
            "identity_tier": {
                "hair_count": 1.0,
                "mean_hair_length_um": 14.0,
                "median_hair_length_um": 14.0,
                "total_hair_length_um": 14.0,
                "root_area_um2": 120.0,
                "main_root_length_um": 40.0,
                "main_root_width_um": 12.0,
                "hair_density_per_mm": 25.0,
            },
            "count_tier": {
                "hair_count": 1.0,
                "mean_hair_length_um": 14.0,
                "median_hair_length_um": 14.0,
                "total_hair_length_um": 14.0,
                "root_area_um2": 120.0,
                "main_root_length_um": 40.0,
                "main_root_width_um": 12.0,
                "hair_density_per_mm": 25.0,
            },
        },
        "canonical_annotations_read_during_inference": False,
        "condition_metadata_used_for_routing": False,
        "blind_images_used": 0,
    }
    stageb = make_detection_payload(
        task_id="T1",
        source_image_sha256="a" * 64,
        source_um_per_px=1.0,
        prediction={
            "base": np.asarray([[6.5, 10.0], [6.5, 15.0]], np.float32),
            "tip": np.asarray([[2.0, 10.0], [2.0, 15.0]], np.float32),
            "score": np.asarray([0.9, 0.8], np.float32),
            "length_um": np.asarray([9.0, 9.0], np.float32),
            "working_shape": [64, 64],
            "source_to_working_scale": 0.5,
        },
        precision_mode="fp32_locked_oof",
    )
    stageb.update({**PROPOSAL_BINDING, **PUBLIC_IDENTITY})
    stageb.pop("detection_identity_sha256", None)
    stageb["detection_identity_sha256"] = sha256_json(stageb)
    return hybrid, stageb


def test_fusion_changes_only_hair_identity_count(tmp_path: Path):
    hybrid, stageb = _fixture(tmp_path)
    before = deepcopy(hybrid)
    lock = root_lock_sha256(hybrid)
    result = fuse_hybrid_root_with_stageb_hairs(
        hybrid, stageb, hybrid_artifact_root=tmp_path, **_fusion_kwargs()
    )
    assert hybrid == before
    assert result["phaxis"]["root_lock_sha256"] == lock
    assert root_lock_sha256(result) == lock
    assert len(result["identity_hairs"]) == 2
    assert len(result["count_hairs"]) == 2
    assert len(result["length_hairs"]) == 1
    assert result["identity_hairs"][0]["points_xy"][0] == [13.0, 20.0]
    tier = result["phenotypes"]["identity_tier"]
    assert tier["hair_count"] == 2.0
    assert tier["hair_density_per_mm"] == 50.0
    assert tier["mean_hair_length_um"] == 14.0
    assert tier["hair_length_measurement_fraction"] == 0.5
    assert result["root_cap_region_output"] is False
    assert result["blind_images_used"] == 0


def test_fusion_rejects_cross_image_hair_payload(tmp_path: Path):
    hybrid, stageb = _fixture(tmp_path)
    stageb["source_image_sha256"] = "b" * 64
    with pytest.raises(ContractError, match="source image hash mismatch"):
        fuse_hybrid_root_with_stageb_hairs(
            hybrid, stageb, hybrid_artifact_root=tmp_path, **_fusion_kwargs()
        )


def test_fusion_rejects_modified_root_artifact(tmp_path: Path):
    hybrid, stageb = _fixture(tmp_path)
    (tmp_path / hybrid["root_mask_relpath"]).write_bytes(b"changed")
    with pytest.raises(ContractError, match="artifact hash mismatch"):
        fuse_hybrid_root_with_stageb_hairs(
            hybrid, stageb, hybrid_artifact_root=tmp_path, **_fusion_kwargs()
        )


def test_fusion_rejects_hybrid_stageb_physical_scale_mismatch(tmp_path: Path):
    hybrid, stageb = _fixture(tmp_path)
    hybrid["scale"]["um_per_px"] = 1.5
    with pytest.raises(ContractError, match="physical scale mismatch"):
        fuse_hybrid_root_with_stageb_hairs(
            hybrid, stageb, hybrid_artifact_root=tmp_path, **_fusion_kwargs()
        )


def test_fusion_records_explicit_stageb_reference_evaluation_scale(tmp_path: Path):
    hybrid, stageb = _fixture(tmp_path)
    hybrid["scale"]["um_per_px"] = 1.5
    result = fuse_hybrid_root_with_stageb_hairs(
        hybrid,
        stageb,
        hybrid_artifact_root=tmp_path,
        physical_scale_contract="stageb_reference_evaluation",
        **_fusion_kwargs(),
    )
    scale_contract = result["phaxis"]["physical_scale_contract"]
    assert scale_contract["mode"] == "stageb_reference_evaluation"
    assert scale_contract["root_provider_scale_um_per_px"] == 1.5
    assert scale_contract["stageb_source_scale_um_per_px"] == 1.0
    assert scale_contract["relative_scale_difference"] == 0.5
    assert scale_contract["root_provider_scale_output_unchanged"] is True


def test_fusion_uses_realized_xy_scale_for_noninteger_resize_roundtrip():
    source_shape = np.asarray([7, 11])
    working_shape = np.asarray([5, 7])
    scale_xy = np.asarray(
        [working_shape[1] / source_shape[1], working_shape[0] / source_shape[0]]
    )
    base_source = np.asarray([10.0, 6.0])
    tip_source = np.asarray([2.0, 1.0])
    payload = {
        "coordinate_space": {
            "working_um_per_px": 2.0,
            "source_um_per_px": 1.3,
            "source_to_working_scale_xy": scale_xy.tolist(),
        },
        "detections": [
            {
                "base_xy_working": (base_source * scale_xy).tolist(),
                "tip_xy_working": (tip_source * scale_xy).tolist(),
                "score": 0.9,
                "predicted_length_um": 20.0,
            }
        ],
    }
    hair = _stageb_hairs(payload)[0]
    np.testing.assert_allclose(hair["points_xy"][0], base_source, atol=1e-12)
    np.testing.assert_allclose(hair["points_xy"][1], tip_source, atol=1e-12)


def test_stageb_hair_provenance_distinguishes_train399_from_legacy():
    payload = {
        "model": {"checkpoint_policy": "five_seed_train399_last_epoch_60"},
        "coordinate_space": {
            "working_um_per_px": 2.0,
            "source_um_per_px": 1.0,
            "source_to_working_scale_xy": [0.5, 0.5],
        },
        "detections": [
            {
                "base_xy_working": [5.0, 6.0],
                "tip_xy_working": [2.0, 3.0],
                "score": 0.9,
                "predicted_length_um": 10.0,
            }
        ],
    }
    train399_hair = _stageb_hairs(payload)[0]
    assert train399_hair["source"] == "phaxis_stage_b_train399"
    assert train399_hair["source_instance_id"] == "PHSB-0001"

    legacy_payload = deepcopy(payload)
    legacy_payload["model"]["checkpoint_policy"] = "legacy_five_fold"
    legacy_hair = _stageb_hairs(legacy_payload)[0]
    assert legacy_hair["source"] == "rhaxiscc_stage_b"
    assert legacy_hair["source_instance_id"] == "RHCCB-0001"
