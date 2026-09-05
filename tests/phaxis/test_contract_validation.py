from __future__ import annotations

from copy import deepcopy

import pytest

from phaxis.contracts import (
    ContractError,
    root_lock_sha256,
    validate_hybrid_prediction,
    validate_stageb_detection_payload,
)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(blind_images_used=1), "blind_images_used"),
        (lambda payload: payload.update(root_cap_region_output=True), "root-cap region"),
        (lambda payload: payload.update(root_cap_point_xy=[1.0]), "distal/root-cap point"),
        (lambda payload: payload.pop("root_mask_relpath"), "root_mask_relpath"),
    ],
)
def test_hybrid_contract_rejects_invalid_scientific_state(
    phaxis_case, mutation, message
):
    hybrid, _, artifact_root = phaxis_case
    mutation(hybrid)
    with pytest.raises(ContractError, match=message):
        validate_hybrid_prediction(hybrid, artifact_root=artifact_root)


def test_root_lock_covers_root_traits_but_not_hair_traits(phaxis_case):
    hybrid, _, _ = phaxis_case
    original = root_lock_sha256(hybrid)

    hair_change = deepcopy(hybrid)
    hair_change["phenotypes"]["identity_tier"]["hair_count"] = 999.0
    assert root_lock_sha256(hair_change) == original

    root_change = deepcopy(hybrid)
    root_change["phenotypes"]["identity_tier"]["root_area_um2"] = 999.0
    assert root_lock_sha256(root_change) != original

    statistics_change = deepcopy(hybrid)
    statistics_change["detailed_root_statistics"]["visible_root_axis_length_um"] = 1.0
    assert root_lock_sha256(statistics_change) != original


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version="future"), "schema"),
        (lambda payload: payload.update(task_id="another"), "task_id mismatch"),
        (
            lambda payload: payload.update(source_image_sha256="b" * 64),
            "source image hash mismatch",
        ),
        (lambda payload: payload.pop("coordinate_space"), "coordinate_space"),
        (
            lambda payload: payload["coordinate_space"].update(working_um_per_px=0),
            "working_um_per_px",
        ),
        (lambda payload: payload.update(detections={}), "must be a list"),
        (
            lambda payload: payload["detections"][0].update(base_xy_working=[1.0]),
            "base_xy_working",
        ),
        (
            lambda payload: payload["detections"][0].update(score=1.01),
            "invalid score",
        ),
        (lambda payload: payload.update(blind_images_used=1), "blind_images_used"),
        (lambda payload: payload["model"].update(ensemble_members=4), "model lock"),
        (
            lambda payload: payload["operating_point"].update(score_threshold=0.2),
            "operating point",
        ),
        (lambda payload: payload.update(n=999), "n does not match"),
    ],
)
def test_stageb_contract_rejects_malformed_payload(phaxis_case, mutation, message):
    hybrid, stageb, _ = phaxis_case
    mutation(stageb)
    with pytest.raises(ContractError, match=message):
        validate_stageb_detection_payload(
            stageb,
            expected_task_id=hybrid["task_id"],
            expected_image_sha256=hybrid["source_image_sha256"],
        )


def test_stageb_contract_rejects_valid_looking_detection_tamper(phaxis_case):
    hybrid, stageb, _ = phaxis_case
    stageb["detections"][0]["predicted_length_um"] = 10.0
    with pytest.raises(ContractError, match="detection_identity_sha256 mismatch"):
        validate_stageb_detection_payload(
            stageb,
            expected_task_id=hybrid["task_id"],
            expected_image_sha256=hybrid["source_image_sha256"],
        )


def test_hybrid_contract_detects_missing_and_changed_artifacts(phaxis_case):
    hybrid, _, artifact_root = phaxis_case
    mask = artifact_root / hybrid["root_mask_relpath"]
    mask.unlink()
    with pytest.raises(ContractError, match="missing locked artifact"):
        validate_hybrid_prediction(hybrid, artifact_root=artifact_root)
