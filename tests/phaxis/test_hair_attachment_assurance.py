from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from phaxis.hair_attachment_assurance import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    FORMAL_ATTACHMENT_TOLERANCE_UM,
    FORMAL_MATCHER_CONFIG,
    HAIR_ATTACHMENT_COORDINATE_SPACE,
    HAIR_ATTACHMENT_INPUT_SCHEMA,
    HAIR_POLYLINE_ORIENTATION,
    PROXY_TOLERANCES_UM,
    SELECTED_PROXY_TOLERANCE_UM,
    HairAttachmentAssuranceError,
    build_from_input_contract,
    build_hair_attachment_assurance,
    main,
    validate_hair_attachment_assurance,
)
from phaxis.io import atomic_write_json, read_json, sha256_json


ANNOTATION_AUTHORITY = sha256_json({"annotation": "canonical-centreline-trunks"})
PREDICTION_AUTHORITY = sha256_json({"prediction": "selected-stageb"})
FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "hair_attachment_assurance_input.json"
)


def _record(source_unit: str) -> dict:
    return {
        "pair_type": "hair_attachment",
        "source_unit": source_unit,
        "source_image_sha256": sha256_json(["image", source_unit]),
        "coordinate_space": HAIR_ATTACHMENT_COORDINATE_SPACE,
        "polyline_orientation": HAIR_POLYLINE_ORIENTATION,
        "annotation_artifact_sha256": sha256_json(["annotation", source_unit]),
        "prediction_artifact_sha256": sha256_json(["prediction", source_unit]),
        "annotated_polylines_xy_um": [
            [[0.0, 0.0], [40.0, 0.0]],
            [[0.0, 20.0], [40.0, 20.0]],
        ],
        "predicted_polylines_xy_um": [
            [[1.0, 0.0], [39.0, 0.0]],
            [[25.0, 20.0], [40.0, 20.0]],
        ],
    }


def _fixture():
    records = [_record("hair-a")]
    payload = build_hair_attachment_assurance(
        records=records,
        source_units=["hair-a"],
        annotation_authority_sha256=ANNOTATION_AUTHORITY,
        prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
    )
    return records, payload


def test_formal_attachment_accuracy_reuses_biological_matches_without_rematching() -> None:
    records, payload = _fixture()
    assert validate_hair_attachment_assurance(
        payload,
        records=records,
        source_units=["hair-a"],
        annotation_authority_sha256=ANNOTATION_AUTHORITY,
        prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
    ) == payload
    summary = payload["summary"]
    proxy = summary["attachment_proxy_threshold_selection"]["20"]
    formal = summary["formal_matched_attachment_accuracy"]
    assert proxy["tp"] == 1
    assert formal["formal_biological_presence"]["tp"] == 2
    assert formal["attachment_qualified_identity"]["tp"] == 1
    assert formal["attachment_position_error_on_all_formal_identity_matches"]["n"] == 2
    assert formal["attachment_position_error_on_all_formal_identity_matches"]["mean_um"] == pytest.approx(13.0)
    assert payload["metric_contract"]["threshold_selection_used_as_formal_accuracy"] is False
    matches = payload["per_image"][0]["formal_matched_attachment_accuracy"]["formal_identity_matches"]
    assert sorted(match["attachment_error_um"] for match in matches) == [1.0, 25.0]
    assert payload["bootstrap"]["unit"] == "source_image"
    assert payload["bootstrap"]["repetitions"] == BOOTSTRAP_REPETITIONS
    assert payload["bootstrap"]["seed"] == BOOTSTRAP_SEED
    bootstrap = formal["bootstrap_95ci"]
    for key in (
        "formal_attachment_precision",
        "formal_attachment_recall",
        "formal_attachment_f1",
    ):
        assert bootstrap[key]["point_estimate"] == 0.5
        assert bootstrap[key]["ci_low_2_5"] == 0.5
        assert bootstrap[key]["ci_high_97_5"] == 0.5
        assert bootstrap[key]["estimable_replicates"] == BOOTSTRAP_REPETITIONS
    assert bootstrap["formal_attachment_error_median_um"]["point_estimate"] == 13.0
    assert bootstrap["formal_attachment_error_median_um"]["ci_low_2_5"] == 13.0
    assert bootstrap["formal_attachment_error_p95_um"]["point_estimate"] == pytest.approx(23.8)
    assert bootstrap["formal_attachment_error_p95_um"]["ci_high_97_5"] == pytest.approx(23.8)
    assert payload["per_image"][0]["bootstrap_sufficient_statistics"] == {
        "predicted_hairs": 2,
        "annotated_hairs": 2,
        "formal_attachment_qualified_true_positive": 1,
        "formal_attachment_errors_um": [1.0, 25.0],
    }


def test_empty_prediction_has_explicit_zero_prf_and_null_position_error() -> None:
    record = _record("hair-empty")
    record["predicted_polylines_xy_um"] = []
    payload = build_hair_attachment_assurance(
        records=[record],
        source_units=["hair-empty"],
        annotation_authority_sha256=ANNOTATION_AUTHORITY,
        prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
    )
    formal = payload["summary"]["formal_matched_attachment_accuracy"]
    assert formal["formal_biological_presence"]["tp"] == 0
    assert formal["attachment_qualified_identity"]["f1"] == 0.0
    error = formal["attachment_position_error_on_all_formal_identity_matches"]
    assert error == {"n": 0, "mean_um": None, "median_um": None, "p95_um": None, "max_um": None}
    bootstrap = formal["bootstrap_95ci"]
    assert bootstrap["formal_attachment_error_median_um"] == {
        "point_estimate": None,
        "ci_low_2_5": None,
        "ci_high_97_5": None,
        "estimable_replicates": 0,
    }


def test_duplicate_neighbour_ambiguity_is_exposed_by_tolerance_sensitivity() -> None:
    record = _record("hair-resolution")
    record["annotated_polylines_xy_um"] = [
        [[0.0, 0.0], [100.0, 0.0]],
        [[0.0, 12.0], [100.0, 12.0]],
    ]
    record["predicted_polylines_xy_um"] = [
        [[0.0, 0.0], [100.0, 0.0]],
        [[0.0, 0.0], [100.0, 0.0]],
    ]
    payload = build_hair_attachment_assurance(
        records=[record],
        source_units=["hair-resolution"],
        annotation_authority_sha256=ANNOTATION_AUTHORITY,
        prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
    )
    proxy = payload["summary"]["attachment_proxy_threshold_selection"]
    assert proxy["5"]["tp"] == 1
    assert proxy["10"]["tp"] == 1
    assert proxy["20"]["tp"] == 2
    assert payload["summary"]["formal_matched_attachment_accuracy"][
        "formal_biological_presence"
    ]["tp"] == 2


def test_rejects_orientation_denominator_and_metric_role_drift() -> None:
    records, payload = _fixture()
    changed = deepcopy(records)
    changed[0]["polyline_orientation"] = "unknown"
    with pytest.raises(HairAttachmentAssuranceError, match="orientation drift"):
        build_hair_attachment_assurance(
            records=changed,
            source_units=["hair-a"],
            annotation_authority_sha256=ANNOTATION_AUTHORITY,
            prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
        )
    with pytest.raises(HairAttachmentAssuranceError, match="denominator drift"):
        build_hair_attachment_assurance(
            records=records,
            source_units=["hair-a", "hair-missing"],
            annotation_authority_sha256=ANNOTATION_AUTHORITY,
            prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
        )
    changed_payload = deepcopy(payload)
    changed_payload["metric_contract"]["threshold_selection_used_as_formal_accuracy"] = True
    with pytest.raises(HairAttachmentAssuranceError, match="masquerades"):
        validate_hair_attachment_assurance(
            changed_payload,
            records=records,
            source_units=["hair-a"],
            annotation_authority_sha256=ANNOTATION_AUTHORITY,
            prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
        )
    duplicated_image = [_record("hair-a"), _record("hair-b")]
    duplicated_image[1]["source_image_sha256"] = duplicated_image[0][
        "source_image_sha256"
    ]
    with pytest.raises(HairAttachmentAssuranceError, match="one record per source image"):
        build_hair_attachment_assurance(
            records=duplicated_image,
            source_units=["hair-a", "hair-b"],
            annotation_authority_sha256=ANNOTATION_AUTHORITY,
            prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
        )


def _input_contract() -> dict:
    payload = {
        "schema_version": HAIR_ATTACHMENT_INPUT_SCHEMA,
        "source_units": ["hair-cli"],
        "annotation_authority_sha256": ANNOTATION_AUTHORITY,
        "prediction_authority_identity_sha256": PREDICTION_AUTHORITY,
        "metric_config": {
            "proxy_tolerances_um": list(PROXY_TOLERANCES_UM),
            "selected_proxy_tolerance_um": SELECTED_PROXY_TOLERANCE_UM,
            "formal_attachment_tolerance_um": FORMAL_ATTACHMENT_TOLERANCE_UM,
            "formal_matcher": dict(FORMAL_MATCHER_CONFIG),
        },
        "records": [_record("hair-cli")],
        "independent_accuracy_claim_allowed": False,
        "blind_images_used": 0,
    }
    payload["input_contract_identity_sha256"] = sha256_json(payload)
    return payload


def test_portable_input_contract_and_cli_emit_hash_bound_receipt(
    tmp_path: Path,
) -> None:
    contract = _input_contract()
    result = build_from_input_contract(contract)
    assert result["input_contract_identity_sha256"] == contract["input_contract_identity_sha256"]
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    atomic_write_json(input_path, contract)
    assert main(["--input", str(input_path), "--output", str(output_path)]) == 0
    written = read_json(output_path)
    assert written == result
    assert validate_hair_attachment_assurance(
        written,
        records=contract["records"],
        source_units=contract["source_units"],
        annotation_authority_sha256=contract["annotation_authority_sha256"],
        prediction_authority_identity_sha256=contract[
            "prediction_authority_identity_sha256"
        ],
    ) == written
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(["--input", str(input_path), "--output", str(output_path)])


def test_input_contract_rejects_hash_drift_and_nonlocked_metric_config() -> None:
    contract = _input_contract()
    contract["records"][0]["predicted_polylines_xy_um"][0][0][0] += 1.0
    with pytest.raises(HairAttachmentAssuranceError, match="input identity mismatch"):
        build_from_input_contract(contract)
    contract = _input_contract()
    contract["metric_config"]["formal_matcher"]["resample_points"] = 16
    contract["input_contract_identity_sha256"] = sha256_json(
        {key: value for key, value in contract.items() if key != "input_contract_identity_sha256"}
    )
    with pytest.raises(HairAttachmentAssuranceError, match="locked formal contract"):
        build_from_input_contract(contract)


def test_repository_fixture_is_a_complete_valid_portable_contract() -> None:
    contract = read_json(FIXTURE_PATH)
    result = build_from_input_contract(contract)
    assert result["status"] == "completed"
    assert result["source_unit_total"] == 1
    assert result["provenance"]["input_contract_identity_sha256"] == contract[
        "input_contract_identity_sha256"
    ]
