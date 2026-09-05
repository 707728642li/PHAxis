from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from phaxis.io import read_json, sha256_json
from phaxis.root_continuity_assurance import (
    BOOTSTRAP_REPETITIONS,
    BOOTSTRAP_SEED,
    ROOT_CONTINUITY_COORDINATE_SPACE,
    ROOT_CONTINUITY_INPUT_SCHEMA,
    ROOT_CONTINUITY_PREDICTION_DEFINITION,
    ROOT_CONTINUITY_REFERENCE_DEFINITION,
    RootContinuityAssuranceError,
    build_from_input_contract,
    build_root_continuity_assurance,
    main,
    validate_root_continuity_assurance,
)


REFERENCE_AUTHORITY = sha256_json({"reference": "canonical-vector-root-axis"})
PREDICTION_AUTHORITY = sha256_json({"prediction": "sealed-root-mask-components"})
FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "root_continuity_assurance_input.json"
)


def _record(source_unit: str, components: list[list[list[float]]]) -> dict:
    return {
        "pair_type": "primary_root_continuity",
        "source_unit": source_unit,
        "source_image_sha256": sha256_json(["image", source_unit]),
        "coordinate_space": ROOT_CONTINUITY_COORDINATE_SPACE,
        "reference_axis_definition": ROOT_CONTINUITY_REFERENCE_DEFINITION,
        "prediction_axis_definition": ROOT_CONTINUITY_PREDICTION_DEFINITION,
        "reference_axis_artifact_sha256": sha256_json(["reference", source_unit]),
        "prediction_axis_artifact_sha256": sha256_json(["prediction", source_unit]),
        "reference_axis_xy_um": [[0.0, 0.0], [100.0, 0.0]],
        "predicted_axis_components_xy_um": components,
    }


def _fixture():
    source_units = ["root-perfect", "root-broken"]
    records = [
        _record("root-perfect", [[[0.0, 0.0], [100.0, 0.0]]]),
        _record(
            "root-broken",
            [
                [[0.0, 0.0], [40.0, 0.0]],
                [[60.0, 0.0], [100.0, 0.0]],
            ],
        ),
    ]
    payload = build_root_continuity_assurance(
        records=records,
        source_units=source_units,
        reference_authority_sha256=REFERENCE_AUTHORITY,
        prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
        support_tolerance_um=5.0,
        sampling_step_um=2.0,
    )
    return source_units, records, payload


def test_continuity_metrics_separate_internal_break_from_visible_extent() -> None:
    source_units, records, payload = _fixture()
    assert validate_root_continuity_assurance(
        payload,
        records=records,
        source_units=source_units,
        reference_authority_sha256=REFERENCE_AUTHORITY,
        prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
    ) == payload
    perfect, broken = payload["per_image"]
    assert perfect["reference_axis_coverage"] == 1.0
    assert perfect["longest_unsupported_gap_um"] == 0.0
    assert perfect["break_free"] is True
    assert broken["reference_axis_coverage"] == pytest.approx(0.92)
    assert broken["longest_unsupported_gap_um"] == pytest.approx(8.0)
    assert broken["break_free"] is False
    assert broken["maximum_single_component_coverage"] < broken[
        "reference_axis_coverage"
    ]
    assert broken["longest_unsupported_gap_um_on_best_component"] > broken[
        "longest_unsupported_gap_um"
    ]
    assert broken["spanning_component_count"] == 0
    # Both ends are visible, so extent is correct while the internal-break
    # metric still catches the missing middle segment.
    assert broken["predicted_visible_axis_extent_um"] == pytest.approx(100.0)
    assert broken["visible_axis_extent_error_um_abs"] == pytest.approx(0.0)
    assert payload["summary"]["break_free_image_rate"] == 0.5
    assert payload["bootstrap"] == {
        "method": "source-image nonparametric percentile bootstrap",
        "unit": "source_image",
        "repetitions": BOOTSTRAP_REPETITIONS,
        "seed": BOOTSTRAP_SEED,
        "interval": "two-sided 95% percentile (2.5%, 97.5%)",
        "sufficient_statistics_location": "per_image[*].bootstrap_sufficient_statistics",
    }
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    indices = generator.integers(0, 2, size=(BOOTSTRAP_REPETITIONS, 2))
    expected = np.mean(np.asarray([1.0, 0.92])[indices], axis=1)
    interval = payload["summary"]["bootstrap_95ci"][
        "reference_axis_coverage_mean"
    ]
    low, high = np.quantile(expected, (0.025, 0.975))
    assert interval["ci_low_2_5"] == pytest.approx(low)
    assert interval["ci_high_97_5"] == pytest.approx(high)
    assert all("bootstrap_sufficient_statistics" in row for row in payload["per_image"])
    for row in payload["per_image"]:
        sufficient = row["bootstrap_sufficient_statistics"]
        assert sufficient["maximum_single_component_coverage"] == row[
            "maximum_single_component_coverage"
        ]
        assert sufficient[
            "longest_unsupported_gap_um_on_best_component"
        ] == row["longest_unsupported_gap_um_on_best_component"]

    single_component_coverage = np.asarray(
        [row["maximum_single_component_coverage"] for row in payload["per_image"]]
    )
    best_component_gap = np.asarray(
        [
            row["longest_unsupported_gap_um_on_best_component"]
            for row in payload["per_image"]
        ]
    )
    expected_bootstrap = {
        "maximum_single_component_coverage_mean": np.mean(
            single_component_coverage[indices], axis=1
        ),
        "maximum_single_component_coverage_median": np.median(
            single_component_coverage[indices], axis=1
        ),
        "longest_unsupported_gap_um_on_best_component_median": np.median(
            best_component_gap[indices], axis=1
        ),
    }
    expected_points = {
        "maximum_single_component_coverage_mean": np.mean(
            single_component_coverage
        ),
        "maximum_single_component_coverage_median": np.median(
            single_component_coverage
        ),
        "longest_unsupported_gap_um_on_best_component_median": np.median(
            best_component_gap
        ),
    }
    for metric_name, estimates in expected_bootstrap.items():
        metric_interval = payload["summary"]["bootstrap_95ci"][metric_name]
        expected_low, expected_high = np.quantile(estimates, (0.025, 0.975))
        assert metric_interval["point_estimate"] == pytest.approx(
            expected_points[metric_name]
        )
        assert metric_interval["ci_low_2_5"] == pytest.approx(expected_low)
        assert metric_interval["ci_high_97_5"] == pytest.approx(expected_high)
        assert metric_interval["estimable_replicates"] == BOOTSTRAP_REPETITIONS


def test_missing_prediction_is_measured_as_full_gap_and_extent_error() -> None:
    records = [_record("missing", [])]
    payload = build_root_continuity_assurance(
        records=records,
        source_units=["missing"],
        reference_authority_sha256=REFERENCE_AUTHORITY,
        prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
    )
    row = payload["per_image"][0]
    assert row["reference_axis_coverage"] == 0.0
    assert row["longest_unsupported_gap_um"] == pytest.approx(100.0)
    assert row["longest_unsupported_gap_um_on_best_component"] == pytest.approx(
        100.0
    )
    assert row["maximum_single_component_coverage"] == 0.0
    assert row["spanning_component_count"] == 0
    assert row["visible_axis_extent_error_um_abs"] == pytest.approx(100.0)
    assert row["break_free"] is False


def test_rejects_interpolated_semantics_and_denominator_or_hash_drift() -> None:
    source_units, records, payload = _fixture()
    changed = deepcopy(records)
    changed[0]["prediction_axis_definition"] = "display line after gap filling"
    with pytest.raises(RootContinuityAssuranceError, match="definition drift"):
        build_root_continuity_assurance(
            records=changed,
            source_units=source_units,
            reference_authority_sha256=REFERENCE_AUTHORITY,
            prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
        )
    with pytest.raises(RootContinuityAssuranceError, match="denominator drift"):
        build_root_continuity_assurance(
            records=records[:-1],
            source_units=source_units,
            reference_authority_sha256=REFERENCE_AUTHORITY,
            prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
        )
    changed_payload = deepcopy(payload)
    changed_payload["summary"]["break_free_images"] = 2
    with pytest.raises(RootContinuityAssuranceError, match="values, denominator, or identity drift"):
        validate_root_continuity_assurance(
            changed_payload,
            records=records,
            source_units=source_units,
            reference_authority_sha256=REFERENCE_AUTHORITY,
            prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
        )
    duplicated_image = [_record("root-a", []), _record("root-b", [])]
    duplicated_image[1]["source_image_sha256"] = duplicated_image[0][
        "source_image_sha256"
    ]
    with pytest.raises(RootContinuityAssuranceError, match="one record per source image"):
        build_root_continuity_assurance(
            records=duplicated_image,
            source_units=["root-a", "root-b"],
            reference_authority_sha256=REFERENCE_AUTHORITY,
            prediction_authority_identity_sha256=PREDICTION_AUTHORITY,
        )


def _input_contract() -> dict:
    record = _record("root-cli", [[[0.0, 0.0], [100.0, 0.0]]])
    payload = {
        "schema_version": ROOT_CONTINUITY_INPUT_SCHEMA,
        "source_units": ["root-cli"],
        "reference_authority_sha256": REFERENCE_AUTHORITY,
        "prediction_authority_identity_sha256": PREDICTION_AUTHORITY,
        "metric_config": {"support_tolerance_um": 5.0, "sampling_step_um": 2.0},
        "records": [record],
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
    from phaxis.io import atomic_write_json

    atomic_write_json(input_path, contract)
    assert main(["--input", str(input_path), "--output", str(output_path)]) == 0
    written = read_json(output_path)
    assert written == result
    assert validate_root_continuity_assurance(
        written,
        records=contract["records"],
        source_units=contract["source_units"],
        reference_authority_sha256=contract["reference_authority_sha256"],
        prediction_authority_identity_sha256=contract[
            "prediction_authority_identity_sha256"
        ],
    ) == written
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(["--input", str(input_path), "--output", str(output_path)])


def test_input_contract_fails_closed_on_identity_or_blind_drift() -> None:
    contract = _input_contract()
    contract["records"][0]["reference_axis_xy_um"][1][0] = 99.0
    with pytest.raises(RootContinuityAssuranceError, match="input identity mismatch"):
        build_from_input_contract(contract)
    contract = _input_contract()
    contract["blind_images_used"] = 1
    contract["input_contract_identity_sha256"] = sha256_json(
        {key: value for key, value in contract.items() if key != "input_contract_identity_sha256"}
    )
    with pytest.raises(RootContinuityAssuranceError, match="blind-tainted"):
        build_from_input_contract(contract)


def test_repository_fixture_is_a_complete_valid_portable_contract() -> None:
    contract = read_json(FIXTURE_PATH)
    result = build_from_input_contract(contract)
    assert result["status"] == "completed"
    assert result["source_unit_total"] == 1
    assert result["provenance"]["input_contract_identity_sha256"] == contract[
        "input_contract_identity_sha256"
    ]
    row = result["per_image"][0]
    # The two disconnected fragments are only 10 um apart.  At a 5-um
    # support tolerance their union covers every midpoint, recreating the
    # exact false-positive topology case guarded here.
    assert row["reference_axis_coverage"] == 1.0
    assert row["longest_unsupported_gap_um"] == 0.0
    assert row["maximum_single_component_coverage"] == 0.5
    assert row["longest_unsupported_gap_um_on_best_component"] == 50.0
    assert row["spanning_component_count"] == 0
    assert row["union_reference_axis_fully_supported"] is True
    assert row["union_coverage_hides_fragmentation"] is True
    assert row["break_free"] is False
    assert result["summary"]["union_coverage_hides_fragmentation_images"] == 1
    assert result["summary"]["break_free_image_rate"] == 0.0
