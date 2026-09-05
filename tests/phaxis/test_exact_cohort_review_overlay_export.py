from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

from phaxis.io import atomic_write_json, read_json, sha256_file, sha256_json
from scripts.phaxis import build_condition_blinded_overlay_evidence as overlay


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path) -> dict[str, object]:
    tasks_by_role = {
        "representative": "RHSCU-review-representative",
        "low_contrast": overlay.LOCKED_ANCHOR_TASK_IDS["low_contrast"],
        "curved_dense": overlay.LOCKED_ANCHOR_TASK_IDS["curved_dense"],
        "continuity": "RHSCU-review-continuity",
        "fail_closed": "RHSCU-review-fail-closed",
    }
    plan = tmp_path / "overlay_case_plan.csv"
    _write_csv(
        plan,
        [
            {"case_role": role, "task_id": task_id}
            for role, task_id in tasks_by_role.items()
        ],
    )
    fusion = tmp_path / "fusion"
    for child in ("predictions", "masks", "axis_geometry"):
        (fusion / child).mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    trait_rows: list[dict[str, object]] = []
    fusion_rows: list[dict[str, object]] = []
    for index, (role, task_id) in enumerate(tasks_by_role.items()):
        raw = tmp_path / "images" / f"{task_id}.ome.tif"
        raw.parent.mkdir(parents=True, exist_ok=True)
        y_grid, x_grid = np.mgrid[:64, :64]
        image = (
            1000
            + index * 50
            + 4 * x_grid
            + 3 * y_grid
            + 80 * np.sin(y_grid / 8.0)
        ).astype(np.uint16)
        tifffile.imwrite(raw, image)
        raw_sha = sha256_file(raw)

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[4:61, 27:38] = 255
        mask_path = fusion / "masks" / f"{task_id}.root.png"
        assert cv2.imwrite(str(mask_path), mask)
        y = np.linspace(58.0, 6.0, 21)
        x = 32.0 + (6.0 * np.sin(np.linspace(0.0, np.pi, 21)))
        path_xy = np.column_stack((x, y)).astype(np.float32)
        distance = np.concatenate(
            (
                [0.0],
                np.cumsum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1)),
            )
        ).astype(np.float32)
        axis_path = fusion / "axis_geometry" / f"{task_id}.npz"
        np.savez_compressed(
            axis_path,
            path_xy=path_xy,
            distance_from_tip_px=distance,
            radius_px=np.full(len(path_xy), 5.0, dtype=np.float32),
            source_image_sha256=np.asarray(raw_sha),
        )
        prediction = {
            "schema_version": "PHAxis-prediction-1.0",
            "task_id": task_id,
            "source_image_sha256": raw_sha,
            "root_mask_relpath": f"masks/{task_id}.root.png",
            "root_mask_sha256": sha256_file(mask_path),
            "root_axis_geometry_relpath": f"axis_geometry/{task_id}.npz",
            "root_axis_geometry_sha256": sha256_file(axis_path),
            "root_cap_point_xy": path_xy[0].astype(float).tolist(),
            "identity_hairs": [
                {
                    "source_instance_id": "H-0001",
                    "points_xy": [[28.0, 30.0], [18.0, 26.0]],
                    "complete_length_measurement_eligible": True,
                }
            ],
            "length_hairs": [
                {
                    "identity_source_instance_id": "H-0001",
                    "points_xy": [
                        [28.0, 30.0],
                        [23.0, 28.0],
                        [18.0, 26.0],
                    ],
                }
            ],
            "condition_metadata_used_for_routing": False,
            "canonical_annotations_read_during_inference": False,
            "root_cap_region_output": False,
            "blind_images_used": 0,
        }
        prediction_path = fusion / "predictions" / f"{task_id}.json"
        atomic_write_json(prediction_path, prediction)
        manifest_rows.append(
            {
                "task_id": task_id,
                "image_path": str(raw.resolve()),
                "image_sha256": raw_sha,
                "um_per_px": 5.0,
            }
        )
        trait_rows.append(
            {
                "task_id": task_id,
                "source_image_sha256": raw_sha,
                "experiment_key": "D15_8d" if index < 3 else "WT_series",
                "condition_code": "RHD6_OE_30C" if index % 2 else "RHD6_EV_22C",
                "formal_statistics_eligible": role != "fail_closed",
                "hair_count": 1,
                "hair_length_measurement_hair_count": 1,
            }
        )
        fusion_rows.append(
            {
                "task_id": task_id,
                "prediction_sha256": sha256_file(prediction_path),
                "phaxis_identity_count": 1,
                "matched_endpoint_complete_lengths": 1,
            }
        )
    manifest = tmp_path / "manifest_all.csv"
    traits = tmp_path / "traits.csv"
    _write_csv(manifest, manifest_rows)
    _write_csv(traits, trait_rows)
    summary = {
        "schema_version": "PHAxis-fusion-run-1.1",
        "status": "completed",
        "images": 5,
        "records": fusion_rows,
        "condition_metadata_used_for_routing": False,
        "canonical_annotations_read": False,
        "root_cap_region_output": False,
        "blind_images_used": 0,
    }
    summary["summary_identity_sha256"] = sha256_json(summary)
    atomic_write_json(fusion / "fusion_summary.json", summary)
    return {
        "case_plan": plan,
        "manifest": manifest,
        "traits": traits,
        "fusion": fusion,
        "tasks_by_role": tasks_by_role,
    }


def _run(fixture: dict[str, object], output: Path) -> dict[str, object]:
    return overlay.build_overlay_evidence(
        case_plan=fixture["case_plan"],
        application_manifest=fixture["manifest"],
        full_traits=fixture["traits"],
        fusion_root=fixture["fusion"],
        output=output,
        expected_task_count=5,
    )


def test_exact_cohort_review_export_is_sealed_categorized_and_single_render_authority(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "overlay-evidence"
    receipt = _run(fixture, output)
    unsigned_receipt = deepcopy(receipt)
    assert unsigned_receipt.pop("overlay_selection_identity_sha256") == sha256_json(
        unsigned_receipt
    )
    assert receipt["schema_version"] == overlay.SCHEMA_VERSION
    assert receipt["status"] == overlay.RECEIPT_STATUS
    assert receipt["exact_cohort_review_images"] == 5
    assert receipt["experimental_condition_metadata_used_for_evidence_assembly"] is False
    assert receipt[
        "experimental_condition_metadata_used_for_evidence_assembly_scope"
    ] == overlay.CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE
    review = receipt["full_cohort_review_export"]
    assert review["images"] == review["index_rows"] == review["checklist_rows"] == 5
    assert review["experimental_condition_metadata_used_for_prediction"] is False
    assert review["experimental_condition_metadata_used_for_rendering"] is False
    assert (
        review["experimental_condition_metadata_used_for_evidence_assembly"]
        is False
    )
    assert review["experimental_condition_metadata_used_for_evidence_assembly_scope"] == (
        overlay.CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE
    )
    assert review["experimental_condition_metadata_used_for_output_organization"] is True
    assert review["blind_images_used"] == 0
    readme = output / overlay.REVIEW_README_NAME
    assert readme.is_file()
    assert sha256_file(readme) == review["readme_cn_sha256"]
    readme_text = readme.read_text(encoding="utf-8")
    assert "identity/presence" in readme_text
    assert "endpoint-complete" in readme_text
    assert "只表示 Stage-B 向量末端" in readme_text
    assert "只有绿色 matched curve" in readme_text
    assert "PNG 像素固定后" in readme_text
    assert "不得把本批审阅意见反馈为本次模型的训练标签" in readme_text

    summary = read_json(output / overlay.REVIEW_SUMMARY_NAME)
    unsigned_summary = deepcopy(summary)
    assert unsigned_summary.pop("review_export_identity_sha256") == sha256_json(
        unsigned_summary
    )
    assert summary["experimental_condition_metadata_used_for_evidence_assembly"] is False
    assert summary["experimental_condition_metadata_used_for_evidence_assembly_scope"] == (
        overlay.CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE
    )
    with (output / overlay.REVIEW_INDEX_NAME).open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        index = list(csv.DictReader(handle))
    with (output / overlay.REVIEW_CHECKLIST_NAME).open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        checklist = list(csv.DictReader(handle))
    assert len(index) == len(checklist) == 5
    assert [row["task_id"] for row in index] == sorted(
        fixture["tasks_by_role"].values()
    )
    assert sum(row["review_partition"] == "formal" for row in index) == 4
    assert sum(row["review_partition"] == "review_only" for row in index) == 1
    assert {row["review_status"] for row in index} == {
        overlay.REVIEW_PENDING_STATUS
    }
    assert {row["identity_hair_count"] for row in index} == {"1"}
    assert {row["endpoint_complete_length_count"] for row in index} == {"1"}
    for row in index:
        png = output / row["output_png_relative_path"]
        assert png.is_file() and not png.is_symlink()
        assert sha256_file(png) == row["output_png_sha256"]
        assert png.stat().st_size == int(row["output_png_bytes"])
        assert (int(row["width_px"]), int(row["height_px"])) == (64, 64)
    assert len(list((output / overlay.REVIEW_ROOT_NAME).rglob("*.png"))) == 5

    with (output / "overlay_selection.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        selection = list(csv.DictReader(handle))
    assert len(selection) == 5
    for row in selection:
        paper_png = output / row["overlay_path"]
        review_png = output / row["full_cohort_review_overlay_path"]
        assert sha256_file(paper_png) == sha256_file(review_png)
        assert row["overlay_sha256"] == row["full_cohort_review_overlay_sha256"]
        assert row["overlay_bytes_reused_from_full_cohort_review_export"] == "True"
        assert row["experimental_condition_metadata_used_for_evidence_assembly"] == "False"
        assert row["experimental_condition_metadata_used_for_evidence_assembly_scope"] == (
            overlay.CONDITION_METADATA_EVIDENCE_ASSEMBLY_SCOPE
        )

    with pytest.raises(overlay.OverlayEvidenceError, match="overwrite"):
        _run(fixture, output)


def test_condition_metadata_changes_paths_but_not_rendered_overlay_bytes(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    first = tmp_path / "first"
    _run(fixture, first)
    traits_path = Path(fixture["traits"])
    with traits_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for index, row in enumerate(rows):
        row["experiment_key"] = f"renamed_experiment_{index % 2}"
        row["condition_code"] = f"renamed_condition_{index % 3}"
    _write_csv(traits_path, rows)
    second = tmp_path / "second"
    _run(fixture, second)

    def hashes(root: Path) -> dict[str, str]:
        with (root / overlay.REVIEW_INDEX_NAME).open(
            encoding="utf-8-sig", newline=""
        ) as handle:
            return {
                row["task_id"]: row["output_png_sha256"]
                for row in csv.DictReader(handle)
            }

    assert hashes(first) == hashes(second)
    assert sha256_file(first / overlay.REVIEW_INDEX_NAME) != sha256_file(
        second / overlay.REVIEW_INDEX_NAME
    )


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escape",
        "a/b",
        "..",
        "",
        "CON",
        "con.txt",
        "PRN.csv",
        "AUX",
        "NUL.json",
        "COM1",
        "com9.png",
        "LPT1",
        "lpt9.csv",
    ],
)
def test_review_path_task_ids_fail_closed(unsafe: str) -> None:
    with pytest.raises(overlay.OverlayEvidenceError, match="unsafe task_id"):
        overlay._safe_task_id(unsafe)


@pytest.mark.parametrize(
    "unsafe", ["CON", "con.txt", "PRN.csv", "AUX", "NUL.json", "COM4", "LPT7.log"]
)
def test_review_category_slugs_reject_windows_device_names(unsafe: str) -> None:
    with pytest.raises(overlay.OverlayEvidenceError, match="unsafe condition_code"):
        overlay._safe_slug(unsafe, role="condition_code")


def test_review_category_slugs_fail_closed_on_casefold_collision(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    traits_path = Path(fixture["traits"])
    with traits_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["condition_code"] = "CaseSensitiveGroup"
    rows[1]["condition_code"] = "casesensitivegroup"
    _write_csv(traits_path, rows)
    output = tmp_path / "casefold-collision"
    with pytest.raises(overlay.OverlayEvidenceError, match="condition slug collision"):
        _run(fixture, output)
    assert not output.exists()


def test_overlay_colour_manifest_matches_renderer_bgr_pixels() -> None:
    renderer_bgr = {
        "root_boundary_colour": (220, 170, 25),
        "axis_colour": (230, 230, 230),
        "distal_colour": (255, 60, 220),
        "length_curve_colour": (90, 245, 115),
        "identity_vector_colour": (20, 205, 255),
        "hair_base_colour": (0, 255, 255),
        "visible_endpoint_colour": (25, 105, 255),
    }
    observed = {
        key: f"#{red:02X}{green:02X}{blue:02X}"
        for key, (blue, green, red) in renderer_bgr.items()
    }
    assert overlay.COLOURS == observed


def test_count_drift_fails_without_publishing_partial_destination(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    traits_path = Path(fixture["traits"])
    with traits_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["hair_count"] = "2"
    _write_csv(traits_path, rows)
    output = tmp_path / "count-drift"
    with pytest.raises(
        overlay.OverlayEvidenceError, match="trait/prediction identity count mismatch"
    ):
        _run(fixture, output)
    assert not output.exists()


def test_duplicate_fusion_record_fails_closed_before_rendering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary_path = Path(fixture["fusion"]) / "fusion_summary.json"
    summary = read_json(summary_path)
    summary.pop("summary_identity_sha256")
    assert len(summary["records"]) == 5
    summary["records"][1] = deepcopy(summary["records"][0])
    summary["summary_identity_sha256"] = sha256_json(summary)
    atomic_write_json(summary_path, summary)

    output = tmp_path / "duplicate-fusion-record"
    with pytest.raises(
        overlay.OverlayEvidenceError,
        match="duplicate/case-colliding tasks",
    ):
        _run(fixture, output)
    assert not output.exists()


def test_overlength_fusion_record_list_fails_closed_before_rendering(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    summary_path = Path(fixture["fusion"]) / "fusion_summary.json"
    summary = read_json(summary_path)
    summary.pop("summary_identity_sha256")
    summary["records"].append(deepcopy(summary["records"][0]))
    summary["summary_identity_sha256"] = sha256_json(summary)
    atomic_write_json(summary_path, summary)

    output = tmp_path / "overlength-fusion-records"
    with pytest.raises(
        overlay.OverlayEvidenceError,
        match="exact list with expected_task_count rows",
    ):
        _run(fixture, output)
    assert not output.exists()


def test_nonlist_fusion_records_fail_closed_before_rendering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary_path = Path(fixture["fusion"]) / "fusion_summary.json"
    summary = read_json(summary_path)
    summary.pop("summary_identity_sha256")
    summary["records"] = {
        row["task_id"]: row for row in summary["records"]
    }
    summary["summary_identity_sha256"] = sha256_json(summary)
    atomic_write_json(summary_path, summary)

    output = tmp_path / "nonlist-fusion-records"
    with pytest.raises(
        overlay.OverlayEvidenceError,
        match="exact list with expected_task_count rows",
    ):
        _run(fixture, output)
    assert not output.exists()
