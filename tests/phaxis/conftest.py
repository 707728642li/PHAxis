from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from phaxis.hair_stageb.serialization import make_detection_payload


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def phaxis_case(tmp_path: Path):
    """Small, label-free fixture exercising the public fusion contract."""

    masks = tmp_path / "masks"
    axes = tmp_path / "axis_geometry"
    masks.mkdir()
    axes.mkdir()
    mask_path = masks / "T1.root.png"
    mask_path.write_bytes(b"immutable-mask-fixture")
    axis_path = axes / "T1.npz"
    np.savez_compressed(
        axis_path,
        path_xy=np.asarray(
            [[10.0, 10.0], [10.0, 20.0], [10.0, 30.0]], np.float32
        ),
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
            {
                "points_xy": [[7, 20], [-7, 20]],
                "complete_length_measurement_eligible": True,
            }
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
    return deepcopy(hybrid), deepcopy(stageb), tmp_path
