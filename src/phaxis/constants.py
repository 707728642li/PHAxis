"""Locked public identities and operating points for PHAxis 1.0.0."""

from __future__ import annotations

PRODUCT_NAME = "PHAxis"
PRODUCT_VERSION = "1.0.0"
PREDICTION_SCHEMA = "PHAxis-prediction-1.0"
STAGEB_SCHEMA = "PHAxis-RHAxiscc-StageB-detections-1.0"
PUBLIC_HAIR_LENGTH_EXPERT_ID = (
    "PHAxis-1.0.0-endpoint-complete-root-hair-length"
)
PUBLIC_HAIR_LENGTH_SEMANTICS = (
    "PHAxis-1.0.0-conditional-on-endpoint-complete-centrelines"
)

# Read-only compatibility identity for historical 443CV detection payloads.
# Formal PHAxis 1.0.0 train399 runs obtain their public expert and bundle IDs
# from the sealed model-contract authority; this value is never a release ID.
LEGACY_HAIR_EXPERT_ID = "RHAxiscc-StageB-5fold-last-e60-hflip-20260827"
HAIR_WORKING_UM_PER_PX = 2.0
HAIR_OUT_STRIDE = 2
HAIR_WINDOW = 1024
HAIR_OVERLAP = 256
HAIR_BATCH = 4
HAIR_SCORE_THRESHOLD = 0.225
HAIR_NMS_KERNEL = 5
HAIR_MAX_INSTANCES = 4000
HAIR_ROOT_GATE_UM = (-90.0, 25.0)
HAIR_CHECKPOINT_SHA256 = (
    "a36c48802a2ed1120602319dc9e6c6d386cc64d87d90dacd421a24d77faafd35",
    "de3d32e99c65e4c9d9a785b974aadc8b1cde8ae15f90644ee8af28102466ab41",
    "d6dfe0b245fbe1c9af8ad56f153ad59d0941b8c9e13a452c8ef4de88ec868311",
    "342271324c4b3d6c3a149133747b512ff4955d2f397fc399dae5bdc0fa364e6a",
    "cc09a97c81cba2cc33f3c8269a8332afcf2c915821b3b5805cfb03c87879b0d5",
)

ROOT_CAP_REGION_OUTPUT = False
BLIND_IMAGES_USED = 0

ROOT_LOCK_TOP_LEVEL_FIELDS = (
    "source_image_sha256",
    "root_mask_relpath",
    "root_mask_sha256",
    "root_axis_geometry_relpath",
    "root_axis_geometry_sha256",
    "root_continuity_status",
    "root_continuity_applied",
    "root_continuity_added_mask_relpath",
    "root_continuity_added_mask_sha256",
    "root_source",
    "root_axis_source",
    "root_width_reference_mask_relpath",
    "root_width_reference_mask_sha256",
    "root_width_reference_axis_geometry_relpath",
    "root_width_reference_axis_geometry_sha256",
    "root_width_reference_point_xy",
    "root_global_width_source",
    "root_global_width_reference_applied",
    "root_global_width_fields",
    "root_global_width_qcdev_evidence",
    "root_cap_region_output",
    "root_cap_point_xy",
    "root_cap_point_source",
    "formal_phenotype_eligible",
    "automatic_measurement_fail_closed",
    "detailed_root_statistics",
    "detailed_root_statistics_review_only",
    "scale",
)

ROOT_PHENOTYPE_FIELDS = (
    "root_area_um2",
    "main_root_length_um",
    "main_root_width_um",
)
