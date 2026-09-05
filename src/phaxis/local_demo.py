"""CPU numerical installation example; no inference or biological image access."""

from __future__ import annotations

import csv
from pathlib import Path
import time

from ._demo_fixture import make_case
from .fusion import fuse_hybrid_root_with_stageb_hairs
from .io import atomic_write_json, sha256_json
from .traits import ROOT_TRAIT_FIELDS, export_traits


def run_demo(output: str | Path, *, zero_hairs: bool = False) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Choose a new demo output directory: {output.name}")
    start = time.perf_counter()
    output.mkdir(parents=True)
    artifacts = output / "synthetic_inputs"
    artifacts.mkdir()
    root, hair, artifact_root = make_case(artifacts)
    root["detailed_root_statistics"] = {
        **{field: 1.0 for field in ROOT_TRAIT_FIELDS},
        "visible_root_axis_length_um": 40.0,
        "median_root_width_um": 6.0,
    }
    # These visibly synthetic identities never impersonate the production model.
    binding = {
        "model_contract_proposal_sha256": "b" * 64,
        "model_contract_proposal_identity_sha256": "c" * 64,
    }
    public = {
        "model_bundle_id": "PHAXIS-V1.0.0-STRICT-TRAIN399-SYNTHETIC",
        "root_expert_id": "PHAxis-root-provider-SYNTHETIC",
    }
    if zero_hairs:
        hair["detections"] = []
        hair["n"] = 0
    hair.update(binding | public)
    hair.pop("detection_identity_sha256", None)
    hair["detection_identity_sha256"] = sha256_json(hair)
    fused = fuse_hybrid_root_with_stageb_hairs(
        root,
        hair,
        hybrid_artifact_root=artifact_root,
        model_contract_proposal=binding,
        model_contract_public_identity=public,
    )
    fused["phaxis"]["hair_identity_count_expert"] = "PHAxis-synthetic-geometry-fixture"
    predictions = artifacts / "predictions"
    atomic_write_json(predictions / "T1.json", fused)
    metadata = artifacts / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "T1",
                "image_sha256": "a" * 64,
                "um_per_px": 1,
                "experiment_key": "synthetic",
                "condition_code": "synthetic",
                "study_role": "installation_test",
                "qc_disposition": "eligible",
            }
        )
    summary = export_traits(
        prediction_root=predictions,
        metadata_csv=metadata,
        output=output / "traits",
        model_contract_proposal=binding,
        model_contract_public_identity=public,
    )
    expected_count = 0 if zero_hairs else 2
    if summary["hair_identities"] != expected_count:
        raise RuntimeError("Synthetic numerical golden test: unexpected hair count")
    from .offline_report import build_report

    build_report(output / "traits", output / "report", synthetic=True)
    # Convenient top-level entry retains relative report assets.
    (output / "report.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        '<meta http-equiv="refresh" content="0;url=report/report.html">'
        '<a href="report/report.html">Open PHAxis synthetic report</a>',
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "PHAxis-synthetic-demo-1.0",
        "software_version": "1.0.0",
        "status": "passed",
        "synthetic": True,
        "model_accuracy_evidence": False,
        "gpu_used": False,
        "network_used": False,
        "source_roots": 1,
        "expected_hair_identities": expected_count,
        "observed_hair_identities": summary["hair_identities"],
        "seconds": round(time.perf_counter() - start, 6),
    }
    atomic_write_json(output / "demo_receipt.json", receipt)
    return receipt
