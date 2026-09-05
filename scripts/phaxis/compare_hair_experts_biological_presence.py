"""Historical QC-development44 comparison under the PHAxis biological metric.

This command is deliberately labelled as a semantic/historical precheck.  The
Stage-B records come from the legacy five-fold OOF artefact whose folds jointly
cover all 443 development images, so this output must never be promoted as the
formal train399/QC-development44 result.  Its purpose is to compare the two
hair representations with the same biologically appropriate matcher while
fully hashing every input prediction.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.evaluation_metrics import (  # noqa: E402
    biological_hair_presence_matcher_contract,
    match_biological_hair_presence,
    precision_recall_f1,
)
from phaxis.io import (  # noqa: E402
    atomic_write_json,
    read_json,
    sha256_file,
    sha256_json,
)


TOLERANCES_UM = (5.0, 10.0, 20.0)
OOF_WORKING_UM_PER_PX = 2.0


def _ids(path: Path) -> list[str]:
    values = [value.strip() for value in path.read_text(encoding="utf-8").splitlines()]
    returned = [value for value in values if value]
    if len(returned) != 44 or len(set(returned)) != 44:
        raise RuntimeError("locked development set must contain 44 unique IDs")
    return returned


def _metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    returned = {row["task_id"]: row for row in rows}
    if len(returned) != len(rows):
        raise RuntimeError(f"duplicate task_id in {path}")
    return returned


def _straight_stageb_polylines(record: dict) -> list[np.ndarray]:
    keep = np.asarray(record["pred"]["score"], dtype=np.float64) >= float(
        record["thresh"]
    )
    bases = np.asarray(record["pred"]["base"], dtype=np.float64)[keep]
    tips = np.asarray(record["pred"]["tip"], dtype=np.float64)[keep]
    return [
        np.stack((base, tip)) * OOF_WORKING_UM_PER_PX
        for base, tip in zip(bases, tips, strict=True)
    ]


def _hybrid_polylines(payload: dict, source_um_per_px: float) -> list[np.ndarray]:
    if int(payload.get("blind_images_used", 0)) != 0:
        raise RuntimeError(f"{payload.get('task_id')}: blind_images_used must be zero")
    return [
        np.asarray(hair["points_xy"], dtype=np.float64) * source_um_per_px
        for hair in payload["identity_hairs"]
    ]


def _presence(polylines: list[np.ndarray], truth: list[np.ndarray]) -> dict[str, int]:
    matcher = biological_hair_presence_matcher_contract()
    return {
        str(int(tolerance)): int(
            match_biological_hair_presence(
                polylines,
                truth,
                1.0,
                tolerance,
                minimum_truth_coverage=matcher["minimum_truth_coverage"],
                minimum_prediction_coverage=matcher[
                    "minimum_prediction_coverage"
                ],
                minimum_direction_cosine=matcher["minimum_direction_cosine"],
                proximal_arc_fraction=matcher["proximal_arc_fraction"],
                resample_points=matcher["resample_points"],
            )[0]["tp"]
        )
        for tolerance in TOLERANCES_UM
    }


def _pool(rows: list[dict], expert: str) -> dict:
    predicted = int(sum(row[expert]["n_pred"] for row in rows))
    annotated = int(sum(row[expert]["n_gt"] for row in rows))
    counts_predicted = np.asarray(
        [row[expert]["n_pred"] for row in rows], dtype=np.float64
    )
    counts_annotated = np.asarray(
        [row[expert]["n_gt"] for row in rows], dtype=np.float64
    )
    error = counts_predicted - counts_annotated
    covariance = float(
        np.cov(counts_predicted, counts_annotated, ddof=1)[0, 1]
    )
    ccc = float(
        2.0
        * covariance
        / (
            np.var(counts_predicted, ddof=1)
            + np.var(counts_annotated, ddof=1)
            + (counts_predicted.mean() - counts_annotated.mean()) ** 2
        )
    )
    return {
        "images": len(rows),
        "predicted_hairs": predicted,
        "annotated_hairs": annotated,
        "tolerant_biological_presence": {
            str(int(tolerance)): precision_recall_f1(
                int(
                    sum(
                        row[expert]["biological_presence_tp"][str(int(tolerance))]
                        for row in rows
                    )
                ),
                predicted,
                annotated,
            )
            for tolerance in TOLERANCES_UM
        },
        "count": {
            "mae": float(np.abs(error).mean()),
            "bias": float(error.mean()),
            "pearson_r": float(np.corrcoef(counts_predicted, counts_annotated)[0, 1]),
            "ccc": ccc,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof-pickle", type=Path, required=True)
    parser.add_argument("--hybrid-predictions", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--locked-val-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    task_ids = _ids(args.locked_val_ids)
    metadata = _metadata(args.dataset_manifest)
    with args.oof_pickle.open("rb") as handle:
        records = pickle.load(handle)
    oof = {record["task_id"]: record for record in records}
    if len(oof) != len(records):
        raise RuntimeError("duplicate task_id in OOF pickle")

    rows: list[dict] = []
    prediction_files: list[dict[str, str]] = []
    for task_id in task_ids:
        record = oof[task_id]
        truth = [
            np.asarray(value, dtype=np.float64) * OOF_WORKING_UM_PER_PX
            for value in record["gt"]["polys"]
        ]
        stageb = _straight_stageb_polylines(record)
        hybrid_path = args.hybrid_predictions / f"{task_id}.json"
        hybrid_payload = read_json(hybrid_path)
        if hybrid_payload.get("task_id") != task_id:
            raise RuntimeError(f"task identity mismatch: {hybrid_path}")
        hybrid = _hybrid_polylines(
            hybrid_payload, float(metadata[task_id]["source_um_per_px"])
        )
        prediction_files.append(
            {"task_id": task_id, "sha256": sha256_file(hybrid_path)}
        )
        row = {"task_id": task_id}
        for name, polylines in (("stageb_historical_oof", stageb), ("hybrid_max", hybrid)):
            row[name] = {
                "n_pred": len(polylines),
                "n_gt": len(truth),
                "biological_presence_tp": _presence(polylines, truth),
            }
        rows.append(row)

    pooled = {
        expert: _pool(rows, expert)
        for expert in ("stageb_historical_oof", "hybrid_max")
    }
    stageb = pooled["stageb_historical_oof"]
    hybrid = pooled["hybrid_max"]
    checks = {
        "images_44": stageb["images"] == hybrid["images"] == 44,
        "annotations_3800": stageb["annotated_hairs"]
        == hybrid["annotated_hairs"]
        == 3800,
        "stageb_predictions_3926": stageb["predicted_hairs"] == 3926,
        "hybrid_predictions_3317": hybrid["predicted_hairs"] == 3317,
        "historical_stageb_biological_f1_20um_locked": bool(
            np.isclose(
                stageb["tolerant_biological_presence"]["20"]["f1"],
                0.883510225213632,
                rtol=0.0,
                atol=1e-12,
            )
        ),
        "blind_images_used_0": all(
            int(read_json(args.hybrid_predictions / f"{task_id}.json").get("blind_images_used", 0))
            == 0
            for task_id in task_ids
        ),
    }
    payload = {
        "schema_version": "PHAxis-historical-hair-biological-presence-QCdev44-1.0",
        "status": "passed" if all(checks.values()) else "failed",
        "evidence_role": "historical_semantic_precheck_only",
        "scope": "locked overlay-visible QC-development44; not independent accuracy",
        "primary_metric_definition": (
            "one-to-one tolerant biological-hair presence; bidirectional partial "
            "centreline coverage without endpoint gates"
        ),
        "primary_matcher_contract": biological_hair_presence_matcher_contract(),
        "primary_matcher_contract_sha256": sha256_json(
            biological_hair_presence_matcher_contract()
        ),
        "metric_parameters": {
            "tolerances_um": list(TOLERANCES_UM),
            "minimum_truth_coverage": biological_hair_presence_matcher_contract()[
                "minimum_truth_coverage"
            ],
            "minimum_prediction_coverage": biological_hair_presence_matcher_contract()[
                "minimum_prediction_coverage"
            ],
            "minimum_direction_cosine": biological_hair_presence_matcher_contract()[
                "minimum_direction_cosine"
            ],
            "resample_points": biological_hair_presence_matcher_contract()[
                "resample_points"
            ],
            "legacy_oof_working_um_per_px": OOF_WORKING_UM_PER_PX,
        },
        "checks": checks,
        "metrics": pooled,
        "f1_delta_stageb_minus_hybrid": {
            key: stageb["tolerant_biological_presence"][key]["f1"]
            - hybrid["tolerant_biological_presence"][key]["f1"]
            for key in ("5", "10", "20")
        },
        "per_image": rows,
        "inputs": {
            "oof_pickle_sha256": sha256_file(args.oof_pickle),
            "dataset_manifest_sha256": sha256_file(args.dataset_manifest),
            "locked_val_ids_sha256": sha256_file(args.locked_val_ids),
            "hybrid_prediction_files": prediction_files,
            "hybrid_prediction_set_identity": sha256_json(prediction_files),
        },
        "stageb_training_role": (
            "legacy five-fold OOF; the fold collection jointly trained across all "
            "443 development images; never formal train399 evidence"
        ),
        "validation_labels_used_for_formal_train399_training": False,
        "independent_accuracy_claim_allowed": False,
        "blind_images_used": 0,
    }
    atomic_write_json(args.output, payload)
    print(
        f"{payload['status']}: biological-presence F1@20 historical StageB="
        f"{stageb['tolerant_biological_presence']['20']['f1']:.6f}, Hybrid="
        f"{hybrid['tolerant_biological_presence']['20']['f1']:.6f}"
    )
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
