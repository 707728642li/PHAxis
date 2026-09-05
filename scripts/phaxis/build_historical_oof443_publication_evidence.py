#!/usr/bin/env python3
"""Build sealed OOF443 biological-presence publication evidence.

The command recomputes every sufficient-statistic cell from the explicitly
named, trusted historical OOF pickle and the canonical dataset manifest.  It
does not accept a pre-aggregated plotting CSV.  The legacy pickle is
deserialized only when the caller supplies ``--trusted-local-oof-pickle``;
its byte hash is captured before deserialization.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.evaluation_metrics import match_biological_hair_presence  # noqa: E402
from phaxis.io import atomic_write_json, sha256_file, sha256_json  # noqa: E402


SCHEMA_VERSION = "PHAxis-historical-OOF443-development-receipt-1.0"
TABLE_COLUMNS = (
    "source_unit",
    "family_key",
    "fold",
    "quality_band",
    "density_band",
    "annotation_mode",
    "n_pred",
    "n_gt",
    "biological_presence_tp_20um",
)
COUNT_BINS = (
    ("negative_0", 0, 0),
    ("sparse_1_49", 1, 49),
    ("medium_50_99", 50, 99),
    ("dense_100_199", 100, 199),
    ("very_dense_ge200", 200, None),
)


class HistoricalEvidenceError(RuntimeError):
    """Historical OOF input is incomplete or not family-isolated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HistoricalEvidenceError(message)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _require(bool(rows), f"empty CSV: {path}")
    return rows


def _density(count: int) -> str:
    for label, lower, upper in COUNT_BINS:
        if count >= lower and (upper is None or count <= upper):
            return label
    raise HistoricalEvidenceError(f"invalid hair count: {count}")


def _prediction_polylines(prediction: Mapping[str, Any], keep: np.ndarray) -> list[np.ndarray]:
    base = np.asarray(prediction.get("base"), dtype=np.float64)
    tip = np.asarray(prediction.get("tip"), dtype=np.float64)
    _require(base.shape == tip.shape == (len(keep), 2), "OOF base/tip geometry changed")
    return [np.stack((start, end)) for start, end in zip(base[keep], tip[keep], strict=True)]


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        prefix=f".{path.name}.",
        suffix=".partial",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False, columns=list(TABLE_COLUMNS))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_historical_evidence(
    *,
    oof_pickle: str | Path,
    dataset_manifest: str | Path,
    split_manifest: str | Path,
    output: str | Path,
    trusted_local_oof_pickle: bool,
    bootstrap_repetitions: int = 5000,
    bootstrap_seed: int = 20260828,
) -> dict[str, Any]:
    source = Path(oof_pickle).resolve()
    manifest_path = Path(dataset_manifest).resolve()
    split_path = Path(split_manifest).resolve()
    destination = Path(output).resolve()
    _require(trusted_local_oof_pickle, "refusing to deserialize pickle without explicit trust flag")
    for role, path in (("OOF pickle", source), ("dataset manifest", manifest_path), ("split manifest", split_path)):
        _require(path.is_file() and not path.is_symlink(), f"{role} is missing or symlinked")
        _require("blind" not in str(path).casefold(), f"{role} has a blind-labelled path")
    _require(not destination.exists(), f"refusing to overwrite {destination}")
    _require(int(bootstrap_repetitions) >= 200, "bootstrap repetitions must be at least 200")

    manifest_rows = _read_csv(manifest_path)
    split_rows = _read_csv(split_path)
    metadata = {str(row.get("task_id")): row for row in manifest_rows}
    split = {str(row.get("task_id")): row for row in split_rows}
    _require(len(metadata) == len(manifest_rows) == 443, "dataset manifest is not 443 unique tasks")
    _require(set(metadata) == set(split), "dataset and split task sets differ")
    train_families = {row["family_key"] for row in split_rows if row.get("split") == "train"}
    val_families = {row["family_key"] for row in split_rows if row.get("split") == "val"}
    _require(len([row for row in split_rows if row.get("split") == "train"]) == 399, "split is not train399")
    _require(len([row for row in split_rows if row.get("split") == "val"]) == 44, "split is not QC-development44")
    _require(not train_families.intersection(val_families), "locked train/QC family overlap is nonzero")

    source_sha = sha256_file(source)
    with source.open("rb") as handle:
        records = pickle.load(handle)  # noqa: S301 - explicit trusted-local flag + pre-hash
    _require(isinstance(records, list) and len(records) == 443, "OOF pickle is not a 443-record list")
    by_id = {str(record.get("task_id")): record for record in records if isinstance(record, Mapping)}
    _require(len(by_id) == 443 and set(by_id) == set(metadata), "OOF and canonical task identities differ")

    rows: list[dict[str, Any]] = []
    family_folds: dict[str, set[int]] = {}
    for task_id in sorted(by_id):
        record = by_id[task_id]
        row = metadata[task_id]
        _require(row.get("family_key") == split[task_id].get("family_key"), f"{task_id}: family metadata drift")
        fold = int(record.get("fold", -1))
        _require(0 <= fold < 5, f"{task_id}: invalid OOF fold")
        family_folds.setdefault(str(row["family_key"]), set()).add(fold)
        prediction = record.get("pred")
        ground_truth = record.get("gt")
        _require(isinstance(prediction, Mapping) and isinstance(ground_truth, Mapping), f"{task_id}: malformed OOF geometry")
        scores = np.asarray(prediction.get("score"), dtype=np.float64)
        threshold = float(record.get("thresh", np.nan))
        _require(scores.ndim == 1 and np.isfinite(scores).all(), f"{task_id}: invalid prediction scores")
        _require(np.isfinite(threshold) and 0.0 <= threshold <= 1.0, f"{task_id}: invalid threshold")
        keep = scores >= threshold
        predicted = _prediction_polylines(prediction, keep)
        annotated = [np.asarray(polyline, dtype=np.float64) for polyline in ground_truth.get("polys", ())]
        _require(all(polyline.ndim == 2 and polyline.shape[1] == 2 and len(polyline) >= 2 for polyline in annotated), f"{task_id}: invalid canonical polylines")
        scale = float(record.get("um_per_px", np.nan))
        metrics, _matches = match_biological_hair_presence(
            predicted,
            annotated,
            scale,
            20.0,
            minimum_truth_coverage=0.25,
            minimum_prediction_coverage=0.25,
            minimum_direction_cosine=0.0,
        )
        n_gt = len(annotated)
        _require(n_gt == int(record.get("n_gt", -1)) == int(row.get("root_hair_count", -2)), f"{task_id}: truth count drift")
        rows.append(
            {
                "source_unit": task_id,
                "family_key": str(row["family_key"]),
                "fold": fold,
                "quality_band": str(row["acquisition_quality_band"]),
                "density_band": _density(n_gt),
                "annotation_mode": "fully_manual" if bool(record.get("fully_manual")) else "model_assisted_refined",
                "n_pred": len(predicted),
                "n_gt": n_gt,
                "biological_presence_tp_20um": int(metrics["tp"]),
            }
        )
    _require(all(len(folds) == 1 for folds in family_folds.values()), "a family spans multiple OOF folds")
    _require(len({fold for folds in family_folds.values() for fold in folds}) == 5, "OOF evidence does not contain five folds")

    frame = pd.DataFrame(rows, columns=list(TABLE_COLUMNS))
    _require(int(frame["biological_presence_tp_20um"].sum()) <= min(int(frame["n_pred"].sum()), int(frame["n_gt"].sum())), "OOF true positives are impossible")
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        table_path = staging / "per_image_sufficient_statistics.csv"
        _atomic_csv(table_path, frame)
        receipt: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "completed_locked_historical_oof443_development",
            "scope": "family-isolated OOF443 development evidence; non-independent",
            "source_table_sha256": {
                "per_image_sufficient_statistics": sha256_file(table_path),
            },
            "source_authority_sha256": {
                "trusted_local_oof_pickle": source_sha,
                "dataset_manifest": sha256_file(manifest_path),
                "split_manifest": sha256_file(split_path),
            },
            "images": 443,
            "families": len(family_folds),
            "folds": 5,
            "family_key_overlap_across_folds": 0,
            "metric_contract": {
                "primary_metric": "one_to_one_tolerant_biological_hair_presence",
                "tolerance_um": 20.0,
                "minimum_truth_coverage": 0.25,
                "minimum_prediction_coverage": 0.25,
                "minimum_direction_cosine": 0.0,
                "endpoint_gate_used": False,
                "selection_proxy_used_as_final_metric": False,
            },
            "uncertainty": {
                "method": "image-level nonparametric bootstrap",
                "repetitions": int(bootstrap_repetitions),
                "seed": int(bootstrap_seed),
                "family_key_role": "split isolation only; not a bootstrap cluster",
            },
            "independent_accuracy_claim_allowed": False,
            "canonical_annotations_read_during_inference": False,
            "canonical_annotations_read_for_development_evaluation": True,
            "root_cap_region_statistics_included": False,
            "blind_images_used": 0,
        }
        receipt["historical_development_identity_sha256"] = sha256_json(receipt)
        atomic_write_json(staging / "historical_development_receipt.json", receipt)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof-pickle", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trusted-local-oof-pickle", action="store_true")
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    return parser


def main() -> None:
    args = _parser().parse_args()
    receipt = build_historical_evidence(
        oof_pickle=args.oof_pickle,
        dataset_manifest=args.dataset_manifest,
        split_manifest=args.split_manifest,
        output=args.output,
        trusted_local_oof_pickle=args.trusted_local_oof_pickle,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(receipt["historical_development_identity_sha256"])


if __name__ == "__main__":
    main()
