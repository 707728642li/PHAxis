"""Build a hash-preserving, load-balanced PHAxis inference manifest."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.io import atomic_write_json, sha256_file


FIELDS = ("task_id", "image_path", "image_sha256", "um_per_px", "source_megapixels")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-metadata", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument(
        "--root-input-manifest",
        type=Path,
        help=(
            "optional immutable root-provider raw-image manifest; when supplied, "
            "task/path/scale identity must equal the production source set"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()
    if args.shards < 1:
        raise ValueError("--shards must be positive")

    metadata_rows = _read_csv(args.analysis_metadata)
    review_rows = _read_csv(args.review_manifest)
    review = {row["task_id"]: row for row in review_rows}
    if len(review) != len(review_rows):
        raise RuntimeError("duplicate task_id in review manifest")
    rows: list[dict[str, str]] = []
    for metadata in metadata_rows:
        task_id = metadata["task_id"]
        if task_id not in review:
            raise RuntimeError(f"{task_id}: absent from review manifest")
        reviewed = review[task_id]
        if reviewed["image_sha256"].casefold() != metadata["image_sha256"].casefold():
            raise RuntimeError(f"{task_id}: cross-manifest image hash mismatch")
        image_path = Path(reviewed["image_path"])
        observed = sha256_file(image_path)
        if observed.casefold() != metadata["image_sha256"].casefold():
            raise RuntimeError(f"{task_id}: source image hash mismatch")
        rows.append(
            {
                "task_id": task_id,
                "image_path": str(image_path.resolve()),
                "image_sha256": observed,
                "um_per_px": metadata["um_per_px"],
                "source_megapixels": metadata["source_megapixels"],
            }
        )
    if len({row["task_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate task_id in analysis metadata")

    root_input_sha256 = None
    root_source_alignment = None
    if args.root_input_manifest is not None:
        root_rows = _read_csv(args.root_input_manifest)
        root_by_task: dict[str, dict[str, str]] = {}
        for row in root_rows:
            task_id = str(row.get("image_id") or row.get("task_id") or "").strip()
            if not task_id or task_id in root_by_task:
                raise RuntimeError("duplicate or absent task ID in root input manifest")
            root_by_task[task_id] = row
        if set(root_by_task) != {row["task_id"] for row in rows}:
            raise RuntimeError("root/production raw-image task sets differ")
        for row in rows:
            task_id = row["task_id"]
            root_row = root_by_task[task_id]
            raw_root_path = root_row.get("input_path") or root_row.get("image_path")
            if not raw_root_path or Path(raw_root_path).resolve() != Path(row["image_path"]):
                raise RuntimeError(f"{task_id}: root/production source paths differ")
            raw_scale = root_row.get("source_um_per_px") or root_row.get("um_per_px")
            try:
                root_scale = float(str(raw_scale))
                production_scale = float(row["um_per_px"])
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"{task_id}: invalid root/production scale") from error
            if not (
                math.isfinite(root_scale)
                and math.isfinite(production_scale)
                and math.isclose(root_scale, production_scale, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise RuntimeError(f"{task_id}: root/production physical scales differ")
        root_input_sha256 = sha256_file(args.root_input_manifest)
        root_source_alignment = "passed_exact_task_path_scale_identity"

    _write_csv(args.output / "manifest_all.csv", rows)
    shard_rows: list[list[dict[str, str]]] = [[] for _ in range(args.shards)]
    shard_loads = [0.0] * args.shards
    for row in sorted(rows, key=lambda item: float(item["source_megapixels"]), reverse=True):
        shard = min(range(args.shards), key=lambda index: shard_loads[index])
        shard_rows[shard].append(row)
        shard_loads[shard] += float(row["source_megapixels"])
    for index, items in enumerate(shard_rows):
        _write_csv(args.output / f"manifest_shard{index:02d}.csv", items)
    summary = {
        "schema_version": "PHAxis-production-manifest-1.0",
        "status": "completed",
        "images": len(rows),
        "source_megapixels": sum(float(row["source_megapixels"]) for row in rows),
        "shards": args.shards,
        "shard_images": [len(items) for items in shard_rows],
        "shard_megapixels": shard_loads,
        "analysis_metadata_sha256": sha256_file(args.analysis_metadata),
        "review_manifest_sha256": sha256_file(args.review_manifest),
        "root_input_manifest_sha256": root_input_sha256,
        "root_source_alignment": root_source_alignment,
        "manifest_all_sha256": sha256_file(args.output / "manifest_all.csv"),
        "shard_sha256": [
            sha256_file(args.output / f"manifest_shard{index:02d}.csv")
            for index in range(args.shards)
        ],
        "blind_images_used": 0,
    }
    atomic_write_json(args.output / "summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
