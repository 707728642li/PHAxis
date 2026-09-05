"""Build hash-verified train or QC-development inference manifests.

This utility never reads annotation contents.  It selects immutable source
images from the canonical dataset manifest and preserves the locked split and
family separation for downstream label-free inference.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.io import atomic_write_json, sha256_file  # noqa: E402


FIELDS = ("task_id", "image_path", "image_sha256", "um_per_px", "source_megapixels")
EXPECTED = {"train": 399, "val": 44, "all": 443}


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _atomic_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build(
    *,
    dataset_root: Path,
    dataset_manifest: Path,
    split_manifest: Path,
    split: str,
    output: Path,
    shards: int,
) -> dict:
    if split not in EXPECTED:
        raise ValueError(f"unsupported split: {split}")
    if shards < 1:
        raise ValueError("shards must be positive")
    dataset_rows = _rows(dataset_manifest)
    split_rows = _rows(split_manifest)
    dataset = {row["task_id"]: row for row in dataset_rows}
    splits = {row["task_id"]: row for row in split_rows}
    if len(dataset) != 443 or len(splits) != 443 or set(dataset) != set(splits):
        raise RuntimeError("canonical dataset/split manifests must contain the same 443 tasks")
    for task_id in dataset:
        # The explicit split manifest is authoritative.  The locked
        # qc_development_v1_0 split deliberately swaps the complete 30C|COL-0
        # and 30C|RGA families so that RHAUD-358 remains training-only.
        if dataset[task_id]["family_key"] != splits[task_id]["family_key"]:
            raise RuntimeError(f"{task_id}: family_key disagreement between manifests")

    selected_ids = sorted(
        task_id
        for task_id, row in dataset.items()
        if split == "all" or splits[task_id]["split"] == split
    )
    if len(selected_ids) != EXPECTED[split]:
        raise RuntimeError(
            f"locked {split} selection has {len(selected_ids)} tasks, expected {EXPECTED[split]}"
        )
    inference_rows = []
    for task_id in selected_ids:
        row = dataset[task_id]
        image_path = dataset_root / row["image_relpath"]
        observed_sha256 = sha256_file(image_path)
        if observed_sha256.casefold() != row["image_sha256"].casefold():
            raise RuntimeError(f"{task_id}: source image SHA-256 mismatch")
        inference_rows.append(
            {
                "task_id": task_id,
                "image_path": str(image_path.resolve()),
                "image_sha256": observed_sha256,
                "um_per_px": row["source_um_per_px"],
                "source_megapixels": row["source_megapixels"],
            }
        )

    train_families = {
        row["family_key"] for row in splits.values() if row["split"] == "train"
    }
    val_families = {
        row["family_key"] for row in splits.values() if row["split"] == "val"
    }
    overlap = sorted(train_families & val_families)
    if overlap:
        raise RuntimeError(f"family_key leakage: {overlap[:5]}")

    _atomic_csv(output / "manifest_all.csv", inference_rows)
    shard_rows: list[list[dict[str, str]]] = [[] for _ in range(shards)]
    shard_load = [0.0] * shards
    for row in sorted(
        inference_rows, key=lambda item: float(item["source_megapixels"]), reverse=True
    ):
        target = min(range(shards), key=lambda index: shard_load[index])
        shard_rows[target].append(row)
        shard_load[target] += float(row["source_megapixels"])
    for index, rows in enumerate(shard_rows):
        _atomic_csv(output / f"manifest_shard{index:02d}.csv", rows)

    overrides = sorted(
        task_id
        for task_id in dataset
        if dataset[task_id]["split"] != splits[task_id]["split"]
    )
    summary = {
        "schema_version": "PHAxis-HumanCurated-inference-manifest-1.0",
        "status": "completed",
        "selection": split,
        "images": len(inference_rows),
        "source_megapixels": sum(
            float(row["source_megapixels"]) for row in inference_rows
        ),
        "shards": shards,
        "shard_images": [len(rows) for rows in shard_rows],
        "shard_megapixels": shard_load,
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "split_manifest_sha256": sha256_file(split_manifest),
        "manifest_all_sha256": sha256_file(output / "manifest_all.csv"),
        "shard_sha256": [
            sha256_file(output / f"manifest_shard{index:02d}.csv")
            for index in range(shards)
        ],
        "train_images": 399,
        "development_images": 44,
        "family_key_overlap": 0,
        "split_authority": str(split_manifest.resolve()),
        "dataset_declared_split_override_tasks": overrides,
        "dataset_declared_split_override_count": len(overrides),
        "rhaud_358_split": splits["RHAUD-358"]["split"],
        "annotation_contents_read": False,
        "blind_images_used": 0,
    }
    atomic_write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(EXPECTED), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    summary = build(
        dataset_root=args.dataset_root,
        dataset_manifest=args.dataset_manifest,
        split_manifest=args.split_manifest,
        split=args.split,
        output=args.output,
        shards=args.shards,
    )
    print(summary)


if __name__ == "__main__":
    main()
