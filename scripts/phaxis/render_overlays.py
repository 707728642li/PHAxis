"""Render PHAxis review overlays for a manifest."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import sys
import tempfile

import cv2
import tifffile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.io import atomic_write_json, read_json, sha256_file
from phaxis.rendering import render_prediction_overlay


def _atomic_csv(path: Path, rows: list[dict[str, object]], fields: tuple[str, ...]) -> None:
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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metadata",
        type=Path,
        help="optional CSV used to group review PNGs into biological categories",
    )
    parser.add_argument("--category-field", default="condition_code")
    args = parser.parse_args()
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    categories: dict[str, str] = {}
    if args.metadata is not None:
        with args.metadata.open("r", encoding="utf-8-sig", newline="") as handle:
            metadata_rows = list(csv.DictReader(handle))
        for metadata in metadata_rows:
            task_id = metadata["task_id"]
            if task_id in categories:
                raise RuntimeError(f"duplicate task_id in metadata: {task_id}")
            categories[task_id] = metadata.get(args.category_field, "") or "unclassified"
    records = []
    for row in rows:
        task_id = row["task_id"]
        image_path = Path(row["image_path"])
        if sha256_file(image_path) != row["image_sha256"]:
            raise RuntimeError(f"{task_id}: image hash mismatch")
        prediction = read_json(args.predictions / f"{task_id}.json")
        image = tifffile.imread(image_path)
        overlay = render_prediction_overlay(
            image, prediction, artifact_root=args.artifact_root
        )
        category = categories.get(task_id, "unclassified")
        category_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", category).strip("._")
        category_slug = category_slug or "unclassified"
        output_path = args.output / category_slug / f"{task_id}.phaxis_overlay.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), overlay, [cv2.IMWRITE_PNG_COMPRESSION, 6]):
            raise RuntimeError(f"failed to write {output_path}")
        records.append(
            {
                "task_id": task_id,
                "category": category,
                "identities": len(prediction["identity_hairs"]),
                "endpoint_complete_lengths": len(prediction.get("length_hairs", ())),
                "output_png": str(output_path.resolve()),
                "output_png_sha256": sha256_file(output_path),
                "width": int(overlay.shape[1]),
                "height": int(overlay.shape[0]),
            }
        )
        print(f"rendered {task_id} -> {output_path}")
    atomic_write_json(
        args.output / "summary.json",
        {
            "schema_version": "PHAxis-review-overlays-1.0",
            "status": "completed",
            "images": records,
            "category_field": args.category_field if args.metadata is not None else None,
            "category_counts": {
                category: sum(record["category"] == category for record in records)
                for category in sorted({record["category"] for record in records})
            },
            "blind_images_used": 0,
        },
    )
    index_fields = (
        "task_id",
        "category",
        "identities",
        "endpoint_complete_lengths",
        "output_png",
        "output_png_sha256",
        "width",
        "height",
    )
    _atomic_csv(args.output / "review_index.csv", records, index_fields)
    checklist = [
        {
            "task_id": record["task_id"],
            "category": record["category"],
            "output_png": record["output_png"],
            "root_continuity_ok": "",
            "distal_point_ok": "",
            "hair_identity_count_ok": "",
            "low_contrast_hairs_ok": "",
            "notes": "",
        }
        for record in records
    ]
    _atomic_csv(
        args.output / "review_checklist.csv",
        checklist,
        (
            "task_id",
            "category",
            "output_png",
            "root_continuity_ok",
            "distal_point_ok",
            "hair_identity_count_ok",
            "low_contrast_hairs_ok",
            "notes",
        ),
    )


if __name__ == "__main__":
    main()
