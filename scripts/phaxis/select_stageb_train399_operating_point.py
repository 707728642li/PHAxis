"""Seal the preregistered QCdev44 Stage-B operating point on CPU."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.hair_stageb.selection import (  # noqa: E402
    build_selection_receipt_from_paths,
    write_selection_receipt_and_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--selected-model-metadata", type=Path, required=True)
    args = parser.parse_args()

    receipt, pending = build_selection_receipt_from_paths(
        candidate_manifest_path=args.candidate_manifest,
        candidate_pool_dir=args.candidate_pool,
        dataset_root=args.dataset_root,
        dataset_manifest=args.dataset_manifest,
        split_manifest=args.split_manifest,
    )
    selected = write_selection_receipt_and_metadata(
        receipt=receipt,
        pending_model_metadata=pending,
        receipt_path=args.selection_receipt,
        selected_model_metadata_path=args.selected_model_metadata,
    )
    metric = receipt["selected"]
    print(f"selected threshold: {selected['selected_score_threshold']:.3f}")
    print(
        "development biological-presence F1@20um: "
        f"{metric['tolerant_biological_presence_20um']['f1']:.6f}"
    )
    print(
        "development attachment-proxy F1@20um (secondary): "
        f"{metric['identity_attachment_proxy_20um']['f1']:.6f}"
    )
    print(f"development count MAE: {metric['count_mae']:.6f}")
    print(f"receipt identity: {receipt['selection_receipt_identity_sha256']}")
    print("scope: QC-development44 model selection only; not independent accuracy")


if __name__ == "__main__":
    main()
