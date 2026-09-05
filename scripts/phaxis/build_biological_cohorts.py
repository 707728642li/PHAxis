#!/usr/bin/env python
"""Build the clean261 PHAxis primary and full283 sensitivity cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.biology import build_biological_cohorts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trait-export", type=Path, required=True)
    parser.add_argument(
        "--analysis-metadata",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/rhaxis_nextgen_hybrid_max_axis_ridge_review_workspace_full283_run8_final_auto/analysis_metadata.csv"
        ),
    )
    parser.add_argument(
        "--design-manifest",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/rhaxis_six_condition_study_design_v1_run2_final/canonical_unit_manifest.csv"
        ),
    )
    parser.add_argument("--overlap-audit", type=Path, required=True)
    parser.add_argument(
        "--analysis-contract",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs/phaxis/v1_0/biological_analysis_contract.json"
        ),
    )
    parser.add_argument("--model-contract-proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_biological_cohorts(
        trait_export=args.trait_export,
        analysis_metadata=args.analysis_metadata,
        design_manifest=args.design_manifest,
        overlap_audit=args.overlap_audit,
        analysis_contract=args.analysis_contract,
        model_contract_proposal=args.model_contract_proposal,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
