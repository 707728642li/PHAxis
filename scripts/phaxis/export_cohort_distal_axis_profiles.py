#!/usr/bin/env python
"""Export clean261 and full283 distal-axis profiles from sealed cohorts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.axial_profiles import export_cohort_distal_axis_profiles  # noqa: E402
from phaxis.model_contract_binding import read_model_contract_authority  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohorts-root",
        type=Path,
        required=True,
        help="sealed cohorts_exact283/output directory",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "phaxis"
            / "v1_0"
            / "axial_profile_contract.json"
        ),
    )
    parser.add_argument("--model-contract-proposal", type=Path, required=True)
    parser.add_argument(
        "--traits-summary",
        type=Path,
        required=True,
        help="sealed full283 traits summary bound by the cohort receipt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new, non-existing destination directory",
    )
    args = parser.parse_args()
    proposal_binding = read_model_contract_authority(args.model_contract_proposal)
    result = export_cohort_distal_axis_profiles(
        cohorts_root=args.cohorts_root,
        contract_json=args.contract,
        output=args.output,
        model_contract_proposal=proposal_binding.receipt_fields(),
        model_contract_public_identity=proposal_binding.public_identity_fields(),
        traits_summary_json=args.traits_summary,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
