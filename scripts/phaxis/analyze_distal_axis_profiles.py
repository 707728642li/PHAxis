"""Summarize fixed PHAxis distal-axis profiles at the source-image unit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.axial_profile_analysis import analyze_distal_axis_profiles  # noqa: E402
from phaxis.model_contract_binding import read_model_contract_authority  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-profiles", required=True)
    parser.add_argument("--sensitivity-profiles", required=True)
    parser.add_argument(
        "--contract",
        default=str(
            PROJECT_ROOT
            / "configs"
            / "phaxis"
            / "v1_0"
            / "axial_profile_analysis_contract.json"
        ),
    )
    parser.add_argument("--model-contract-proposal", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    proposal_binding = read_model_contract_authority(args.model_contract_proposal)
    result = analyze_distal_axis_profiles(
        primary_profiles=args.primary_profiles,
        sensitivity_profiles=args.sensitivity_profiles,
        contract_json=args.contract,
        output=args.output,
        model_contract_proposal=proposal_binding.receipt_fields(),
        model_contract_public_identity=proposal_binding.public_identity_fields(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
