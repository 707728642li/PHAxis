"""Export preregistered distal-axis root-hair profiles from PHAxis traits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.axial_profiles import export_distal_axis_profiles  # noqa: E402
from phaxis.model_contract_binding import read_model_contract_authority  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traits", required=True, help="PHAxis traits.csv")
    parser.add_argument("--hair-instances", required=True, help="PHAxis hair_instances.csv")
    parser.add_argument(
        "--contract",
        default=str(PROJECT_ROOT / "configs" / "phaxis" / "v1_0" / "axial_profile_contract.json"),
    )
    parser.add_argument("--model-contract-proposal", required=True)
    parser.add_argument(
        "--traits-summary",
        required=True,
        help="sealed upstream traits summary carrying the same proposal binding",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    proposal_binding = read_model_contract_authority(args.model_contract_proposal)
    result = export_distal_axis_profiles(
        traits_csv=args.traits,
        hair_instances_csv=args.hair_instances,
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
