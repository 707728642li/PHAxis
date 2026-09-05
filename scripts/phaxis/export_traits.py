"""Export PHAxis 1.0.0 root, hair, and per-instance phenotype tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from phaxis.traits import export_traits
from phaxis.model_contract_binding import read_model_contract_authority


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--model-contract-proposal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    binding = read_model_contract_authority(args.model_contract_proposal)
    result = export_traits(
        prediction_root=args.predictions,
        metadata_csv=args.metadata,
        output=args.output,
        model_contract_proposal=binding.receipt_fields(),
        model_contract_public_identity=binding.public_identity_fields(),
    )
    print(json.dumps(result | {"prediction_sha256": f"{result['tasks']} records"}, indent=2))


if __name__ == "__main__":
    main()
