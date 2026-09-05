# Input contract

The production authority is PHAxis-analysis-workflow-manifest-1.0; all relative paths resolve against its containing directory, not the shell working directory. Each file reference carries a SHA-256. `manifest_identity_sha256` binds the full plan.

Required biological table fields include task_id (unique), image_sha256 and um_per_px (positive and trusted). Experimental metadata are joined after inference; they do not route models. One source root is the sampling unit. The production CLI validates exact schemas and task-set agreement. See [the user guide](phaxis/USER_GUIDE.md).

`phaxis analyze --manifest workflow.json --output results` is the validate-only/dry-run equivalent. It does not execute without `--execute`. Do not invent model hashes or physical calibration. Schema examples in `schemas/` describe the lightweight release/report interfaces; model/workflow validation remains the implemented Python contract.
