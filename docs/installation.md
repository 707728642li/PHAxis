# Installation

Python 3.10–3.12 is the declared CPU contract range. Local receipts report the exact version tested; configured CI is not an executed platform claim.

Prefer a dedicated Conda environment (`conda create -p ./envs/phaxis python=3.12 pip`). Activate it and install the supplied wheel with pip. No Bioconda recipe or channel is involved.

The base distribution requires NumPy, SciPy and packaging. `deployment` adds the image and model runtime; `analysis`, `inference`, `visualization`, `publication`, `test`, `docs`, `dev` and `build` separate optional uses. CPU demo and report need no Torch.

An authorized model capsule must separately supply the root-provider runtime and five train399 checkpoint hashes. Merely installing `[deployment]` does not install that capsule. For full deployment see [the user guide](phaxis/USER_GUIDE.md).
