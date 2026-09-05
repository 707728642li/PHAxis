# Models, original images and annotations

The owner-authorized research assets are distributed separately from the Git
source tree in the [research asset preview](https://github.com/707728642li/PHAxis/releases/tag/assets-v1.0.0-preview).
The release becomes downloadable after all uploaded files pass SHA-256 checks.

| Collection | Contents |
| --- | --- |
| HumanCurated443 | 443 original TIFF images and 443 original annotation JSONs; 399 training and 44 QC-development images |
| Application cohort | 283 original TIFF images; 22 image hashes also occur in HumanCurated443, leaving Clean261 after overlap exclusion |
| Root-hair models | Five selected train399 checkpoints, seeds 2026082801–2026082805 |
| Primary-root model | The hash-bound root-provider bundle and its manifest |
| Metadata | Split, sample and calibration manifests, model/trait contracts, file hashes and asset rights |

No blind/final-validation data are included. Images and annotations are original
files, not rendered figure panels or recomputed labels. The image collections
contain 726 files representing 704 unique image hashes.

## Download and extract

Start with `README_ASSETS.md`, `ASSET_RIGHTS.md`, `ASSETS_MANIFEST.json` and
`SHA256SUMS.txt` on the release page. The complete download is approximately
10.6 GB; uncompressed images alone require approximately 20.3 GB. Keep room for
both downloaded archives and extracted files.

With GitHub CLI installed, download into an empty directory:

```console
gh release download assets-v1.0.0-preview --repo 707728642li/PHAxis --dir phaxis-assets
```

On Linux, verify the downloads before extraction:

```console
cd phaxis-assets
sha256sum --check SHA256SUMS.txt
```

On Windows, compare `Get-FileHash -Algorithm SHA256 <file>` with the corresponding
entry in `SHA256SUMS.txt`. Every uploaded file is also checked against the GitHub
SHA-256 digest before the release is made public.

The 18 image archives are **independently extractable tar.gz files**, not segments
to concatenate. Extract each into the same destination directory, then extract
the annotations-and-metadata archive there. Each archive contains a
`PART_CONTENTS_SHA256.json` inventory; retain that inventory under its archive's
name if your extraction utility asks to overwrite the previous part's inventory.
The image paths are `HumanCurated443/images/all/` and `exact283/images/`.

Pair annotations and images using the supplied manifests. The original annotation
JSONs retain legacy viewer paths; their embedded `image_path` is not a portable
installation path. The five checkpoint files are distributed individually.

## Use in PHAxis

These assets preserve the selected PHAxis 1.0.0 model identity and supply the
original microscopy evidence. They are not a fully configured, one-command GPU
deployment capsule. The inference environment, selection receipts and sealed
workflow manifest must be configured according to the
[user guide](phaxis/USER_GUIDE.md) before running `phaxis analyze --execute`.
The CPU-only `phaxis demo` remains independent of these downloads.

Read `ASSET_RIGHTS.md` before reuse. The Apache-2.0 source license does not
automatically apply to model or microscopy assets. Checkpoint files may contain
pickle serialization; verify their hashes and load trusted project artifacts
only. This GitHub asset preview does not publish the package to PyPI.
