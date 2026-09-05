# CPU container recipes

Build from repository root: `docker build -f containers/Dockerfile -t phaxis:1.0.0-local .`.
Run with a writable mounted output directory, e.g. `docker run --rm -v <absolute-output>:/results phaxis:1.0.0-local demo --output /results/demo`.
Apptainer: `apptainer build phaxis-1.0.0-local.sif containers/Apptainer.def`.
These are prepared CPU demo/report recipes, not tested GPU/model images. Pin a verified base-image digest and dependency lock for public release. No registry push is configured.
