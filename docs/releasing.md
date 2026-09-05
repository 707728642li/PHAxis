# Release status and local preparation

**Update, 2026-09-05:** the owner authorized public GitHub source upload, followed
by model weights, original images and annotations. See
[current source status](https://github.com/707728642li/PHAxis/blob/main/GITHUB_SOURCE_STATUS.md)
and [research assets](research-assets.md). CPU/package/container and documentation
CI passed for source commit `466b149319b47ca9c207fb6beb3ddb5c7507e26c`.
PyPI publication and a stable software release remain separate steps.

## Historical local preparation

The preceding local preparation was completed without upload authorization.
This successor source preview does not replace the formal release-authority
registry. The sealed builder baseline is kept outside this candidate;
SOURCE_MANIFEST.json from that baseline must not be reused to certify these
added docs/UX files. LOCAL_SOURCE_MANIFEST.json inventories the historical local
candidate; GITHUB_SOURCE_MANIFEST.json inventories the current public source.

Prepared here: source, wheel, sdist, synthetic demo, offline report, documentation, CPU tests, containers, workflow example and disabled/manual-only public publishing templates. No Bioconda artifacts are created.

Before public release, the owner supplies/approves named authors, maintainer contact, repository/issue/docs URLs, citation/DOI, code and separate model/data rights. The current collective CITATION entry is not a substitute for that review. Public name availability, TestPyPI, remote CI, hosted docs, branch protections, OIDC environments, full deployment capsule and image-inference clean installation require separate verification.

Recommended branch settings: reviewed pull requests; require CPU contracts, package, docs and security checks; protect workflows with confirmed CODEOWNERS; prohibit force-push. The CODEOWNERS file is deliberately commented until actual GitHub account names are supplied.

Enable Trusted Publishing only after explicit release approval. No API tokens belong in this tree. Configured workflows are not proof they have run. Remote signed attestations and container digests cannot be fabricated locally.
