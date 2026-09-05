# PHAxis 1.0.0 third-party notices

This inventory covers direct dependency declarations and build requirements in the generated PHAxis source release. It does not replace license texts shipped by resolved wheels, and it does not claim that a future resolver's transitive closure is unchanged.

| Distribution | Declared requirement | Scope(s) | License expression | Upstream | Note |
|---|---|---|---|---|---|
| build | `build>=1.2,<2` | build | `MIT` | https://pypi.org/project/build/ | - |
| imageio | `imageio>=2.35,<3` | deployment | `BSD-2-Clause` | https://pypi.org/project/imageio/ | - |
| joblib | `joblib>=1.4,<2` | deployment | `BSD-3-Clause` | https://pypi.org/project/joblib/ | - |
| matplotlib | `matplotlib>=3.8,<4` | analysis, deployment, test | `PSF-2.0` | https://pypi.org/project/matplotlib/ | - |
| numpy | `numpy>=1.26,<3` | core | `BSD-3-Clause` | https://pypi.org/project/numpy/ | - |
| opencv-python-headless | `opencv-python-headless>=4.9,<6` | analysis, deployment, inference, test, visualization | `LicenseRef-opencv-python-headless-wheel-multiple` | https://pypi.org/project/opencv-python-headless/ | Upstream reports the wrapper as MIT and OpenCV as Apache-2.0; published wheels also bundle FFmpeg under LGPL-2.1 and may contain additional artifact-specific notices. Preserve the wheel's LICENSE-3RD-PARTY.txt and audit the exact locked artifact. |
| packaging | `packaging>=24,<26` | core | `Apache-2.0 OR BSD-2-Clause` | https://pypi.org/project/packaging/ | - |
| pandas | `pandas>=2.2,<4` | analysis, deployment, test | `BSD-3-Clause` | https://pypi.org/project/pandas/ | - |
| Pillow | `Pillow>=10,<13` | analysis, deployment, inference, publication, test, visualization | `MIT-CMU` | https://pypi.org/project/Pillow/ | - |
| pytest | `pytest>=8,<10` | test | `MIT` | https://pypi.org/project/pytest/ | - |
| python-docx | `python-docx>=1.1,<2` | publication, test | `MIT` | https://pypi.org/project/python-docx/ | - |
| scikit-image | `scikit-image>=0.24,<0.27` | analysis, deployment, test | `BSD-3-Clause` | https://pypi.org/project/scikit-image/ | - |
| scikit-learn | `scikit-learn>=1.5,<2` | deployment | `BSD-3-Clause` | https://pypi.org/project/scikit-learn/ | - |
| scipy | `scipy>=1.11,<2` | core | `BSD-3-Clause` | https://pypi.org/project/scipy/ | - |
| setuptools | `setuptools>=77` | build-system | `MIT` | https://pypi.org/project/setuptools/ | - |
| statsmodels | `statsmodels>=0.14,<1` | analysis, deployment, test | `BSD-3-Clause` | https://pypi.org/project/statsmodels/ | - |
| tifffile | `tifffile>=2024.8,<2027` | analysis, deployment, inference, test | `BSD-3-Clause` | https://pypi.org/project/tifffile/ | - |
| timm | `timm>=1.0.28,<2` | deployment, inference, test | `Apache-2.0` | https://pypi.org/project/timm/ | - |
| torch | `torch>=2.6,<3` | deployment, inference, test | `BSD-3-Clause` | https://pypi.org/project/torch/ | - |
| torchvision | `torchvision>=0.21,<1` | deployment, inference, test | `BSD-3-Clause` | https://pypi.org/project/torchvision/ | - |
| twine | `twine>=6,<7` | build | `Apache-2.0` | https://pypi.org/project/twine/ | - |
| wheel | `wheel>=0.45` | build-system | `MIT` | https://pypi.org/project/wheel/ | - |

## Vendored fallback

PHAxis vendors the unmodified pure-Python source files from Tomli 2.4.0 for Python 3.10 and isolated no-site source verification. The exact files and SHA-256 values are recorded in THIRD_PARTY_LICENSES.json and SBOM.cdx.json. The complete MIT license is retained at `src/phaxis/_vendor/tomli/LICENSE.txt` and in the wheel.

The exact platform-specific deployment wheelhouse is materialized later by the formal dependency-lock stage with one SHA-256 per artifact. Users and redistributors must retain each artifact's upstream license and notice files.
