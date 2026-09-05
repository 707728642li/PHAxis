# Contributing to PHAxis

PHAxis welcomes focused improvements to Arabidopsis root and root-hair
measurement, reproducibility, documentation, and usability. Open an issue
before a large change so its biological measurement contract and evidence plan
are explicit.

Create an isolated Python environment, then run:

```console
python -B -m pip install -e ".[test]"
python -B -m pytest tests/phaxis -q
```

Pull requests should include tests, exact commands, and the affected
measurement/provenance contract. Never submit private images, annotations,
weights, credentials, local machine paths, or blind/final-validation material.
Condition labels and biological outcomes must not be introduced into model
routing or threshold selection.
