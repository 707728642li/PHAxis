# Five-minute CPU tutorial

After installing the local wheel:

```console
phaxis --version
phaxis demo --output demo-results
phaxis report --traits demo-results/traits --output second-report
```

Open either report.html offline. Inspect traits/image_traits.csv (82 fields, 32 phenotype fields), traits/hair_instances.csv and demo_receipt.json. The demo has one source-root fixture with two visible identities. Golden tests independently check count=2 and zero-hair count=0; unsupported length remains missing.

Every new result directory must be absent. Repeated runs go to new directories; analytical tables should match. Timing/provenance timestamps need not match. No CUDA or external data are accessed.
