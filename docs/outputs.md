# Outputs and data dictionary

The pipeline preserves root_provider/, fusion/predictions/, traits/ and distal_axis_profiles/ plus versioned JSON receipts. [The user guide](phaxis/USER_GUIDE.md) lists exact filenames. [The trait contract](phaxis/TRAIT_CONTRACT_CN.md) defines units and missing/censored values for all 32 descriptors.

The additive `phaxis report` command reads existing trait CSVs without altering them. It writes report.html, report_data.json, tables/*.csv, README_results.md and provenance.json to a new directory. The JSON schema is PHAxis-offline-report-1.0. CSV downloads are copied byte-for-byte; SHA-256 in provenance.json records each copied input. Reports omit raw source paths and do not use external CDNs or telemetry.

Full descriptor records may include QC/eligibility flags, reasons and support counts. A missing endpoint-supported length is not a zero. Do not treat per-hair rows as independent biological replicates.
