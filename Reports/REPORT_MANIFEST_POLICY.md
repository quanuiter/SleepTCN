# Report manifest policy

`REPORT_MANIFEST.sha256` covers files that are present in this repository and
the reproducibility notes that accompany them.  The historical Gate-5 and
multiseed JSON outputs have been restored from the project Git history and are
listed as verifiable entries again.  The Gate-7/Gate-8 package files are
validated separately by their package validators and are not duplicated in
this report-level manifest.

Every listed path is relative to `Reports/`.  A manifest entry is valid only
when the path exists and its SHA-256 digest matches the recorded value.
