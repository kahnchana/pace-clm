# Manifests

Deterministic train/dev/test manifests belong here. Do not create WEB-50 or
EGO-50 pilot manifests until the PACE-200 design has been frozen.

The exp10 JSONL cohorts and gold JSON files are generated local data and are
ignored by Git. Commit `exp10/config.v1.json`, `exp10/freeze.json`, and the
small `exp10/*.freeze.json` records to preserve settings and file hashes.
The freeze scripts recreate the ignored files from their documented source data
and local outputs; keep the generated files locally for exp10 runs.
