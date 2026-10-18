# Clinical Validation Scope

This project implements a retrospective technical validation workflow on the
public Synapse dataset. It is not a clinical validation study.

## Implemented Evidence

- Model version and checkpoint path are recorded.
- Mean and per-class Dice/HD95 are parsed from a fixed test log.
- Acceptance thresholds are explicit and machine-checkable.
- The generated JSON/HTML report includes limitations and intended use.
- The report status distinguishes `technical_retrospective_validation` from
  `clinical_validation_status: not_performed`.

## Not Implemented

- Prospective clinical trial.
- Multi-site external validation.
- Reader study by qualified clinicians.
- Regulatory submission or medical-device certification.
- Patient-level bias and subgroup analysis.

## Recommended Clinical Study

1. Freeze the model version and validation manifest before data collection.
2. Use de-identified data from independent clinical sites.
3. Predefine inclusion/exclusion criteria and acceptance thresholds.
4. Evaluate Dice, HD95, organ-level failure rate, and inference latency.
5. Include blinded reader comparison and disagreement analysis.
6. Monitor subgroup performance and document failure cases.
7. Obtain ethics approval and regulatory guidance before any clinical use.
