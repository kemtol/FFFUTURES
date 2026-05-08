# Super Structure ML

Versioned research space for meta-modeling on top of Super Structure core signals.

## Layout

- `sources/` raw inputs and extraction helpers
- `features/` feature builders for entry-time and context features
- `labels/` label builders and target definitions
- `datasets/` versioned parquet outputs
- `train/` model training entry points
- `eval/` walkforward and holdout evaluation
- `reports/` generated metrics and comparison summaries
- `artifacts/` model-ready exports, thresholds, and config snapshots

## Rule

Keep the live strategy untouched. Add new research code here with explicit versioning.
