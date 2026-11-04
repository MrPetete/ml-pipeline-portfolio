# Findings & Decisions

## Requirements (Codecademy ML Career Path portfolio project)

- End-to-end ML workflow in a Jupyter Notebook, then a productionalizable pipeline.
- Git version control, shared on personal GitHub, usable on CV for DS / MLE roles.
- Codecademy steps:
  1. Scope: goals, data, analysis (UChicago Data Project Scoping Guide).
  2. Pick an ML-solvable problem (churn, image classification, housing prices).
  3. Pick a dataset (UCI ML Repository, Data is Plural).
  4. Git repo with README.md, Jupyter Notebook, CSV data file(s).
  5. ETL: extract from multiple sources, transform with pandas, load.
  6. Feature engineering: train/test/validation split BEFORE any transformation.
  7. ML workflow: prep, features, multiple models, evaluation, tuning, validation (most time-consuming).
  8. Build scikit-learn Pipeline.
  9. Communicate findings: clear visuals, conclusion.
  10. Conclusion and next steps: takeaways, limitations, extensions.

## Technical Decisions

| Decision | Rationale |
|----------|-----------|
| Topic: telecom customer churn (Pete's choice) | Recognized by recruiters; multi-table data allows real ETL |
| Repo at D:/Huy/ml-pipeline-portfolio | C: is near full |
| Project venv on D: using `py` 3.13 | `python` is the Hermes venv |
| sklearn Pipeline + ColumnTransformer, joblib, predict CLI, pytest | Brief requires Pipeline; extras show production skills |
| Push to GitHub only after Pete confirms again | Pete's instruction |

## Issues Encountered

| Issue | Resolution |
|-------|------------|
| jupyter missing in py 3.13 | Install into project venv |
| write_file refused on template files (partial-read guard) | Wrote via shell |

## Resources

- UChicago Data Project Scoping Guide
- UCI ML Repository, Data is Plural
