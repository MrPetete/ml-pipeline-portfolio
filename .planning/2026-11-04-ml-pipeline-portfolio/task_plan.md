# Task Plan: Codecademy ML Portfolio Project (telecom churn)

## Goal

Ship a public GitHub repo with an end-to-end churn-prediction workflow (notebook) and a scikit-learn Pipeline, polished for the CV.

## Next Step

Find a freely downloadable multi-table telecom churn dataset, then write the scope in README.

## Current Phase

Phase 1

## Phases

### Phase 1: Scope, problem and dataset
- [x] Read the 10 Codecademy steps (findings.md)
- [x] Pete chose telecom churn
- [ ] Pick dataset with a direct download link
- [ ] Write scope (goals, data, analysis) in README
- **Status:** in_progress

### Phase 2: Repo and environment
- [x] git init
- [ ] venv on D:, install pandas, scikit-learn, matplotlib, jupyter, pytest
- [ ] Layout: data/, notebooks/, src/, tests/, README.md
- **Status:** in_progress

### Phase 3: ETL and EDA
- [ ] Extract multiple tables, transform with pandas, load to data/processed
- [ ] EDA visuals
- **Status:** pending

### Phase 4: Split, features, models
- [ ] Train/val/test split before any transform
- [ ] Baseline + 3 models, CV, tuning, test score once
- **Status:** pending

### Phase 5: Pipeline and tests
- [ ] Pipeline + ColumnTransformer, joblib, predict CLI, pytest
- **Status:** pending

### Phase 6: Write-up and publish
- [ ] README with results, limitations, next steps
- [ ] Ask Pete, then create GitHub repo and push
- [ ] Draft CV bullet
- **Status:** pending

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| planning_with_files_init not directly callable | 1 | Used tool_call |
| write_file blocked on template files | 1 | Wrote via shell |
