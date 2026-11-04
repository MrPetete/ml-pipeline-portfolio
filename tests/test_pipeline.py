import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.etl import RAW, extract, transform
from src.pipeline import TARGET, make_pipeline, split_columns

needs_data = pytest.mark.skipif(not (RAW / "Telco_customer_churn_status.csv").exists(), reason="raw data missing")


@needs_data
def test_etl_is_one_row_per_customer_without_leakage():
    df = transform(extract())
    assert df["Customer ID"].is_unique and len(df) == 7043
    assert df.isna().sum().sum() == 0
    leaky = {"Churn Score", "Churn Reason", "Churn Category", "Customer Status", "Churn Label", "CLTV", "Satisfaction Score"}
    assert not leaky & set(df.columns)


@needs_data
def test_pipeline_handles_missing_and_unseen_category():
    df = transform(extract()).sample(1500, random_state=0)
    num, cat = split_columns(df)
    pipe = make_pipeline(LogisticRegression(max_iter=1000), num, cat)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    pipe.fit(X, y)
    new = X.head(5).copy()
    new.loc[new.index[0], "Contract"] = "Brand New Plan"
    new.loc[new.index[1], num[0]] = np.nan
    p = pipe.predict_proba(new)[:, 1]
    assert p.shape == (5,) and ((p >= 0) & (p <= 1)).all()
