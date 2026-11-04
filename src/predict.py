"""Score customers with the saved pipeline: python -m src.predict data/sample_customers.csv"""
import sys
from pathlib import Path
import joblib
import pandas as pd

MODEL = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"


def score(df: pd.DataFrame, model_path: Path = MODEL) -> pd.DataFrame:
    art = joblib.load(model_path)
    missing = set(art["columns"]) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    p = art["pipeline"].predict_proba(df[art["columns"]])[:, 1]
    out = df.assign(churn_probability=p.round(4), churn_predicted=(p >= art["threshold"]).astype(int))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m src.predict <customers.csv>")
    res = score(pd.read_csv(sys.argv[1]))
    cols = [c for c in ("Customer ID",) if c in res] + ["churn_probability", "churn_predicted"]
    print(res[cols].to_string(index=False))
