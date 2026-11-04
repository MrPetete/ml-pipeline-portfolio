"""Extract the 5 raw IBM Telco tables, merge on customer ID, load a clean CSV."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "churn.csv"

# Status-table columns Churn Score, Churn Reason/Category, Customer Status, CLTV and
# Satisfaction Score are left out on purpose: they are recorded with or after the outcome (target leakage).


def extract(raw: Path = RAW) -> dict:
    names = ["demographics", "location", "population", "services", "status"]
    return {n: pd.read_csv(raw / f"Telco_customer_churn_{n}.csv") for n in names}


def transform(t: dict) -> pd.DataFrame:
    for n in ("demographics", "location", "services", "status"):
        assert not t[n]["Customer ID"].duplicated().any(), f"duplicate IDs in {n}"
    df = t["status"][["Customer ID", "Churn Value"]]
    for n, drop in (("demographics", ["Count"]), ("services", ["Service ID", "Count", "Quarter"]),
                    ("location", ["Location ID", "Count", "Country", "State", "Lat Long"])):
        df = df.merge(t[n].drop(columns=drop), on="Customer ID", how="left", validate="1:1")
    df = df.merge(t["population"][["Zip Code", "Population"]].drop_duplicates("Zip Code"),
                  on="Zip Code", how="left", validate="m:1")
    df = df.rename(columns={"Churn Value": "Churn"})
    df["Offer"] = df["Offer"].fillna("No Offer")
    df["Internet Type"] = df["Internet Type"].fillna("No Internet")
    assert len(df) == len(t["status"]) and df["Churn"].isin([0, 1]).all()
    return df


def load(df: pd.DataFrame, out: Path = OUT) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    df = transform(extract())
    print(f"wrote {load(df)} shape={df.shape} churn_rate={df['Churn'].mean():.3f}")
