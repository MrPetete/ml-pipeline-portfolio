"""Feature definitions and the scikit-learn Pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
DROP = ["Customer ID", "Zip Code", "City", "Latitude", "Longitude"]  # identifiers / high-cardinality


def split_columns(df: pd.DataFrame):
    X = df.drop(columns=[TARGET] + [c for c in DROP if c in df])
    cat = X.select_dtypes(exclude="number").columns.tolist()
    num = X.select_dtypes(include="number").columns.tolist()
    return num, cat


def make_preprocessor(num, cat) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def make_pipeline(model, num, cat) -> Pipeline:
    return Pipeline([("prep", make_preprocessor(num, cat)), ("model", model)])
