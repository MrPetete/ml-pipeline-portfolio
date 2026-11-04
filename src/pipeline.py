"""Feature definitions and the scikit-learn Pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
SEED = 42
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


def split_test(df):
    """Hold out 20% as the test set. Every notebook must use this, or test rows leak into training."""
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)  # X_tv, X_test, y_tv, y_test


def split_val(X_tv, y_tv):
    """Hold out 25% of the rest (20% of all rows) for validation."""
    return train_test_split(X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=SEED)  # X_train, X_val, y_train, y_val
