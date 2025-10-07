from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent      # points to dags/
DATA_DIR = ROOT / "data"
WORK_DIR = ROOT / "working_data"
MODEL_DIR = ROOT / "model"

for d in (WORK_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)


def load_reg_data() -> str:
    """
    Load the training CSV and persist it as Parquet.
    Returns: path to the parquet file (str) for downstream steps.
    """
    src = DATA_DIR / "file.csv"
    df = pd.read_csv(src)
    required = {"BALANCE", "PURCHASES", "CREDIT_LIMIT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in file.csv: {missing}")
    out = WORK_DIR / "train_raw.csv"
    df.to_csv(out, index=False)
    return str(out)


def preprocess_reg(train_parquet_path: str) -> tuple[str, str, str]:
    import pandas as pd, numpy as np, pickle
    from sklearn.preprocessing import StandardScaler
    from pathlib import Path

    # Read (supports either csv/parquet)
    if train_parquet_path.endswith(".parquet"):
        df = pd.read_parquet(train_parquet_path)
    else:
        df = pd.read_csv(train_parquet_path)

    # Coerce to numeric in case CSV had strings; invalids -> NaN
    for col in ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where target or any feature is missing
    before = len(df)
    df = df.dropna(subset=["BALANCE", "PURCHASES", "CREDIT_LIMIT"])
    after = len(df)
    if after == 0:
        raise ValueError(
            "No training rows left after cleaning; all rows had NaNs in "
            "[BALANCE, PURCHASES, CREDIT_LIMIT]."
        )
    print(f"[PREPROCESS] dropped {before - after} rows with NaNs; kept {after}")

    X = df[["BALANCE", "PURCHASES"]].astype("float64").to_numpy()
    y = df["CREDIT_LIMIT"].astype("float64").to_numpy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    from pathlib import Path
    from .simple_regression import WORK_DIR  # keep your constants setup

    X_path = WORK_DIR / "reg_train_X.npy"
    y_path = WORK_DIR / "reg_train_y.npy"
    np.save(X_path, Xs)
    np.save(y_path, y)

    scaler_path = WORK_DIR / "reg_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print(f"[PREPROCESS] X:{Xs.shape} y:{y.shape} saved -> {X_path}, {y_path}")
    return str(X_path), str(y_path), str(scaler_path)


def train_reg_model(X_path: str, y_path: str, filename: str = "reg_model.sav") -> tuple[str, float]:
    """
    Fit Linear Regression, save it, and report train R^2.
    Returns: (model_path, r2_train)
    """
    Xs = np.load(X_path)
    y = np.load(y_path)

    model = LinearRegression()
    model.fit(Xs, y)
    r2_train = float(model.score(Xs, y))

    model_path = MODEL_DIR / filename
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return str(model_path), r2_train


def predict_reg(model_path: str, scaler_path: str) -> float:
    """
    Predict CREDIT_LIMIT for the first row of test.csv (after scaling).
    Returns: predicted value (float) for the first test row.
    """
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    test_df = pd.read_csv(DATA_DIR / "test.csv").dropna(subset=["BALANCE", "PURCHASES"])
    Xt = test_df[["BALANCE", "PURCHASES"]].to_numpy(dtype=float)
    Xt_scaled = scaler.transform(Xt)
    preds = model.predict(Xt_scaled)
    return float(preds[0])
