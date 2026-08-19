#!/usr/bin/env python3
"""Train and compare regression models for house price prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "house_prices_srilanka.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "price_model.joblib"
COLUMNS_PATH = MODEL_DIR / "model_columns.joblib"

TARGET_COLUMN = "price_lkr"
CATEGORICAL_COLUMNS = [
    "district",
    "area",
    "parking_spots",
    "has_garden",
    "has_ac",
    "water_supply",
    "electricity",
    "floors",
    "year_built",
]
# price_lkr is the TARGET and must never appear here. Including it leaks the
# answer into the features, and the model degenerates to returning its own
# input (R2 = 1.0, zero error) while learning nothing about the other columns.
NUMERIC_COLUMNS = ["kitchen_area_sqft", "bedrooms", "bathrooms", "perch"]

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run src/eda.py first to generate it."
        )

    df = pd.read_csv(path)
    required = set(NUMERIC_COLUMNS + CATEGORICAL_COLUMNS + [TARGET_COLUMN])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset at {path} is missing columns: {sorted(missing)}")
    return df


def fit_encoder(train_df: pd.DataFrame) -> OneHotEncoder:
    """Fit the one-hot encoder for the categorical columns on the training split."""
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(train_df[CATEGORICAL_COLUMNS])
    return encoder


def encode_features(df: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame:
    """Combine the numeric columns with the one-hot encoded categorical columns."""
    encoded = encoder.transform(df[CATEGORICAL_COLUMNS])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(CATEGORICAL_COLUMNS),
        index=df.index,
    )
    return pd.concat([df[NUMERIC_COLUMNS], encoded_df], axis=1)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    predictions = model.predict(X_test)
    return {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": r2_score(y_test, predictions),
    }


def print_comparison_table(scores: dict[str, dict[str, float]]) -> None:
    header = f"{'Model':<24}{'MAE':>14}{'RMSE':>14}{'R²':>10}"
    print("\nTest set performance")
    print(header)
    print("-" * len(header))
    for name, metrics in scores.items():
        print(
            f"{name:<24}{metrics['MAE']:>14,.2f}"
            f"{metrics['RMSE']:>14,.2f}{metrics['R2']:>10.4f}"
        )


def print_feature_importances(
    model: RandomForestRegressor, feature_names: list[str]
) -> None:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False)

    print("\nRandom Forest feature importances (descending)")
    for name, value in importances.items():
        print(f"{name:<24}{value:>10.4f}")


def main() -> None:
    df = load_dataset(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")

    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # The encoder is fitted on the training split only, then reused for the test
    # split so both end up with identical columns in identical order.
    encoder = fit_encoder(X_train_raw)
    X_train = encode_features(X_train_raw, encoder)
    X_test = encode_features(X_test_raw, encoder)
    model_columns = X_train.columns.tolist()
    print(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")
    print(f"Features after encoding: {model_columns}")

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    forest_model = RandomForestRegressor(random_state=RANDOM_STATE)
    forest_model.fit(X_train, y_train)

    models = {
        "Linear Regression": linear_model,
        "Random Forest": forest_model,
    }
    scores = {name: evaluate(model, X_test, y_test) for name, model in models.items()}
    print_comparison_table(scores)

    # Lower RMSE wins; it penalises the large errors that matter most for prices.
    best_name = min(scores, key=lambda name: scores[name]["RMSE"])
    best_model = models[best_name]
    print(
        f"\nBest model: {best_name} "
        f"(RMSE {scores[best_name]['RMSE']:,.2f}, R² {scores[best_name]['R2']:.4f})"
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(
        {
            "encoder": encoder,
            "model_columns": model_columns,
            "numeric_columns": NUMERIC_COLUMNS,
            "categorical_columns": CATEGORICAL_COLUMNS,
        },
        COLUMNS_PATH,
    )
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved encoder and column list to: {COLUMNS_PATH}")

    print_feature_importances(forest_model, model_columns)


if __name__ == "__main__":
    main()
