#!/usr/bin/env python3
"""Exploratory data analysis for the housing dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# DATA_PATH = PROJECT_ROOT / "data" / "housing.csv"
DATA_PATH = PROJECT_ROOT / "data" / "house_prices_srilanka.csv"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "eda_outputs"
REQUIRED_COLUMNS = [
    "perch",
    "bedrooms",
    "bathrooms",
    "district",
    "year_built",
    "price_lkr",
]


def ensure_dataset(path: Path) -> pd.DataFrame:
    """Load the dataset, or fail with the schema mismatch spelled out.

    An earlier version fell back to generating a synthetic dataset here. That
    was removed: the generator emitted an older schema (area_sqft / location /
    price) that does not satisfy REQUIRED_COLUMNS, so every function below it
    still failed - and because it wrote to `path`, the fallback would overwrite
    the real 20k-row CSV with 500 rows of synthetic data.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}.\n"
            f"Expected a CSV with at least these columns: "
            f"{', '.join(REQUIRED_COLUMNS)}."
        )

    df = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset at {path} is missing required columns: {missing}.\n"
            f"Found: {list(df.columns)}"
        )
    return df


def print_dataset_overview(df: pd.DataFrame) -> None:
    print("Data shape:")
    print(df.shape)
    print("\nData types:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nSummary statistics:")
    print(df.describe(include="all").transpose())


def compute_price_correlations(df: pd.DataFrame) -> pd.Series:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_df = df[numeric_cols].corr(numeric_only=True)
    corr_with_price = corr_df["price_lkr"].drop("price_lkr").sort_values(ascending=False)
    print("\nCorrelation with price:")
    print(corr_with_price)
    return corr_with_price


def save_price_distribution_plot(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["price_lkr"], bins=30, color="#77c012", edgecolor="black")
    plt.title("Housing Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_price_vs_area_plot(df: pd.DataFrame, output_path: Path) -> None:
    # 'perch' is the land extent, and the only real size measure in this
    # dataset - it is also how Sri Lankan listings quote size. This plot used
    # kitchen_area_sqft, which is the kitchen alone (35-250 sqft) and says
    # nothing about how big the property is.
    plt.figure(figsize=(8, 6))
    plt.scatter(df["perch"], df["price_lkr"], alpha=0.4, s=12, color="#920c75")
    plt.title("Price vs Land Extent")
    plt.xlabel("Land extent (perches)")
    plt.ylabel("Price (LKR)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)

    plt.figure(figsize=(9, 7))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            value = corr.iloc[i, j]
            plt.text(j, i, f"{value:.2f}", ha="center", va="center", color="black")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    df = ensure_dataset(DATA_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print_dataset_overview(df)
    compute_price_correlations(df)

    save_price_distribution_plot(df, OUTPUT_DIR / "price_distribution.png")
    save_price_vs_area_plot(df, OUTPUT_DIR / "price_vs_area_scatter.png")
    save_correlation_heatmap(df, OUTPUT_DIR / "correlation_heatmap.png")

    print(f"\nSaved plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
