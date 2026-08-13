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
DATA_PATH = PROJECT_ROOT / "data" / "housing.csv"
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "eda_outputs"
REQUIRED_COLUMNS = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "location",
    "age_years",
    "price",
]


def generate_synthetic_housing_data(path: Path, rows: int = 500) -> pd.DataFrame:
    """Create a believable synthetic housing dataset if the real CSV is missing."""
    rng = np.random.default_rng(42)

    locations = ["Downtown", "Suburban", "Riverside", "Hillcrest", "Old Town"]
    location_effects = {
        "Downtown": 85000,
        "Suburban": 55000,
        "Riverside": 62000,
        "Hillcrest": 95000,
        "Old Town": 30000,
    }

    area_sqft = np.clip(rng.normal(2200, 500, size=rows), 800, 5000)
    bedrooms = rng.integers(1, 6, size=rows)
    bathrooms = rng.integers(1, 5, size=rows)
    age_years = rng.integers(0, 60, size=rows)
    location = rng.choice(locations, size=rows)

    # A realistic formula for home price with market variation and noise.
    base_price = (
        180.0 * area_sqft
        + 35000.0 * bedrooms
        + 25000.0 * bathrooms
        + np.array([location_effects[loc] for loc in location])
        - 1200.0 * age_years
        + 55000.0
    )
    noise = rng.normal(0, 40000, size=rows)
    price = base_price + noise
    price = np.maximum(price, 100000.0)

    df = pd.DataFrame(
        {
            "area_sqft": area_sqft.round(1),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "location": location,
            "age_years": age_years,
            "price": np.round(price, 2),
        }
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def ensure_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"Dataset not found at {path}. Generating synthetic data...")
        return generate_synthetic_housing_data(path)

    df = pd.read_csv(path)
    if not set(REQUIRED_COLUMNS).issubset(df.columns):
        print(
            f"Dataset at {path} does not match the required schema. "
            "Generating a synthetic housing dataset instead."
        )
        return generate_synthetic_housing_data(path)
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
    corr_with_price = corr_df["price"].drop("price").sort_values(ascending=False)
    print("\nCorrelation with price:")
    print(corr_with_price)
    return corr_with_price


def save_price_distribution_plot(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(df["price"], bins=30, color="#4c72b0", edgecolor="black")
    plt.title("Housing Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_price_vs_area_plot(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["area_sqft"], df["price"], alpha=0.7, s=30, color="#55a868")
    plt.title("Price vs Area")
    plt.xlabel("Area (sqft)")
    plt.ylabel("Price")
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
