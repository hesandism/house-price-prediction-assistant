#!/usr/bin/env python3
"""LangChain tools for the house price assistant: price prediction and doc retrieval."""

from __future__ import annotations

import datetime as dt
import sys
from functools import lru_cache
from pathlib import Path

import chromadb
import joblib
import pandas as pd
from chromadb.config import Settings
from langchain_core.tools import tool

try:  # langchain-community is being sunset; prefer the standalone package if present.
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "price_model.joblib"
COLUMNS_PATH = PROJECT_ROOT / "models" / "model_columns.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "house_prices_srilanka.csv"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Both collections share one embedding model and cosine space, so their
# distances are directly comparable and results can be merged and re-ranked.
COLLECTION_NAMES = ("housing_docs", "markdown_docs")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 3

TARGET_COLUMN = "price_lkr"

# The tool signature the agent sees is deliberately simpler than the training
# schema. These map the five agent-facing arguments onto real training columns;
# every other column is filled from the dataset's central tendency.
LOCATION_COLUMN = "district"
SUBAREA_COLUMN = "area"
AREA_SQFT_COLUMN = "kitchen_area_sqft"
YEAR_BUILT_COLUMN = "year_built"


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[object, dict]:
    """Load the fitted model and the encoder/column bundle saved by train_model.py."""
    for path in (MODEL_PATH, COLUMNS_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {path}. Run src/train_model.py first."
            )
    return joblib.load(MODEL_PATH), joblib.load(COLUMNS_PATH)


@lru_cache(maxsize=1)
def _load_defaults() -> tuple[dict, pd.DataFrame]:
    """Median (numeric) and modal (categorical) values for columns the agent can't supply."""
    model, bundle = _load_artifacts()
    df = pd.read_csv(DATA_PATH)

    defaults: dict[str, object] = {}
    for column in bundle["numeric_columns"]:
        if column != TARGET_COLUMN:
            defaults[column] = float(df[column].median())
    for column in bundle["categorical_columns"]:
        defaults[column] = df[column].mode().iat[0]
    return defaults, df


def _resolve_location(location: str, known: list[str]) -> str | None:
    """Match a free-text location to a district the encoder was fitted on."""
    cleaned = location.strip().casefold()
    for candidate in known:
        if str(candidate).casefold() == cleaned:
            return candidate
    # Tolerate "Colombo District" / "in Colombo" style phrasing from the LLM.
    for candidate in known:
        if str(candidate).casefold() in cleaned:
            return candidate
    return None


@lru_cache(maxsize=1)
def _load_embedder() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        # Must match ingest_docs.py / build_vectorstore.py, or cosine distances
        # against the stored vectors are meaningless.
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _open_collections() -> tuple[chromadb.Collection, ...]:
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"No Chroma store at {CHROMA_DIR}. Run src/ingest_docs.py first."
        )
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )
    available = {c.name for c in client.list_collections()}
    return tuple(
        client.get_collection(name)
        for name in COLLECTION_NAMES
        if name in available
    )


@tool
def predict_price(
    area_sqft: float,
    bedrooms: int,
    bathrooms: int,
    location: str,
    age_years: int,
) -> str:
    """Estimate the market price of a Sri Lankan house from its physical attributes.

    Call this for any question asking "how much", "what would it cost", "what is
    it worth", or "estimate the price" about a SPECIFIC property whose size,
    room counts, district and age are known or can be inferred from the question.
    This returns a single number from a trained regression model - it does not
    explain market conditions, trends, or reasons.

    Do NOT call this for general questions about the market, price trends, why
    prices moved, or what reports say - use retrieve_docs for those.

    Args:
        area_sqft: Built floor area in square feet, e.g. 1200.
        bedrooms: Number of bedrooms, e.g. 3.
        bathrooms: Number of bathrooms, e.g. 2.
        location: Sri Lankan district name, e.g. "Colombo", "Kandy", "Galle".
        age_years: Age of the property in years, e.g. 10 for a house built a decade ago.

    Returns:
        A formatted price estimate in LKR, echoing the inputs it was based on.
    """
    model, bundle = _load_artifacts()

    # train_model.py lists price_lkr in NUMERIC_COLUMNS, which are used as
    # features - so the fitted model takes the sale price as an input and
    # returns it unchanged. No honest prediction can be made from this
    # artifact, and quietly substituting a median would return a lookup
    # dressed up as a prediction. Fail loudly instead.
    if TARGET_COLUMN in bundle["numeric_columns"]:
        return (
            "Prediction unavailable: the saved model is invalid. It was trained "
            f"with '{TARGET_COLUMN}' (the sale price) as an input feature, so it "
            "only ever returns the price it is given. Fix: remove "
            f"'{TARGET_COLUMN}' from NUMERIC_COLUMNS in src/train_model.py, "
            "re-run it to retrain, then retry. Do not report a price to the user."
        )

    defaults, df = _load_defaults()
    districts = [str(c) for c in bundle["encoder"].categories_[
        bundle["categorical_columns"].index(LOCATION_COLUMN)
    ]]

    district = _resolve_location(location, districts)
    if district is None:
        return (
            f"Unknown location '{location}'. This model only covers Sri Lankan "
            f"districts. Valid values: {', '.join(sorted(districts))}."
        )

    if area_sqft <= 0 or bedrooms <= 0 or bathrooms <= 0 or age_years < 0:
        return (
            "Invalid input: area_sqft, bedrooms and bathrooms must be positive "
            "and age_years cannot be negative."
        )

    row = dict(defaults)
    row[AREA_SQFT_COLUMN] = float(area_sqft)
    row["bedrooms"] = int(bedrooms)
    row["bathrooms"] = int(bathrooms)
    row[LOCATION_COLUMN] = district
    row[YEAR_BUILT_COLUMN] = dt.date.today().year - int(age_years)

    # The sub-area column is nested inside the district, so the global mode
    # would usually contradict the district the caller asked for.
    in_district = df.loc[df[LOCATION_COLUMN] == district, SUBAREA_COLUMN]
    if not in_district.empty:
        row[SUBAREA_COLUMN] = in_district.mode().iat[0]

    frame = pd.DataFrame([row])
    encoded = bundle["encoder"].transform(frame[bundle["categorical_columns"]])
    encoded_df = pd.DataFrame(
        encoded,
        columns=bundle["encoder"].get_feature_names_out(bundle["categorical_columns"]),
        index=frame.index,
    )
    features = pd.concat([frame[bundle["numeric_columns"]], encoded_df], axis=1)
    features = features.reindex(columns=bundle["model_columns"], fill_value=0.0)

    price = float(model.predict(features)[0])

    return (
        f"Estimated price: LKR {price:,.0f}\n"
        f"Based on: {area_sqft:,.0f} sqft, {bedrooms} bed, {bathrooms} bath, "
        f"{district} district, {age_years} years old."
    )


@tool
def retrieve_docs(query: str) -> str:
    """Search Central Bank of Sri Lanka housing and financial stability reports.

    Call this for explanatory or factual questions about the Sri Lankan property
    market - trends, drivers, policy, interest rates, what happened in a given
    year or quarter, or anything phrased as "why", "how has", "what do the
    reports say", or "explain". It returns excerpts from source documents, which
    you should summarise and cite by filename.

    Do NOT call this to compute a price for a specific house - use predict_price
    for that. Do call it alongside predict_price when the user wants an estimate
    AND the market reasoning behind it.

    Args:
        query: A natural-language search phrase, e.g. "Colombo apartment price
            trends 2024" or "drivers of housing demand".

    Returns:
        The top matching excerpts, each labelled with its source filename.
    """
    if not query or not query.strip():
        return "Empty query - provide a search phrase describing what to look for."

    collections = _open_collections()
    if not collections:
        return (
            f"No document collections found in {CHROMA_DIR}. "
            "Run src/ingest_docs.py (PDFs) or src/build_vectorstore.py (Markdown)."
        )

    vector = _load_embedder().embed_query(query)

    hits: list[tuple[float, str, dict]] = []
    for collection in collections:
        result = collection.query(
            query_embeddings=[vector],
            n_results=min(TOP_K, collection.count()),
        )
        if not result["documents"] or not result["documents"][0]:
            continue
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append((distance, text, meta or {}))

    if not hits:
        return f"No relevant passages found for: {query}"

    # Shared embedding model and cosine space, so distances merge across collections.
    hits.sort(key=lambda hit: hit[0])

    blocks = []
    for rank, (distance, text, meta) in enumerate(hits[:TOP_K], start=1):
        source = meta.get("source", "unknown source")
        page = meta.get("page")
        label = f"{source} (page {page})" if page else source
        blocks.append(
            f"[{rank}] {label}  |  relevance {1 - distance:.3f}\n"
            f"{' '.join(text.split())}"
        )

    return "\n\n".join(blocks)


TOOLS = [predict_price, retrieve_docs]


if __name__ == "__main__":
    # Report text carries typographic characters (U+2010 hyphens, en dashes)
    # that a cp1252 Windows console cannot encode. Degrade instead of crashing.
    sys.stdout.reconfigure(errors="replace")

    print(retrieve_docs.invoke({"query": "Colombo house price trends"}))
    print()
    print(
        predict_price.invoke(
            {
                "area_sqft": 1800,
                "bedrooms": 3,
                "bathrooms": 2,
                "location": "Colombo",
                "age_years": 10,
            }
        )
    )
