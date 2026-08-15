#!/usr/bin/env python3
"""Chunk the PDFs in docs/ and embed them into a persistent Chroma collection."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "housing_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ~1000 characters is roughly 200-250 tokens, which sits comfortably inside the
# 256-token window of MiniLM. The overlap keeps a sentence that straddles a
# chunk boundary retrievable from either side.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Pages below this many characters are cover pages, blank pages or pure charts.
# Embedding them only adds noise to the retrieval results.
MIN_PAGE_CHARS = 100

EMBED_BATCH_SIZE = 256
CHROMA_BATCH_SIZE = 500

MARKET_ANALYSIS_RE = re.compile(r"real_estate_market_analysis_(\d{4})_q(\d)", re.I)
FSR_RE = re.compile(r"fsr_(\d{4})", re.I)


@dataclass
class Page:
    """One extracted PDF page, before splitting."""

    source: str
    page_number: int
    text: str


def describe_document(filename: str) -> dict[str, str | int]:
    """Turn the filename convention into metadata we can filter on later."""
    stem = Path(filename).stem

    market = MARKET_ANALYSIS_RE.search(stem)
    if market:
        year, quarter = int(market.group(1)), int(market.group(2))
        return {
            "doc_type": "market_analysis",
            "publisher": "Central Bank of Sri Lanka",
            "year": year,
            "quarter": quarter,
            "period": f"{year} Q{quarter}",
            "title": f"Real Estate Market Analysis {year} Q{quarter}",
        }

    fsr = FSR_RE.search(stem)
    if fsr:
        year = int(fsr.group(1))
        is_summary = "summary" in stem.lower()
        return {
            "doc_type": "financial_stability_review",
            "publisher": "Central Bank of Sri Lanka",
            "year": year,
            "quarter": 0,  # Annual report; Chroma metadata cannot hold None.
            "period": str(year),
            "title": (
                f"Financial Stability Review {year}"
                + (" (Summary)" if is_summary else "")
            ),
        }

    return {
        "doc_type": "other",
        "publisher": "unknown",
        "year": 0,
        "quarter": 0,
        "period": "unknown",
        "title": stem,
    }


def normalise(text: str) -> str:
    """Collapse the ragged whitespace that PDF extraction leaves behind."""
    return re.sub(r"\s+", " ", text).strip()


def extract_pages(pdf_path: Path) -> tuple[list[Page], int, int]:
    """Extract every page of a PDF, returning (pages, total_pages, skipped)."""
    reader = PdfReader(str(pdf_path))
    pages: list[Page] = []
    skipped = 0

    for index, page in enumerate(reader.pages, start=1):
        text = normalise(page.extract_text() or "")
        if len(text) < MIN_PAGE_CHARS:
            skipped += 1
            continue
        pages.append(Page(source=pdf_path.name, page_number=index, text=text))

    return pages, len(reader.pages), skipped


def chunk_pages(
    pages: list[Page], splitter: RecursiveCharacterTextSplitter
) -> tuple[list[str], list[dict], list[str]]:
    """Split pages into chunks and build a parallel id / metadata list."""
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for page in pages:
        base_metadata = describe_document(page.source)
        for position, chunk in enumerate(splitter.split_text(page.text)):
            documents.append(chunk)
            metadatas.append(
                {
                    **base_metadata,
                    "source": page.source,
                    "page": page.page_number,
                    "chunk": position,
                    "chars": len(chunk),
                }
            )
            # A deterministic id means re-running the script upserts the same
            # chunk instead of creating a duplicate copy of the corpus.
            ids.append(f"{Path(page.source).stem}-p{page.page_number}-c{position}")

    return documents, metadatas, ids


def get_collection(rebuild: bool) -> chromadb.Collection:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Dropped existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        # Cosine matches the normalised embeddings MiniLM produces.
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )


def embed(model: SentenceTransformer, documents: list[str]) -> list[list[float]]:
    vectors = model.encode(
        documents,
        batch_size=EMBED_BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return vectors.tolist()


def store(
    collection: chromadb.Collection,
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    """Write in batches; Chroma rejects a single oversized add()."""
    for start in range(0, len(documents), CHROMA_BATCH_SIZE):
        end = start + CHROMA_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )


def print_per_document_table(rows: list[dict]) -> None:
    header = f"{'Document':<44}{'Pages':>7}{'Skipped':>9}{'Chunks':>8}"
    print("\nPer-document breakdown")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['source']:<44}{row['pages']:>7}"
            f"{row['skipped']:>9}{row['chunks']:>8}"
        )


def smoke_test(collection: chromadb.Collection, model: SentenceTransformer) -> None:
    """Retrieve against the fresh index so the run proves the store is usable."""
    question = "How did house asking prices in Colombo change?"
    vector = model.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=vector, n_results=3)

    print(f"\nSmoke-test query: {question!r}")
    for document, metadata, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        preview = document[:160].replace("\n", " ")
        print(f"  [{1 - distance:.3f}] {metadata['source']} p.{metadata['page']}")
        print(f"        {preview}...")


def main() -> None:
    # PDF text carries typographic characters (U+2010 hyphens, en dashes) that a
    # cp1252 Windows console cannot encode. Degrade them instead of crashing.
    sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop the collection before ingesting instead of upserting into it.",
    )
    args = parser.parse_args()

    pdf_paths = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {DOCS_DIR}")
    print(f"Found {len(pdf_paths)} PDFs in {DOCS_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Prefer to break on paragraphs, then sentences, then words.
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []
    report: list[dict] = []

    for pdf_path in pdf_paths:
        pages, total_pages, skipped = extract_pages(pdf_path)
        doc_chunks, doc_metadatas, doc_ids = chunk_pages(pages, splitter)

        documents.extend(doc_chunks)
        metadatas.extend(doc_metadatas)
        ids.extend(doc_ids)
        report.append(
            {
                "source": pdf_path.name,
                "pages": total_pages,
                "skipped": skipped,
                "chunks": len(doc_chunks),
            }
        )
        print(
            f"  {pdf_path.name:<44} {total_pages:>3} pages -> "
            f"{len(doc_chunks):>4} chunks ({skipped} pages skipped)"
        )

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Embedding {len(documents)} chunks...")
    embeddings = embed(model, documents)

    collection = get_collection(rebuild=args.rebuild)
    store(collection, documents, embeddings, metadatas, ids)

    print_per_document_table(report)
    lengths = [meta["chars"] for meta in metadatas]
    print(
        f"\nTotal chunks: {len(documents)}"
        f"\nAverage chunk length: {sum(lengths) / len(lengths):.0f} chars"
        f"\nEmbedding dimension: {len(embeddings[0])}"
        f"\nCollection '{COLLECTION_NAME}' now holds {collection.count()} vectors"
        f"\nPersisted to: {CHROMA_DIR}"
    )

    smoke_test(collection, model)


if __name__ == "__main__":
    main()
