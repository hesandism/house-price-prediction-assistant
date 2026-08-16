#!/usr/bin/env python3
"""Chunk the Markdown files in docs/ and embed them into a persistent Chroma collection."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:  # langchain-community is being sunset; prefer the standalone package if present.
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Kept separate from the PDF collection in ingest_docs.py so re-running either
# script cannot clobber the other's corpus.
COLLECTION_NAME = "markdown_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Measured in tokens, not characters: the splitter below is built from the
# embedding model's own tokenizer, so these are the counts the model will see.
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Chunks shorter than this are headings or stray list items on their own; they
# embed to noise rather than to anything retrievable.
MIN_CHUNK_CHARS = 30

CHROMA_BATCH_SIZE = 500


@dataclass
class Chunk:
    """One split of one Markdown file, ready to embed."""

    source: str
    path: str
    position: int
    text: str


def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        # Cosine distance in Chroma assumes unit vectors.
        encode_kwargs={"normalize_embeddings": True, "batch_size": 256},
    )


def build_splitter(embeddings: HuggingFaceEmbeddings) -> RecursiveCharacterTextSplitter:
    """Split on token counts from the embedding model's own tokenizer."""
    model = embeddings.client  # the underlying SentenceTransformer
    max_seq_length = getattr(model, "max_seq_length", None)
    if max_seq_length and CHUNK_SIZE > max_seq_length:
        print(
            f"Warning: CHUNK_SIZE={CHUNK_SIZE} exceeds the {max_seq_length}-token "
            f"window of {EMBEDDING_MODEL}; the tail of each chunk will be "
            f"truncated before embedding.",
            file=sys.stderr,
        )

    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        model.tokenizer,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Prefer Markdown structure: sections, then paragraphs, then sentences.
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )


def chunk_file(
    md_path: Path, splitter: RecursiveCharacterTextSplitter
) -> tuple[list[Chunk], int]:
    """Split one Markdown file, returning (chunks, number dropped as too short)."""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    relative = md_path.relative_to(DOCS_DIR).as_posix()

    chunks: list[Chunk] = []
    skipped = 0

    for piece in splitter.split_text(text):
        piece = piece.strip()
        if len(piece) < MIN_CHUNK_CHARS:
            skipped += 1
            continue
        chunks.append(
            Chunk(
                source=md_path.name,
                path=relative,
                position=len(chunks),
                text=piece,
            )
        )

    return chunks, skipped


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
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )


def store(collection: chromadb.Collection, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    """Write in batches; Chroma rejects a single oversized add()."""
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "source": chunk.source,
            "path": chunk.path,
            "chunk": chunk.position,
            "chars": len(chunk.text),
        }
        for chunk in chunks
    ]
    # A deterministic id means re-running the script upserts the same chunk
    # instead of creating a duplicate copy of the corpus. The id is built from
    # the full relative path so two same-named files in different folders --
    # docs/notes.md and docs/nested/notes.md -- do not overwrite each other.
    ids = [
        f"{chunk.path.removesuffix('.md').replace('/', '__')}-c{chunk.position}"
        for chunk in chunks
    ]

    for start in range(0, len(documents), CHROMA_BATCH_SIZE):
        end = start + CHROMA_BATCH_SIZE
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )


def main() -> None:
    # Markdown carries typographic characters (en dashes, smart quotes) that a
    # cp1252 Windows console cannot encode. Degrade them instead of crashing.
    sys.stdout.reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop the collection before ingesting instead of upserting into it.",
    )
    args = parser.parse_args()

    if not DOCS_DIR.is_dir():
        raise FileNotFoundError(f"Docs directory does not exist: {DOCS_DIR}")

    md_paths = sorted(DOCS_DIR.rglob("*.md"))
    if not md_paths:
        print(f"No .md files found in {DOCS_DIR} - nothing to embed.")
        return
    print(f"Found {len(md_paths)} Markdown files in {DOCS_DIR}")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embeddings = load_embeddings()
    splitter = build_splitter(embeddings)

    all_chunks: list[Chunk] = []
    print(f"\nChunking (~{CHUNK_SIZE} tokens, {CHUNK_OVERLAP}-token overlap)")
    for md_path in md_paths:
        chunks, skipped = chunk_file(md_path, splitter)
        all_chunks.extend(chunks)

        note = f" ({skipped} fragments skipped)" if skipped else ""
        relative = md_path.relative_to(DOCS_DIR).as_posix()
        print(f"  {relative:<44} -> {len(chunks):>4} chunks{note}")

    if not all_chunks:
        print("\nEvery file produced only empty or trivially short chunks - nothing to embed.")
        return

    print(f"\nEmbedding {len(all_chunks)} chunks...")
    vectors = embeddings.embed_documents([chunk.text for chunk in all_chunks])

    collection = get_collection(rebuild=args.rebuild)
    store(collection, all_chunks, vectors)

    lengths = [len(chunk.text) for chunk in all_chunks]
    print(
        f"\nTotal chunks: {len(all_chunks)}"
        f"\nAverage chunk length: {sum(lengths) / len(lengths):.0f} chars"
        f"\nEmbedding dimension: {len(vectors[0])}"
        f"\nCollection '{COLLECTION_NAME}' now holds {collection.count()} vectors"
        f"\nPersisted to: {CHROMA_DIR}"
    )


if __name__ == "__main__":
    main()
