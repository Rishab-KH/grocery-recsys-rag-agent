#!/usr/bin/env python3
"""
Build FAISS policy index from .md files.

Embedding model: text-embedding-3-small  (recommended for this use case)
  - 1536-d vectors; strong recall for short policy chunks
  - Cost-efficient ($0.02 / 1M tokens) — right for static docs indexed once

Index: IndexFlatIP with L2-normalised vectors = exact cosine similarity search.

Usage:
    python rag_agent/build_index.py
    python rag_agent/build_index.py --policies-dir path/to/policies
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

load_dotenv()

# ── paths ──────────────────────────────────────────────────────────────────
_RAG_DIR = Path(__file__).parent

INDEX_DIR   = _RAG_DIR / "policy_index"
INDEX_FILE  = INDEX_DIR / "index.faiss"
CHUNKS_FILE = INDEX_DIR / "chunks.jsonl"
BM25_FILE   = INDEX_DIR / "bm25.pkl"

EMBED_MODEL   = "text-embedding-3-small"
CHUNK_SIZE    = 900    # chars; smaller chunks for finer retrieval
CHUNK_OVERLAP = 450    # chars; more overlap for better recall


# ── policy directory resolution ────────────────────────────────────────────
def _resolve_policies_dir(override: str | None = None) -> Path:
    if override:
        p = Path(override)
        if not p.exists():
            sys.exit(f"[ERROR] --policies-dir not found: {p}")
        return p
    # auto-detect: rag_agent/policies (sibling of this file)
    default = _RAG_DIR / "policies"
    if default.exists() and list(default.glob("*.md")):
        return default
    sys.exit(
        "[ERROR] No policy directory found.\n"
        "  Expected rag_agent/policies/ with .md files,\n"
        "  or pass --policies-dir <path>.\n"
    )


# ── chunking ───────────────────────────────────────────────────────────────
def chunk_text(text: str, doc: str) -> list[dict]:
    """
    Split text into overlapping character-level chunks.
    Prefers to break at newlines near the boundary so policy bullet points
    and table rows stay intact.
    """
    chunks: list[dict] = []
    start    = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))

        # prefer a newline break in the last 25% of the window
        if end < len(text):
            nl = text.rfind("\n", start + int(CHUNK_SIZE * 0.75), end)
            if nl != -1:
                end = nl + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"doc": doc, "chunk_id": chunk_id, "text": chunk})
            chunk_id += 1

        start = end - CHUNK_OVERLAP if end < len(text) else len(text)

    return chunks


# ── embedding ──────────────────────────────────────────────────────────────
def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embed in batches of 100; returns (N, D) float32 array."""
    BATCH = 100
    vecs: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + BATCH])
        vecs.extend(d.embedding for d in resp.data)
    return np.array(vecs, dtype=np.float32)


def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


# FIX: regex tokeniser matching retrieve.py — strips punctuation so
# "delivery," and "delivery" are the same token.
def tokenize(text: str) -> list[str]:
    """Regex tokeniser for BM25 — consistent with retrieve.py."""
    return re.findall(r"\w+", text.lower())


# ── main ───────────────────────────────────────────────────────────────────
def build_index(policies_dir: Path) -> None:
    client   = OpenAI()  # reads OPENAI_API_KEY from env / .env
    md_files = sorted(policies_dir.glob("*.md"))

    if not md_files:
        sys.exit(f"[ERROR] No .md files found in {policies_dir}")

    print(f"[build_index] Policies dir : {policies_dir}")
    print(f"[build_index] Files found  : {len(md_files)}")

    all_chunks: list[dict] = []
    for path in md_files:
        text   = path.read_text(encoding="utf-8")
        chunks = chunk_text(text, path.name)
        all_chunks.extend(chunks)
        print(f"  {path.name:<38} {len(chunks)} chunks")

    print(f"\n[build_index] Total chunks : {len(all_chunks)}")
    print(f"[build_index] Embedding with {EMBED_MODEL} ...")

    texts = [c["text"] for c in all_chunks]
    vecs  = embed_texts(client, texts)
    vecs  = l2_normalize(vecs)

    dim   = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    print(f"[build_index] FAISS index  : {index.ntotal} vectors, dim={dim}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"[build_index] Saved index  → {INDEX_FILE}")
    print(f"[build_index] Saved chunks → {CHUNKS_FILE}")

    # ── BM25 sparse index ────────────────────────────────────────────────
    print("[build_index] Building BM25 sparse index ...")
    corpus = [tokenize(c["text"]) for c in all_chunks]
    bm25   = BM25Okapi(corpus)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    print(f"[build_index] Saved BM25   → {BM25_FILE}")
    print("[build_index] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS policy index")
    parser.add_argument(
        "--policies-dir", default=None,
        help="Path to directory containing .md policy files. "
             "Auto-detects rag_agent/policies if omitted.",
    )
    args = parser.parse_args()
    build_index(_resolve_policies_dir(args.policies_dir))