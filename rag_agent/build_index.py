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
MAX_CHUNK_CHARS = 1200   # hard ceiling per chunk; only splits if a section exceeds this


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
def _split_long_section(text: str, doc: str, section_header: str,
                        max_chars: int) -> list[str]:
    """
    Split a single section that exceeds max_chars into sub-chunks.
    Breaks at paragraph boundaries (double newline) or bullet points,
    prepending the section header to every sub-chunk so context is preserved.
    """
    prefix = section_header.strip() + "\n"
    # Split by paragraphs / bullet groups first
    paragraphs = re.split(r'\n\n+', text)
    sub_chunks: list[str] = []
    current = prefix

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Would adding this paragraph exceed the limit?
        if len(current) + len(para) + 2 > max_chars and len(current) > len(prefix):
            sub_chunks.append(current.strip())
            current = prefix  # restart with header
        current += para + "\n\n"

    if current.strip() and current.strip() != prefix.strip():
        sub_chunks.append(current.strip())

    # Edge case: single paragraph that itself is too long — fall back to
    # sentence-level splitting
    final: list[str] = []
    for sc in sub_chunks:
        if len(sc) <= max_chars:
            final.append(sc)
        else:
            # Split by sentences (period + space or newline)
            sentences = re.split(r'(?<=[.!?])\s+|\n', sc)
            buf = prefix
            for sent in sentences:
                sent = sent.strip()
                if not sent:
                    continue
                if len(buf) + len(sent) + 1 > max_chars and len(buf) > len(prefix):
                    final.append(buf.strip())
                    buf = prefix
                buf += sent + " "
            if buf.strip() and buf.strip() != prefix.strip():
                final.append(buf.strip())

    return final if final else [text.strip()]


def chunk_text(text: str, doc: str) -> list[dict]:
    """
    Semantic markdown-section chunking:
      1. Split on ## headers — each section becomes a candidate chunk.
      2. Prepend the doc title (# heading) and section heading to every chunk
         so each chunk is self-contained with full context.
      3. If a section exceeds MAX_CHUNK_CHARS, split further at paragraph
         boundaries while repeating the section header.
      4. Tiny consecutive sections (< 200 chars) are merged together.

    This produces fewer, more coherent chunks than character-level splitting
    and eliminates mid-sentence breaks and orphaned bullet points.
    """
    chunks: list[dict] = []

    # Extract document title (first # line)
    title_match = re.match(r'^(#\s+.+?)(?:\n|$)', text, re.MULTILINE)
    doc_title = title_match.group(1).strip() if title_match else ""
    title_prefix = doc_title + "\n\n" if doc_title else ""

    # Split into sections by ## headers
    # This produces pairs: (header_or_empty, body_text)
    section_splits = re.split(r'(^##\s+.+$)', text, flags=re.MULTILINE)

    # Build list of (header, body) tuples
    sections: list[tuple[str, str]] = []
    i = 0
    # Content before first ## is preamble (usually just the # title + scope)
    if section_splits and not section_splits[0].startswith('## '):
        preamble = section_splits[0].strip()
        if preamble:
            sections.append(("", preamble))
        i = 1

    while i < len(section_splits):
        header = section_splits[i].strip() if i < len(section_splits) else ""
        body = section_splits[i + 1].strip() if i + 1 < len(section_splits) else ""
        if header or body:
            sections.append((header, body))
        i += 2

    # Merge and chunk
    MIN_MERGE_CHARS = 200
    chunk_id = 0
    merge_buf_header = ""
    merge_buf_text = ""

    def _flush(header: str, body: str) -> None:
        nonlocal chunk_id
        # Build chunk text: doc_title + section_header + body
        # Skip title prefix for preamble chunks that already contain it
        if not header and body.startswith(doc_title):
            full_text = body
        else:
            full_text = (title_prefix + header + "\n" + body).strip() if header else (title_prefix + body).strip()

        if len(full_text) <= MAX_CHUNK_CHARS:
            chunks.append({"doc": doc, "chunk_id": chunk_id, "text": full_text})
            chunk_id += 1
        else:
            # Section too long — split at paragraphs
            sub_chunks = _split_long_section(
                body, doc, title_prefix + header, MAX_CHUNK_CHARS
            )
            for sc in sub_chunks:
                chunks.append({"doc": doc, "chunk_id": chunk_id, "text": sc})
                chunk_id += 1

    for header, body in sections:
        combined_len = len(merge_buf_text) + len(body) + len(header) + len(title_prefix)

        # If this section is tiny, try merging with previous
        if len(body) < MIN_MERGE_CHARS and combined_len <= MAX_CHUNK_CHARS:
            if merge_buf_text:
                merge_buf_text += "\n\n" + header + "\n" + body
            else:
                merge_buf_header = header
                merge_buf_text = body
            continue

        # Flush any accumulated merge buffer first
        if merge_buf_text:
            _flush(merge_buf_header, merge_buf_text)
            merge_buf_header = ""
            merge_buf_text = ""

        _flush(header, body)

    # Flush final merge buffer
    if merge_buf_text:
        _flush(merge_buf_header, merge_buf_text)

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