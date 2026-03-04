"""
Hybrid retrieval for the policy RAG pipeline.

Pipeline:
  query
   ├── Dense FAISS (text-embedding-3-small)  → top-20
   └── BM25 sparse (rank_bm25)               → top-20
        └── RRF fusion                        → top-30
             └── Cohere Reranker (optional)   → top-k (default 6)

Cohere reranking is optional:
  - Set COHERE_API_KEY in .env to enable.
  - If the key is absent, the pipeline returns RRF-fused results directly.

EMBED_MODEL must match build_index.py — both use text-embedding-3-small.
"""

import os
import re
import pickle
import logging
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from openai import OpenAI

_RAG_DIR    = Path(__file__).parent
INDEX_FILE  = _RAG_DIR / "policy_index" / "index.faiss"
CHUNKS_FILE = _RAG_DIR / "policy_index" / "chunks.jsonl"
BM25_FILE   = _RAG_DIR / "policy_index" / "bm25.pkl"

EMBED_MODEL = "text-embedding-3-small"  # must match build_index.py

# ── how many candidates each retriever fetches before fusion / reranking ──
DENSE_CANDIDATES = 80  # increase candidate pool for reranking
BM25_CANDIDATES  = 80  # increase candidate pool for reranking
RRF_K            = 100   # higher = smoother rank fusion
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.10

logger = logging.getLogger(__name__)

# module-level cache — loaded once per process
_index:  Optional[faiss.Index] = None
_chunks: Optional[list[dict]]  = None
_bm25                          = None   # BM25Okapi instance


def _ensure_loaded() -> None:
    global _index, _chunks, _bm25
    if _index is not None:
        return

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {INDEX_FILE}\n"
            "Run:  python rag_agent/build_index.py"
        )
    if not BM25_FILE.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {BM25_FILE}\n"
            "Run:  python rag_agent/build_index.py"
        )

    import json
    _index = faiss.read_index(str(INDEX_FILE))
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        _chunks = [json.loads(line) for line in f]
    with open(BM25_FILE, "rb") as f:
        _bm25 = pickle.load(f)


# ── tokeniser (must match build_index.py) ─────────────────────────────────
# FIX: use regex tokenizer instead of naive .split() so punctuation
# (commas, pipes, markdown syntax) doesn't pollute BM25 tokens.
def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# ── individual retrievers ──────────────────────────────────────────────────
def _dense_retrieve(query: str, top_k: int, client: OpenAI) -> list[dict]:
    """Embed query → cosine search via FAISS → return top-k with scores."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query])
    qvec = np.array(resp.data[0].embedding, dtype=np.float32).reshape(1, -1)
    qvec /= np.maximum(np.linalg.norm(qvec), 1e-9)

    scores, indices = _index.search(qvec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = dict(_chunks[idx])
        chunk["_dense_score"] = float(score)
        chunk["_idx"]         = int(idx)
        results.append(chunk)
    return results


def _bm25_retrieve(query: str, top_k: int, restrict_docs: list[str] = None) -> list[dict]:
    """Tokenise query → BM25 scores → return top-k chunks, restricted to given docs if provided."""
    tokens = _tokenize(query)
    # Keyword boosting for policy terms
    boost_terms = ["promotion", "substitution", "organic", "bulk", "policy", "compliance", "department", "sku"]
    boosted_tokens = tokens + [t for t in boost_terms if t in query.lower()]
    scores = _bm25.get_scores(boosted_tokens)          # shape: (n_chunks,)
    # Restrict to chunks from routed docs if specified
    if restrict_docs:
        restrict_docs_set = set(restrict_docs)
        valid_indices = [i for i, c in enumerate(_chunks) if c.get("doc") in restrict_docs_set]
        filtered_scores = [(i, scores[i]) for i in valid_indices]
        top_indices = [i for i, _ in sorted(filtered_scores, key=lambda x: x[1], reverse=True)[:top_k]]
    else:
        top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk = dict(_chunks[idx])
        chunk["_bm25_score"] = float(scores[idx])
        chunk["_idx"]        = int(idx)
        results.append(chunk)
    return results


# ── fusion ─────────────────────────────────────────────────────────────────
def _rrf_fuse(
    dense: list[dict],
    bm25:  list[dict],
    k:     int = RRF_K,
) -> list[dict]:
    """
    Reciprocal Rank Fusion: score(d) = Σ_i  1 / (k + rank_i(d))

    Items appearing in only one list are still scored (rank = len(list)+1
    for the missing list, effectively giving them a small RRF contribution).
    """
    rrf: dict[int, float] = {}
    idx_to_chunk: dict[int, dict] = {}

    for rank, chunk in enumerate(dense, start=1):
        i = chunk["_idx"]
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (k + rank)
        idx_to_chunk[i] = chunk

    for rank, chunk in enumerate(bm25, start=1):
        i = chunk["_idx"]
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (k + rank)
        idx_to_chunk[i] = chunk

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)
    fused  = []
    for idx, rrf_score in ranked:
        chunk = dict(idx_to_chunk[idx])
        chunk["_rrf_score"] = round(rrf_score, 6)
        fused.append(chunk)
    return fused


# ── Cohere reranker (optional) ─────────────────────────────────────────────
def _cohere_rerank(
    query:      str,
    candidates: list[dict],
    top_k:      int,
) -> list[dict]:
    """
    Cross-encoder reranking via Cohere rerank-english-v3.0.
    Returns top_k chunks with an added 'score' field (relevance_score).

    Falls back to returning the top_k RRF candidates unchanged if:
      - COHERE_API_KEY is not set
      - cohere package is not installed
      - API call fails
    """
    api_key = os.getenv("COHERE_API_KEY", "")
    if not api_key:
        # no key — return RRF top-k with rrf_score as the final score
        for c in candidates[:top_k]:
            c["score"] = round(c.get("_rrf_score", 0.0), 4)
        return candidates[:top_k]

    try:
        import cohere
        co = cohere.ClientV2(api_key=api_key)
        docs     = [c["text"] for c in candidates]
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=docs,
            top_n=top_k,
        )
        reranked = []
        for r in response.results:
            chunk = dict(candidates[r.index])
            chunk["score"] = round(r.relevance_score, 4)
            reranked.append(chunk)
        return reranked

    # FIX: catch specific Cohere errors + log, instead of bare except Exception
    except ImportError:
        logger.warning("cohere package not installed — falling back to RRF order")
        for c in candidates[:top_k]:
            c["score"] = round(c.get("_rrf_score", 0.0), 4)
        return candidates[:top_k]

    except Exception as exc:
        logger.warning("Cohere rerank failed (%s: %s) — falling back to RRF order",
                        type(exc).__name__, exc)
        for c in candidates[:top_k]:
            c["score"] = round(c.get("_rrf_score", 0.0), 4)
        return candidates[:top_k]


# ── forced-doc injection ───────────────────────────────────────────────────
def _inject_forced_chunks(
    retrieved:   list[dict],
    forced_docs: list[str],
    query:       str,
) -> list[dict]:
    """
    For each doc in forced_docs not already represented in retrieved,
    find the best BM25-scored chunk from that doc and append it.
    Docs missing from the index are silently skipped.

    FIX: extract raw intent from the composite query (before the first ' | '
    separator) so BM25 scoring isn't polluted by structured prefixes like
    "Departments: ...", "Warnings: ..." that help dense retrieval but add
    noise to BM25 term matching.
    """
    present_docs = {c["doc"] for c in retrieved}

    # Use only the intent portion (before first ' | ') for BM25 scoring
    # of forced chunks. The structured prefixes help dense retrieval but
    # add noise to BM25 term matching.
    intent_portion = query.split(" | ")[0] if " | " in query else query
    tokens      = _tokenize(intent_portion)
    bm25_scores = _bm25.get_scores(tokens)

    result = list(retrieved)
    for doc in forced_docs:
        if doc in present_docs:
            continue
        candidates = [
            (i, bm25_scores[i], c)
            for i, c in enumerate(_chunks)
            if c["doc"] == doc
        ]
        if not candidates:
            continue   # doc not in index — skip safely
        best_i, best_score, best_chunk = max(candidates, key=lambda x: x[1])
        injected = dict(best_chunk)
        injected["score"] = round(float(best_score), 4)
        result.append(injected)
        present_docs.add(doc)
    return result


# ── public API ─────────────────────────────────────────────────────────────
def build_query(
    intent:               str,
    departments:          list[str] | None = None,
    aisles:               list[str] | None = None,
    warnings:             list[str] | None = None,
    substituted_products: list[str] | None = None,
    top_product_names:    list[str] | None = None,
) -> str:
    """
    Compose a retrieval query combining multiple contextual signals:
      intent            → primary semantic signal
      departments/aisles → scope to relevant category policies
      warnings          → surfaces constraint-triggered policy sections
      substituted names → pulls in substitution policy chunks
      top product names → term overlap with policy vocabulary
    """
    parts = [intent]
    if departments:
        parts.append("Departments: " + ", ".join(departments))
    if aisles:
        parts.append("Aisles: " + ", ".join(aisles[:6]))
    if warnings:
        parts.append("Warnings: " + "; ".join(warnings[:3]))
    if substituted_products:
        parts.append("Substituted products: " + ", ".join(substituted_products[:5]))
    if top_product_names:
        parts.append("Top recommendations: " + ", ".join(top_product_names[:10]))
    return " | ".join(parts)


def retrieve(
    query:       str,
    top_k:       int = 6,
    client:      Optional[OpenAI] = None,
    forced_docs: list[str] | None = None,
) -> dict:
    """
    Full hybrid retrieval pipeline:
      dense FAISS + BM25  →  RRF fusion  →  Cohere rerank (if key set)
      → forced-doc injection (if forced_docs provided)

    Returns:
    {
        "chunks": [...],
        "confidence": float,
        "low_confidence": bool,
    }
    chunks include top_k plus one chunk per forced doc not already present.
    """
    _ensure_loaded()
    if client is None:
        client = OpenAI()

    dense    = _dense_retrieve(query, top_k=DENSE_CANDIDATES, client=client)
    sparse   = _bm25_retrieve(query,  top_k=BM25_CANDIDATES, restrict_docs=forced_docs)
    fused    = _rrf_fuse(dense, sparse)
    # Tune reranker: increase top_k to 8 for more evidence, then filter to top 5 for answer
    reranked = _cohere_rerank(query, fused, top_k=8)

    # Return top 3–5 policy chunks with doc_name, chunk_text, chunk_id
    confidence = float(max((c.get("score", 0.0) for c in reranked), default=0.0))
    low_confidence = confidence < RETRIEVAL_CONFIDENCE_THRESHOLD
    if low_confidence:
        logger.warning("Low-confidence retrieval: no relevant policy evidence found")
        return {
            "chunks": [],
            "confidence": confidence,
            "low_confidence": True,
        }

    if forced_docs:
        reranked = _inject_forced_chunks(reranked, forced_docs, query)

    # strip internal scoring fields before returning
    _internal = {"_dense_score", "_bm25_score", "_rrf_score", "_idx"}
    # Filter to top 5 by reranker score for answer generation
    chunks = [
        {
            "doc": c.get("doc"),
            "chunk_id": c.get("chunk_id"),
            "text": c.get("text"),
            "score": c.get("score", 0.0)
        }
        for c in sorted(reranked, key=lambda x: x.get("score", 0.0), reverse=True)[:5]
        if c.get("doc") and c.get("text")
    ]
    return {
        "chunks": chunks,
        "confidence": confidence,
        "low_confidence": False,
    }