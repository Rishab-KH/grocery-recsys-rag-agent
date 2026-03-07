"""
Hybrid retrieval for the policy RAG pipeline.

Pipeline:
  query  (composite: intent + departments + aisles + product names)
   ├── Dense FAISS (text-embedding-3-small)  → top-80
   └── BM25 sparse (rank_bm25)               → top-80
        └── RRF fusion                        → scored pool
             └── Cohere Reranker (optional)   → scored pool
                  └── Score-blend (RRF + reranker) → top-k

Design:  Dense and BM25 both search the full index (no doc restriction).
RRF fusion produces a strong ranking.  The Cohere reranker adds cross-
encoder signal but uses a *cleaner* query (intent + departments only)
so structural noise (|, product lists) doesn't confuse it.  Final
ranking blends normalised RRF and reranker scores, preventing the
reranker from completely overriding strong lexical+semantic evidence.

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
DENSE_CANDIDATES = 80
BM25_CANDIDATES  = 80
RRF_K            = 100   # higher = smoother rank fusion
RETRIEVAL_CONFIDENCE_THRESHOLD = 0.10

# ── score blending weight ────────────────────────────────────────────────
# final_score = RERANKER_WEIGHT * reranker_score + (1 - RERANKER_WEIGHT) * norm_rrf
# 0.6 gives the reranker strong influence while preventing it from
# completely overriding a top-ranked RRF candidate.
RERANKER_WEIGHT = 0.6

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


def _bm25_retrieve(query: str, top_k: int) -> list[dict]:
    """Tokenise query → BM25 scores → return top-k chunks (unrestricted)."""
    tokens = _tokenize(query)
    # Keyword boosting for policy terms
    boost_terms = ["promotion", "substitution", "organic", "bulk", "policy", "compliance", "department", "sku"]
    boosted_tokens = tokens + [t for t in boost_terms if t in query.lower()]
    scores = _bm25.get_scores(boosted_tokens)          # shape: (n_chunks,)
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
    Returns top_k chunks with an added '_reranker_score' field.

    Falls back to returning the top_k RRF candidates unchanged if:
      - COHERE_API_KEY is not set
      - cohere package is not installed
      - API call fails
    """
    api_key = os.getenv("COHERE_API_KEY", "")
    if not api_key:
        for c in candidates[:top_k]:
            c["_reranker_score"] = 0.0
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
            chunk["_reranker_score"] = round(r.relevance_score, 4)
            reranked.append(chunk)
        return reranked

    except ImportError:
        logger.warning("cohere package not installed — falling back to RRF order")
        for c in candidates[:top_k]:
            c["_reranker_score"] = 0.0
        return candidates[:top_k]

    except Exception as exc:
        logger.warning("Cohere rerank failed (%s: %s) — falling back to RRF order",
                        type(exc).__name__, exc)
        for c in candidates[:top_k]:
            c["_reranker_score"] = 0.0
        return candidates[:top_k]


# ── score blending ─────────────────────────────────────────────────────────
def _blend_scores(
    candidates: list[dict],
    alpha: float = RERANKER_WEIGHT,
) -> list[dict]:
    """
    Blend normalised RRF scores and reranker scores to produce a final ranking.

    final_score = alpha * reranker_score + (1 - alpha) * normalised_rrf_score

    This prevents the reranker from completely overriding strong RRF evidence
    while still leveraging cross-encoder relevance understanding.
    """
    # Normalise RRF scores to [0, 1]
    rrf_scores = [c.get("_rrf_score", 0.0) for c in candidates]
    rrf_max = max(rrf_scores) if rrf_scores else 1.0
    rrf_min = min(rrf_scores) if rrf_scores else 0.0
    rrf_range = rrf_max - rrf_min if rrf_max > rrf_min else 1.0

    blended = []
    for c in candidates:
        norm_rrf    = (c.get("_rrf_score", 0.0) - rrf_min) / rrf_range
        reranker_sc = c.get("_reranker_score", 0.0)
        c["score"]  = round(alpha * reranker_sc + (1 - alpha) * norm_rrf, 4)
        blended.append(c)

    blended.sort(key=lambda x: x["score"], reverse=True)
    return blended


# ── reranker query builder ─────────────────────────────────────────────────
# ── policy-term mapping for richer reranker queries ────────────────────────
_INTENT_POLICY_TERMS: dict[str, list[str]] = {
    "substitut": ["substitution policy", "replacement rules"],
    "promo":     ["promotional pricing", "promotion eligibility"],
    "bulk":      ["bulk order limits", "quantity caps"],
    "deliver":   ["delivery windows", "fulfillment schedule"],
    "cold":      ["cold chain", "temperature compliance"],
    "refund":    ["refund policy", "return rules"],
    "organic":   ["organic certification", "organic preferences"],
    "perishab":  ["perishable handling", "delivery windows for perishables"],
    "frozen":    ["frozen storage", "cold chain compliance"],
}


def build_reranker_query(composite_query: str) -> str:
    """
    Build a cleaner, natural-language query for the Cohere cross-encoder.

    The composite query uses '|' delimiters and long product lists that
    confuse cross-encoders.  Extract the intent and departments,
    discover policy-relevant terms from the intent, and rephrase as
    natural English for better relevance matching.
    """
    parts = [p.strip() for p in composite_query.split(" | ")]
    intent = parts[0] if parts else composite_query

    departments = ""
    for p in parts:
        if p.startswith("Departments:"):
            departments = p.replace("Departments:", "").strip()
            break

    # Discover policy-relevant terms from the intent
    intent_lower = intent.lower()
    extra_terms: list[str] = []
    for trigger, terms in _INTENT_POLICY_TERMS.items():
        if trigger in intent_lower:
            extra_terms.extend(terms)

    base = f"Policy compliance rules for {departments} departments" if departments else "Policy compliance rules"
    result = f"{base}. Customer intent: {intent}"
    if extra_terms:
        result += ". Relevant policies: " + ", ".join(dict.fromkeys(extra_terms))  # deduplicate
    return result


# ── department-affinity penalty ────────────────────────────────────────────
# Map from dept policy filename stems to the department names that appear
# in the query's "Departments: …" field.  E.g. "dept_dairy_eggs" matches
# "dairy eggs", "dairy", or "eggs".
_DEPT_FILE_TO_KEYWORDS: dict[str, list[str]] = {
    "dept_produce":    ["produce"],
    "dept_dairy_eggs": ["dairy", "eggs", "dairy eggs"],
    "dept_frozen":     ["frozen"],
    "dept_snacks":     ["snacks"],
    "dept_meat":       ["meat", "seafood", "meat seafood"],
}

DEPT_MISMATCH_PENALTY = 0.25   # multiply score by this if department doesn't match


def _apply_dept_affinity(candidates: list[dict], composite_query: str) -> list[dict]:
    """
    Penalise department-specific chunks (dept_*.md) whose department does
    not match any department in the query.  General policy docs (substitutions,
    bulk_limits, delivery_windows, etc.) are unaffected.

    This is a soft signal — mismatched dept docs still appear if nothing
    better is available, but they won't outrank relevant general docs.
    """
    # Extract departments from query
    query_depts: set[str] = set()
    for part in composite_query.split(" | "):
        if part.strip().startswith("Departments:"):
            dept_str = part.strip().replace("Departments:", "").strip().lower()
            query_depts = {d.strip() for d in dept_str.split(",")}
            break

    if not query_depts:
        return candidates

    result = []
    for c in candidates:
        doc = c.get("doc", "")
        stem = doc.replace(".md", "")          # e.g. "dept_frozen"

        if stem in _DEPT_FILE_TO_KEYWORDS:
            # Check if any keyword matches any query department
            keywords = _DEPT_FILE_TO_KEYWORDS[stem]
            matches = any(
                kw in qd or qd in kw
                for kw in keywords
                for qd in query_depts
            )
            if not matches:
                c = dict(c)
                c["score"] = round(c.get("score", 0.0) * DEPT_MISMATCH_PENALTY, 4)
        result.append(c)

    result.sort(key=lambda x: x.get("score", 0.0), reverse=True)
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
) -> dict:
    """
    Full hybrid retrieval pipeline (no forced-doc injection):
      dense FAISS + BM25 → RRF fusion → Cohere rerank → score blend
      → department-affinity penalty → doc-diversity cap → top-k

    The reranker receives a cleaner natural-language query (intent +
    departments only) to avoid confusion from |‑delimited product lists.
    Final ranking blends normalised RRF and reranker scores so the
    reranker adds signal without completely overriding strong RRF evidence.

    Returns:
    {
        "chunks": [...],
        "confidence": float,
        "low_confidence": bool,
    }
    """
    _ensure_loaded()
    if client is None:
        client = OpenAI()

    # Stage 1–3: dense + BM25 → RRF fusion (both search full index)
    dense  = _dense_retrieve(query, top_k=DENSE_CANDIDATES, client=client)
    sparse = _bm25_retrieve(query,  top_k=BM25_CANDIDATES)
    fused  = _rrf_fuse(dense, sparse)

    # Stage 4: Cohere reranker with a cleaner query
    reranker_query = build_reranker_query(query)
    reranked = _cohere_rerank(reranker_query, fused, top_k=min(20, len(fused)))

    # Stage 5: blend RRF + reranker scores for final ranking
    blended = _blend_scores(reranked)

    # Stage 6: department-affinity penalty
    # If a chunk is from a dept_*.md file whose department doesn't match
    # any of the user's departments, penalise its score.  This prevents
    # e.g. dept_frozen.md from outranking substitutions.md when the user
    # has no frozen items.  General (non-dept) docs are unaffected.
    blended = _apply_dept_affinity(blended, query)

    confidence = float(max((c.get("score", 0.0) for c in blended), default=0.0))
    low_confidence = confidence < RETRIEVAL_CONFIDENCE_THRESHOLD
    low_confidence = confidence < RETRIEVAL_CONFIDENCE_THRESHOLD
    if low_confidence:
        logger.warning("Low-confidence retrieval: no relevant policy evidence found")
        return {
            "chunks": [],
            "confidence": confidence,
            "low_confidence": True,
        }

    # ── doc-diversity selection ─────────────────────────────────────────
    # Cap chunks per source document to ensure broad coverage.
    # Walk blended list in score order; skip a chunk if its source doc
    # already has MAX_PER_DOC chunks selected.
    MAX_PER_DOC = 2
    doc_counts: dict[str, int] = {}
    final_chunks: list[dict] = []
    for c in blended:
        if len(final_chunks) >= top_k:
            break
        doc = c.get("doc", "")
        if doc_counts.get(doc, 0) < MAX_PER_DOC:
            final_chunks.append(c)
            doc_counts[doc] = doc_counts.get(doc, 0) + 1

    # Strip internal scoring fields before returning
    chunks = [
        {
            "doc": c.get("doc"),
            "chunk_id": c.get("chunk_id"),
            "text": c.get("text"),
            "score": c.get("score", 0.0)
        }
        for c in final_chunks
        if c.get("doc") and c.get("text")
    ]
    return {
        "chunks": chunks,
        "confidence": confidence,
        "low_confidence": False,
    }