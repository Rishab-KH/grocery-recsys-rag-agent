"""
Inventory constraint layer for the RAG retail ops pipeline.

Filters and annotates a ranked recommendation list based on simulated
stock status. Out-of-stock items are replaced with the best available
substitute from an explicit candidate pool.

In production, swap inventory_layer signals for a real inventory API
without changing any logic in this file.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .inventory_layer import compute_stock_flags, load_signals


def _load_product_meta(products_path: str) -> Dict[int, Tuple[int, int]]:
    """Return {product_id: (aisle_id, department_id)} from products.csv."""
    df = pd.read_csv(products_path, usecols=['product_id', 'aisle_id', 'department_id'])
    return {
        int(row.product_id): (int(row.aisle_id), int(row.department_id))
        for row in df.itertuples(index=False)
    }


def _rank_substitutes(
    oos_pid:      int,
    candidates:   List[int],
    product_meta: Dict[int, Tuple[int, int]],
) -> List[int]:
    """
    Sort candidates by catalog proximity to the OOS product.

    Priority order:
      1. Same department AND same aisle  (most relevant replacement)
      2. Same department, different aisle
      3. Everything else

    IMPORTANT: Within each tier, candidates retain their original iteration
    order from ``candidates``. When candidates come from the FAISS top-k
    pool (the typical case), this means higher-similarity items are preferred
    within each catalog-proximity tier. If the candidate pool source changes,
    consider adding an explicit secondary sort key.
    """
    oos_aisle, oos_dept = product_meta.get(oos_pid, (None, None))

    tier1: List[int] = []  # same dept + same aisle
    tier2: List[int] = []  # same dept, different aisle
    tier3: List[int] = []  # different dept

    for p in candidates:
        p_aisle, p_dept = product_meta.get(p, (None, None))
        if oos_dept is not None and p_dept == oos_dept:
            if oos_aisle is not None and p_aisle == oos_aisle:
                tier1.append(p)
            else:
                tier2.append(p)
        else:
            tier3.append(p)

    return tier1 + tier2 + tier3


def apply_inventory_constraints(
    recommended_product_ids: List[int],
    signals_path:  str = './models/product_signals.json',
    candidate_pool: Optional[List[int]] = None,
    products_path: str = './data/products.csv',
) -> Dict:
    """
    Apply stock constraints to a ranked recommendation list.

    Args:
        recommended_product_ids:
            Ranked list of product IDs from the retrieval / reranking stage.
        signals_path:
            Path to models/product_signals.json (from build_product_signals.py).
        candidate_pool:
            Wider set of product IDs for OOS substitution (e.g. k=200 FAISS
            candidates). If None, substitution is skipped and a clear warning
            is issued for each OOS item — no silent failures.
        products_path:
            Path to products.csv for department/aisle-aware substitute ordering.

    Returns:
        {
            'final_recs':    list[int]            — product IDs after constraints,
            'stock_status':  dict[int, str]       — status per recommended product,
            'substitutions': dict[int, int|None]  — oos_id -> substitute_id,
            'warnings':      list[str]            — human-readable stock alerts,
        }
    """
    popularity_pct, reorder_rate = load_signals(signals_path)

    # Check stock for every ID we might touch (recs + substitution pool)
    all_ids = list(set(recommended_product_ids) | set(candidate_pool or []))
    stock = compute_stock_flags(all_ids, popularity_pct, reorder_rate)

    # Load aisle/dept metadata for substitution ranking (graceful fallback if missing)
    product_meta: Dict[int, Tuple[int, int]] = {}
    if Path(products_path).exists():
        product_meta = _load_product_meta(products_path)

    primary_set = set(recommended_product_ids)

    # Build mutable substitute pool: pool items not already recommended and not OOS
    available_subs: List[int] = []
    if candidate_pool is not None:
        available_subs = [
            pid for pid in candidate_pool
            if pid not in primary_set and stock.get(pid, 'in_stock') != 'out_of_stock'
        ]

    final_recs:    List[int]                 = []
    substitutions: Dict[int, Optional[int]]  = {}
    warnings:      List[str]                 = []

    for pid in recommended_product_ids:
        status = stock.get(pid, 'in_stock')

        if status == 'out_of_stock':
            substitutions[pid] = None

            if candidate_pool is None:
                # Substitution requires an explicit candidate pool — do not attempt
                # fake substitution from within the primary list.
                warnings.append(
                    f"[OUT OF STOCK] Product {pid} unavailable. "
                    f"Provide candidate_pool to enable substitution."
                )
            else:
                # Rank available substitutes by catalog proximity
                ranked = _rank_substitutes(pid, available_subs, product_meta)
                substitute = ranked[0] if ranked else None
                substitutions[pid] = substitute

                if substitute is not None:
                    final_recs.append(substitute)
                    available_subs.remove(substitute)  # consume so it isn't reused
                    warnings.append(
                        f"[OUT OF STOCK] Product {pid} unavailable — "
                        f"substituted with product {substitute}."
                    )
                else:
                    warnings.append(
                        f"[OUT OF STOCK] Product {pid} unavailable — "
                        f"no eligible substitute in candidate pool; slot dropped."
                    )

        elif status == 'low_stock':
            final_recs.append(pid)
            warnings.append(f"[LOW STOCK] Product {pid} has limited availability.")

        else:  # in_stock
            final_recs.append(pid)

    return {
        'final_recs':    final_recs,
        'stock_status':  {pid: stock.get(pid, 'in_stock') for pid in recommended_product_ids},
        'substitutions': substitutions,
        'warnings':      warnings,
    }