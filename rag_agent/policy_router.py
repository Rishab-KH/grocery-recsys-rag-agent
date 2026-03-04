def route_policy_docs(intent: str, departments=None, has_oos_or_low_stock=False, substitutions_occurred=False) -> list:
    """
    Route to relevant policy docs based on intent and context.
    """
    intent_lower = intent.lower()
    docs = set()
    # Helper: check if keyword appears in any department name (substring match)
    def _dept_has(keyword: str) -> bool:
        return departments and any(keyword in d.lower() for d in departments)

    if "delivery" in intent_lower or "perishable" in intent_lower or "fast" in intent_lower:
        docs.add("delivery_windows.md")
        docs.add("cold_chain.md")
    if "substitution" in intent_lower or substitutions_occurred:
        docs.add("substitutions.md")
    if "bulk" in intent_lower:
        docs.add("bulk_limits.md")
    if "promo" in intent_lower:
        docs.add("promo_rules.md")
    if "organic" in intent_lower or _dept_has("produce"):
        docs.add("dept_produce.md")
    if "dairy" in intent_lower or "yogurt" in intent_lower or "milk" in intent_lower or "egg" in intent_lower or _dept_has("dairy"):
        docs.add("dept_dairy_eggs.md")
    if "frozen" in intent_lower or _dept_has("frozen"):
        docs.add("dept_frozen.md")
    if "snack" in intent_lower or _dept_has("snack"):
        docs.add("dept_snacks.md")
    # Inventory constraints
    if has_oos_or_low_stock:
        docs.add("substitutions.md")     # substitution rules always relevant for OOS/low-stock
    # Always include general policies if nothing matches
    if not docs:
        docs.add("substitutions.md")     # safe default — applies broadly
    return list(docs)
"""
Simulated inventory service — for demo/development purposes only.

In production, replace load_signals() and compute_stock_flags() with
calls to a real inventory API, database, or warehouse management system.
This module is intentionally self-contained: it only assigns stock flags
and knows nothing about recommendation logic or substitution strategy.
"""

import json
import random
from typing import Dict, List, Optional, Tuple


def load_signals(signals_path: str) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Load pre-computed popularity_pct and reorder_rate from product_signals.json.
    Run scripts/build_product_signals.py once to generate this file.

    In production, replace this with a query to your feature store or
    inventory database.
    """
    with open(signals_path) as f:
        raw = json.load(f)

    popularity_pct = {int(k): v['popularity_pct'] for k, v in raw.items()}
    reorder_rate   = {int(k): v['reorder_rate']   for k, v in raw.items()}
    return popularity_pct, reorder_rate


def compute_stock_flags(
    product_ids:    List[int],
    popularity_pct: Dict[int, float],
    reorder_rate:   Dict[int, float],
    overrides:      Optional[Dict[int, str]] = None,
) -> Dict[int, str]:
    """
    Return a simulated stock status for each product_id.

    Status values: "in_stock" | "low_stock" | "out_of_stock"

    Args:
        product_ids:    List of product IDs to check.
        popularity_pct: Popularity percentile per product (0.0–1.0).
        reorder_rate:   Reorder rate per product (0.0–1.0).
        overrides:      Optional dict of {product_id: status} that bypass the
                        simulation. Useful for testing edge cases (e.g. forcing
                        a specific product to be OOS) without modifying the
                        simulation logic.

    Probability model:
    - out_of_stock:  2% (top popularity) → ~20% (bottom popularity)
                     High reorder rate slightly reduces OOS risk.
    - low_stock:     an independent 10%–15% band above the OOS threshold,
                     giving a realistic total combined risk of ~12%–35%.
    - in_stock:      remainder.

    Deterministic: each product_id seeds its own Random instance, so the
    same product always returns the same status regardless of call order.

    NOTE: This is a heuristic proxy. Replace with real inventory data in production.
    """
    if overrides is None:
        overrides = {}

    flags: Dict[int, str] = {}

    for pid in product_ids:
        # FIX: allow explicit overrides for testability
        if pid in overrides:
            flags[pid] = overrides[pid]
            continue

        pct = popularity_pct.get(pid, 0.0)  # unknown products treated as least popular
        rr  = reorder_rate.get(pid, 0.0)

        # OOS probability: 2% → ~20% inversely proportional to popularity.
        # High reorder rate reduces OOS risk (frequently replenished items).
        oos_prob = (0.02 + 0.18 * (1.0 - pct)) * (1.0 - 0.1 * rr)

        # Independent low-stock probability: 10%–15% depending on popularity.
        # Applied as an additive band above the OOS threshold.
        low_stock_extra = 0.10 + 0.05 * (1.0 - pct)
        low_stock_threshold = oos_prob + low_stock_extra

        # Per-product isolated RNG — does not affect global random state.
        r = random.Random(pid).random()

        if r < oos_prob:
            flags[pid] = 'out_of_stock'
        elif r < low_stock_threshold:
            flags[pid] = 'low_stock'
        else:
            flags[pid] = 'in_stock'

    return flags