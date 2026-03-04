"""
Demo: apply inventory constraints to a recommendation list.

Run after build_product_signals.py has been executed:
    python scripts/demo_inventory_constraints.py
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_agent.constraints import apply_inventory_constraints

# Example: ranked product IDs from the retrieval / reranking stage
recommended_product_ids = [13176, 38928, 25133, 47766, 27966, 21137, 24852, 47209]

# Optional: full k=200 candidate pool for substitution fallback
# If omitted, substitutes are drawn from the tail of recommended_product_ids
candidate_pool = None

result = apply_inventory_constraints(
    recommended_product_ids=recommended_product_ids,
    signals_path='./models/product_signals.json',
    candidate_pool=candidate_pool,
)

print("Stock status per product:")
for pid, status in result['stock_status'].items():
    print(f"  {pid}: {status}")

print("\nFinal recommendations after constraints:")
for pid in result['final_recs']:
    print(f"  {pid}")

if result['substitutions']:
    print("\nSubstitutions (OOS -> substitute):")
    for oos, sub in result['substitutions'].items():
        print(f"  {oos} -> {sub}")

if result['warnings']:
    print("\nWarnings:")
    for w in result['warnings']:
        print(f"  {w}")
