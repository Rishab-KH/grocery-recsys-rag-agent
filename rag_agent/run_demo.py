from pathlib import Path
OUTPUT_FILE = Path("rag_agent/demo_outputs.jsonl")
def _hit_rate(final_recs, ground_truth):
    """Compute hit count and hit@10 percentage for final recommendations vs ground truth."""
    if not final_recs:
        return 0, 0.0
    final_pids = [r["product_id"] for r in final_recs]
    hits = sum(1 for pid in final_pids if pid in ground_truth)
    hit_rate = hits / len(final_pids) if final_pids else 0.0
    return hits, hit_rate
from rag_agent.graph import run_pipeline
DEMO_CASES = [
    {"user_id": 6, "intent": "fast delivery and substitutions for perishables"},
    {"user_id": 2, "intent": "bulk staples with promo sensitivity"},
    {"user_id": 10, "intent": "substitutions only; keep organic preferences"},
]
#!/usr/bin/env python3
"""
Demo: run the RAG pipeline for three representative users.

Outputs:
  stdout                       — per-user answer, hit-rate comparison, citations, warnings
  rag_agent/demo_outputs.jsonl — full pipeline output including telemetry (JSON per line)

Usage:
    python rag_agent/run_demo.py
"""

import json
import sys
from pathlib import Path


from dotenv import load_dotenv

def _print_comparison(result: dict):
    raw   = result.get("raw_recommendations", [])
    final = result.get("final_recommendations", [])
    subs  = result.get("substitutions", {})   # {str(oos_pid): sub_pid}
    gt    = set(result.get("ground_truth_items", []))

    col_w = 42
    print("\n[PIPELINE OUTPUT]")
    print(f"  {'#':<3}  {'Pipeline output':<{col_w}}")
    print(f"  {'-'*3}  {'-'*col_w}")

    for i in range(max(len(raw), len(final))):
        fin_item = final[i] if i < len(final) else None
        if fin_item:
            fin_hit  = " *HIT*" if fin_item["product_id"] in gt else ""
            fin_note = ""
            if fin_item["product_id"] not in {r["product_id"] for r in raw}:
                orig_pid = subs.get(str(fin_item["product_id"]))
                if orig_pid:
                    orig = next((r for r in raw if r["product_id"] == int(orig_pid)), None)
                    orig_name = orig["product_name"][:18] if orig else f"pid {orig_pid}"
                    fin_note = f" [sub←{orig_name}]"
            fin_cell = f"{fin_item['product_name'][:col_w - 20]}{fin_note}{fin_hit}"
        else:
            fin_cell = ""
        print(f"  {i+1:<3}  {fin_cell:<{col_w}}")

    # ── hit@10 summary ────────────────────────────────────────────────
    if gt:
        fin_hits, fin_h10   = _hit_rate(final, gt)
        print(f"\n  Ground truth items (test purchases): {len(gt)}")
        print(f"  Pipeline   Hit@{len(final):<2}: {fin_hits:>2} / {len(final)}  ({fin_h10:.0%})")
    else:
        print("\n  (ground truth unavailable for this user)")


# ── per-result printer ──────────────────────────────────────────────────────

def _print_result(result: dict) -> None:
    """Print answer, hit-rate comparison, citations, and warnings."""
    sep = "─" * 72
    print(f"\n{sep}")
    print(f"USER {result['user_id']}  |  {result['intent']}")
    print(sep)

    _print_comparison(result)

    print("\n[ANSWER]")
    print(result["answer"])

    if result.get("citations"):
        print("\n[CITATIONS]")
        for c in result["citations"]:
            print(f"  {c}")

    if result.get("warnings"):
        print("\n[WARNINGS]")
        for w in result["warnings"]:
            print(f"  {w}")


# ── entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    """Run the RAG pipeline for all demo cases and write outputs to JSONL."""
    results = []
    for case in DEMO_CASES:
        print(f"\n>> Running pipeline  user_id={case['user_id']}  intent='{case['intent']}' ...")
        try:
            result = run_pipeline(case["user_id"], case["intent"])
        except (RuntimeError, ValueError, KeyError) as e:
            print(f"   [ERROR] {e}")
            result = {**case, "error": str(e)}

        results.append(result)
        _print_result(result)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            tel = r.get("telemetry_ms", {})
            # Always add DeepEval metric aliases, even if value is missing (set to None)
            r["hallucination_rate"] = tel.get("deepeval_hallucination")
            r["context_recall"] = tel.get("deepeval_retrieval_quality")
            r["context_precision"] = tel.get("deepeval_retrieval_quality")
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n\n>> Full output + telemetry logged → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
