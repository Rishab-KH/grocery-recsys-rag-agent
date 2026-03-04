#!/usr/bin/env python3
"""
Optional CrewAI wrapper for the RAG recommendation pipeline.

Design:
  - Wraps rag/graph.py's run_pipeline() — zero new logic implemented here.
  - CrewAI provides a thin orchestration + audit layer on top of the graph.
  - LangSmith tracing is auto-enabled when env vars are present; no-op otherwise.
  - crewai is optional: gracefully falls back to direct run_pipeline() if absent.

LangSmith vars (all optional):
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=ls__...
    LANGCHAIN_PROJECT=instacart-recsys   (default)

Usage:
    python rag/crew_runner.py
    LANGCHAIN_TRACING_V2=true LANGCHAIN_API_KEY=ls__... python rag/crew_runner.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

OUTPUT_FILE = Path(__file__).parent / "crew_outputs.jsonl"

DEMO_CASES = [
    {"user_id": 6,  "intent": "fast delivery and substitutions for perishables"},
    {"user_id": 2,  "intent": "bulk staples with promo sensitivity"},
    {"user_id": 10, "intent": "substitutions only; keep organic preferences"},
]


# ── LangSmith ──────────────────────────────────────────────────────────────
def _setup_langsmith() -> bool:
    """
    Enable LangSmith tracing only when both required vars are set.
    Silent no-op if either is missing — code runs identically without them.
    """
    tracing_on = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
    api_key    = os.getenv("LANGCHAIN_API_KEY", "")
    if tracing_on and api_key:
        os.environ.setdefault(
            "LANGCHAIN_PROJECT",
            os.getenv("LANGCHAIN_PROJECT", "instacart-recsys"),
        )
        return True
    return False


# ── CrewAI wrapper ─────────────────────────────────────────────────────────
def run_crew_pipeline(user_id: int, intent: str) -> dict:
    """
    Run the RAG pipeline via CrewAI if available.

    CrewAI role here is a thin coordination layer — it does NOT reimplement
    pipeline logic. The actual work is done by run_pipeline() from graph.py.
    CrewAI adds an LLM-generated summary of the pipeline result (audit layer).

    Falls back silently to run_pipeline() if crewai is not installed.
    """
    langsmith_active = _setup_langsmith()
    if langsmith_active:
        project = os.environ.get("LANGCHAIN_PROJECT", "instacart-recsys")
        print(f"[crew_runner] LangSmith tracing enabled → project={project}")

    try:
        from crewai import Agent, Crew, Process, Task
        from langchain_openai import ChatOpenAI
        crewai_available = True
    except ImportError:
        crewai_available = False

    if not crewai_available:
        print("[crew_runner] crewai not installed — running direct graph pipeline")
        from rag.graph import run_pipeline
        return run_pipeline(user_id, intent)

    # ── run the actual pipeline first (all logic lives in graph.py) ─────
    from rag.graph import run_pipeline
    result = run_pipeline(user_id, intent)

    # ── CrewAI: single agent that summarises the pipeline output ─────────
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    coordinator = Agent(
        role="Retail Ops RAG Coordinator",
        goal=(
            "Produce a concise audit summary for a retail recommendation pipeline run: "
            "what was recommended, what was substituted, which policies were cited, "
            "and any compliance flags."
        ),
        backstory=(
            "You are an experienced retail operations auditor that reviews recommendation "
            "pipeline outputs for policy compliance and delivery risk."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=False,
    )

    task_summary = Task(
        description=(
            "Summarise the following pipeline output for user_id={user_id} "
            "(intent: '{intent}'). "
            "Recommendation count: {recommendation_count}. "
            "Substitution count: {substitution_count}. "
            "Warning count: {warnings_count}. "
            "Top citations found: {top_citations}. "
            "Produce a 3-5 sentence audit summary covering: "
            "(a) recommendation quality signal, "
            "(b) substitution compliance, "
            "(c) policy areas covered by citations."
        ),
        agent=coordinator,
        expected_output=(
            "A 3-5 sentence plain-English audit summary covering recommendation quality, "
            "substitution compliance, and policy coverage."
        ),
    )

    crew = Crew(
        agents=[coordinator],
        tasks=[task_summary],
        process=Process.sequential,
        verbose=False,
    )

    try:
        crew_output = crew.kickoff(inputs={
            "user_id":             str(user_id),
            "intent":              intent,
            "recommendation_count": str(len(result["final_recommendations"])),
            "substitution_count":  str(len(result["substitutions"])),
            "warnings_count":      str(len(result["warnings"])),
            "top_citations":       ", ".join(result["citations"][:5]) or "none",
        })
        result["crew_summary"] = str(crew_output)
    except Exception as e:
        result["crew_summary"] = f"[crew summary unavailable: {e}]"

    return result


# ── entry point ────────────────────────────────────────────────────────────
def main() -> None:
    results = []
    for case in DEMO_CASES:
        uid, intent = case["user_id"], case["intent"]
        print(f"\n>> Crew pipeline  user_id={uid}  intent='{intent}' ...")
        try:
            result = run_crew_pipeline(uid, intent)
        except Exception as e:
            print(f"   [ERROR] {e}")
            result = {**case, "error": str(e)}

        results.append(result)
        print(f"   citations    : {result.get('citations', [])[:3]}")
        print(f"   warnings     : {len(result.get('warnings', []))}")
        if "crew_summary" in result:
            summary_preview = result["crew_summary"][:120].replace("\n", " ")
            print(f"   crew_summary : {summary_preview}...")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n>> Outputs written → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
