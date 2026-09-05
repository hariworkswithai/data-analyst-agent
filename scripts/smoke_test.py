"""Manual smoke test of the full multi-agent workflow.

Usage:  python scripts/smoke_test.py [llm]
If "llm" is passed and OPENROUTER_API_KEY is configured, all agents
run with real hosted LLM intelligence.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / "backend" / ".env")

from backend.config import BASE_DIR, config
from backend.services.llm import LLMService
from backend.services.orchestrator import Orchestrator
from backend.tools.profile import compute_dataset_profile
from backend.utils.validation import validate_and_load

USE_LLM = len(sys.argv) > 1 and sys.argv[1].lower() == "llm"

ASSERTED = []


def check(name, cond, detail=""):
    status = "OK " if cond else "FAIL"
    ASSERTED.append(cond)
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))


async def main():
    sample = BASE_DIR / "sample_data" / "sales.csv"
    loaded = validate_and_load(str(sample), "sales.csv")
    check("dataset_loaded", loaded.rows == 126, f"rows={loaded.rows}, cols={loaded.columns}")

    profile = compute_dataset_profile(loaded.df)
    check("profile_quality_score", "quality_score" in profile, str(profile["quality_score"]))

    llm = None
    if USE_LLM:
        try:
            llm = LLMService()
        except Exception as exc:  # noqa: BLE001
            print("LLM not available:", exc)

    events = []

    def emit(event, payload):
        events.append((event, payload))

    orch = Orchestrator(
        df=loaded.df,
        profile=profile,
        question="Analyze sales performance and find regions and products that need attention.",
        filename="sales.csv",
        llm=llm,
        emit=emit,
    )
    state = await orch.run()

    print("\n--- STATE ---")
    print("status:", state.status)
    print("review_status:", state.review_status)
    print("total_steps:", state.total_steps)
    print("mode:", "LLM" if USE_LLM else "deterministic")
    print("agents:", list(state.agent_results.keys()))
    for name, res in state.agent_results.items():
        print(f"  {name}: status={res.status} conf={res.confidence} summary={res.summary[:70]}")

    check("workflow_completed", state.status == "completed")
    check("reviewer_present", "reviewer" in state.agent_results)
    check("reporter_present", "reporter" in state.agent_results)

    cleaner = state.agent_results.get("cleaner")
    check("cleaner_completed", cleaner is not None and cleaner.status == "completed")
    check("cleaner_findings", cleaner is not None and cleaner.output and len(cleaner.output.get("findings", [])) > 0,
          f"{len(cleaner.output.get('findings', []))} findings" if cleaner and cleaner.output else "missing")

    analyst = state.agent_results.get("analyst")
    check("analyst_completed", analyst is not None and analyst.status == "completed")
    check("analyst_findings", analyst is not None and analyst.output and len(analyst.output.get("findings", [])) > 0,
          f"{len(analyst.output.get('findings', []))} findings" if analyst and analyst.output else "missing")

    reviewer = state.agent_results.get("reviewer")
    check("reviewer_decision", bool(reviewer),
          f"decision={reviewer.output.get('decision')}" if reviewer and reviewer.output else "missing")

    visualizer = state.agent_results.get("visualizer")
    charts = visualizer.output.get("charts", []) if visualizer and visualizer.output else []
    check("visualizer_charts", len(charts) >= 2, f"{len(charts)} charts")
    for c in charts:
        p = BASE_DIR / "generated" / c["relative_path"]
        check("chart_file_exists", p.exists(), c["relative_path"])

    check("report_generated", len(state.final_report) > 200, f"{len(state.final_report)} chars")
    check("report_file", Path(state.report_path).exists() if state.report_path else False)

    print("\n--- EXECUTIVE SUMMARY ---")
    exec_idx = state.final_report.find("## Executive Summary")
    if exec_idx >= 0:
        snippet = state.final_report[exec_idx:exec_idx + 900]
        print(snippet)

    print(f"\n{'ALL PASSED' if all(ASSERTED) else 'SOME FAILED'}: {sum(ASSERTED)}/{len(ASSERTED)} checks")


if __name__ == "__main__":
    asyncio.run(main())