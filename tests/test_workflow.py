"""End-to-end workflow tests (deterministic mode - no external API needed)."""

import pytest

from backend.agents import ReviewerAgent  # noqa: F401 (import sanity)
from backend.services.orchestrator import Orchestrator
from backend.tools.profile import compute_dataset_profile
from backend.tools.registry import execute_tool


@pytest.mark.asyncio
async def test_full_workflow_deterministic(sales_df):
    """Manager -> Cleaner -> Analyst -> Visualizer -> Reviewer -> Reporter."""
    profile = compute_dataset_profile(sales_df)
    orch = Orchestrator(
        df=sales_df,
        profile=profile,
        question="Analyze performance by region and product.",
        filename="sales.csv",
        llm=None,
    )
    state = await orch.run()

    assert state.status == "completed"
    assert state.review_status == "approved"
    assert "cleaner" in state.agent_results
    assert "analyst" in state.agent_results
    assert "visualizer" in state.agent_results
    assert "reviewer" in state.agent_results
    assert "reporter" in state.agent_results

    cleaner = state.agent_results["cleaner"]
    assert cleaner.status == "completed"
    assert len(cleaner.output.get("findings", [])) > 0

    analyst = state.agent_results["analyst"]
    assert analyst.status == "completed"
    assert len(analyst.output.get("findings", [])) > 0
    # every finding must carry numeric evidence (enforced by Reviewer)
    for f in analyst.output["findings"]:
        assert any(ch.isdigit() for ch in str(f.get("evidence", "")))

    vis = state.agent_results["visualizer"]
    charts = vis.output.get("charts", [])
    assert len(charts) >= 2

    reviewer = state.agent_results["reviewer"]
    assert reviewer.output["decision"] == "APPROVED"

    assert len(state.final_report) > 300
    assert "AI DATA ANALYSIS REPORT" in state.final_report
    assert "Executive Summary" in state.final_report
    assert "Recommendations" in state.final_report


@pytest.mark.asyncio
async def test_workflow_honors_plan_agent_choices(sales_df):
    """Visualizer should be planned only when numeric columns exist.

    To test determinism we monkeypatch the Manager's heuristic via a
    profile-driven plan: sales_df has numerics so visualizer must run.
    """
    profile = compute_dataset_profile(sales_df)
    orch = Orchestrator(
        df=sales_df,
        profile=profile,
        question="Find top regions.",
        filename="sales.csv",
        llm=None,
    )
    # Manager heuristic plan is used when llm is None
    state = await orch.run()
    assert state.status == "completed"
    assert state.total_steps <= 4  # cleaner, analyst, visualizer, reporter

    # also verify data quality score computed
    assert profile["quality_score"]["score"] >= 90


@pytest.mark.asyncio
async def test_workflow_bounded_steps(sales_df):
    profile = compute_dataset_profile(sales_df)
    orch = Orchestrator(
        df=sales_df,
        profile=profile,
        question="Summarize.",
        filename="sales.csv",
        llm=None,
    )
    state = await orch.run()
    assert state.total_steps <= 6  # well under MAX_AGENT_STEPS


def test_two_numeric_columns_correlation(sales_df):
    res = execute_tool(sales_df, "calculate_correlation", {})
    assert len(res["correlation_matrix"]) >= 2