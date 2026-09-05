"""Orchestrator - executes the multi-agent workflow with bounded limits.

Workflow:
  USER -> MANAGER plan -> CLEANER/ANALYST/VISUALIZER (parallel-safe)
  -> REVIEWER (with correction loop, max 3 cycles)
  -> REPORTER -> final state

Limits: MAX_AGENT_STEPS, MAX_REVIEW_CYCLES. On reaching a limit the
workflow stops safely with whatever partial results exist.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable

import pandas as pd

from backend.agents import (
    AgentContext,
    AnalystAgent,
    CleanerAgent,
    ManagerAgent,
    ReportAgent,
    ReviewerAgent,
    VisualizationAgent,
)
from backend.config import config
from backend.models import AgentResult, WorkflowState
from backend.services.llm import LLMService
from backend.tools.profile import profile_to_llm_text
from backend.utils.errors import LLMConfigError, WorkflowLimitError

logger = logging.getLogger("analyst.orchestrator")

AGENT_FACTORY = {
    "cleaner": CleanerAgent,
    "analyst": AnalystAgent,
    "visualizer": VisualizationAgent,
}


class Orchestrator:
    def __init__(
        self,
        df: pd.DataFrame,
        profile: dict,
        question: str,
        filename: str = "",
        llm: LLMService | None = None,
        emit: Callable[[str, dict], None] | None = None,
        workflow_id: str | None = None,
    ):
        self.df = df
        self.profile = profile
        self.question = question or "Explore and summarize the dataset."
        self.filename = filename
        self.llm = llm
        self.emit = emit
        self.profile_text = profile_to_llm_text(profile)

        self.state = WorkflowState(
            workflow_id=workflow_id or uuid.uuid4().hex[:12],
            filename=filename,
            question=self.question,
            status="pending",
            review_status="pending",
            memory={
                "plan": {},
                "review_cycles": 0,
                "history": [],
                "dataset_overview": {
                    "rows": profile.get("overview", {}).get("rows"),
                    "columns": profile.get("overview", {}).get("columns"),
                    "quality_score": profile.get("quality_score", {}),
                    "dtypes": profile.get("overview", {}).get("dtypes", {}),
                    "column_names": profile.get("overview", {}).get("column_names", []),
                },
            },
        )

    # ------------------------------------------------------------------ events
    def _emit(self, event: str, payload: dict) -> None:
        if self.emit:
            try:
                self.emit(event, payload)
            except Exception:  # noqa: BLE001
                logger.warning("Event callback failed for %s", event)

    def _update_state(self, *, agent: str = "", task: str = "", message: str = "") -> None:
        self.state.current_agent = agent
        self.state.current_task = task
        if message:
            self.state.current_message = message
        self._emit("state", self.state.model_dump())

    def _agent_emit(self, event: str, agent: str, payload: dict) -> None:
        """Adapter from AgentContext.emit to orchestrator.emit."""
        if event == "agent_start":
            self._update_state(agent=agent, task=payload.get("task", ""), message=f"{agent} working")
            self._emit("agent_status", {"agent": agent, "status": "working"})
        elif event == "agent_done":
            self._emit("agent_status", {"agent": agent, "status": "completed"})
        elif event == "tool_result":
            self._emit(
                "activity",
                {
                    "level": "tool" if payload.get("ok") else "warn",
                    "message": (
                        f"{payload['tool']} ok" if payload.get("ok")
                        else f"{payload['tool']} failed"
                    ),
                    "agent": agent,
                },
            )

    # ------------------------------------------------------------------ run
    async def run(self) -> WorkflowState:
        self.state.status = "running"
        self._update_state(agent="manager", task="Creating analysis plan", message="Manager planning")

        try:
            await self._run_workflow()
        except WorkflowLimitError as exc:
            self.state.error = str(exc)
            self.state.status = "partial"
            self._emit("error", {"message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Workflow crashed: %s", exc)
            self.state.error = f"Workflow failed: {exc}"
            self.state.status = "failed"
            self._emit("error", {"message": self.state.error})
        return self.state

    async def _run_workflow(self) -> None:
        ctx = self._make_context()
        manager = ManagerAgent(llm=self.llm)
        plan = await manager.create_plan(ctx)
        self.state.memory["plan"] = plan.model_dump()
        self._emit("activity", {"level": "ok", "message": "Manager created analysis plan", "agent": "manager"})

        tasks = plan.tasks
        logger.info("[%s] Plan: %s", self.state.workflow_id, tasks)

        # Resolve task order honoring dependencies manually (agents run here
        # in assigned order; cleaner before analyst, analyst result feeds visualizer).
        task_by_agent = {t["agent"]: t for t in tasks}
        order = ["cleaner", "analyst", "visualizer"]

        for agent_name in order:
            if agent_name not in task_by_agent:
                if agent_name == "visualizer":
                    # visualizer is plan-optional
                    continue
                # Cleaner/Analyst are mandatory; run with a default task even
                # if the Manager omitted them (protects the review gate).
                task_by_agent[agent_name] = {
                    "agent": agent_name,
                    "task": {
                        "cleaner": "Inspect data quality",
                        "analyst": "Analyze the dataset and answer: " + self.question,
                    }[agent_name],
                    "priority": "high",
                }
            self._bump_steps(agent_name)
            task = task_by_agent[agent_name]
            result = await self._execute_agent(
                agent_name,
                AgentContext(
                    question=self.question,
                    task=task["task"],
                    df=self.df,
                    profile=self.profile,
                    profile_text=self.profile_text,
                    workflow_id=self.state.workflow_id,
                    prior_results=dict(self.state.agent_results),
                    emit=self._agent_emit,
                ),
            )
            self.state.agent_results[agent_name] = result
            self.state.completed_tasks.append(task["task"])
            if result.status != "completed":
                self.state.failed_tasks.append(task["task"])
            self._emit("state", self.state.model_dump())

        # ---- review loop
        await self._run_review_loop(ctx)
        self.state.memory["review_cycles"] = self.state.retry_count

        # ---- report after approval (or cycle limit with best effort)
        if self.state.review_status in ("approved", "cycle_limit", "pending"):
            self._bump_steps("reporter")
            self._update_state(agent="reporter", task="Writing final report", message="Report Agent writing")
            report_ctx = AgentContext(
                question=self.question,
                task="Write final report",
                df=self.df,
                profile=self.profile,
                profile_text=self.profile_text,
                workflow_id=self.state.workflow_id,
                prior_results=dict(self.state.agent_results),
                emit=self._agent_emit,
            )
            reporter = ReportAgent(llm=self.llm)
            report_result: AgentResult = await reporter.run(report_ctx)
            self.state.agent_results["reporter"] = report_result
            self.state.completed_tasks.append("Write final report")
            if report_result.output.get("markdown"):
                self.state.final_report = report_result.output["markdown"]
                self.state.report_path = report_result.output.get("report_path", "")
                self._emit("activity", {"level": "ok", "message": "Report Agent completed", "agent": "reporter"})
            else:
                self.state.failed_tasks.append("Write final report")
                self._emit("activity", {"level": "warn", "message": "Report Agent failed", "agent": "reporter"})

        self.state.status = "completed"
        charts = self._collect_charts()
        self.state.charts = charts
        self._update_state(message="Workflow completed")

    async def _execute_agent(self, agent_name: str, ctx: AgentContext) -> AgentResult:
        self._update_state(agent=agent_name, task=ctx.task, message=f"{agent_name} working")
        agent_cls = AGENT_FACTORY[agent_name]
        agent = agent_cls(llm=self.llm)
        logger.info("[%s] Starting agent %s", self.state.workflow_id, agent_name)
        try:
            result = await agent.run(ctx)
        except WorkflowLimitError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%s] agent %s crashed", self.state.workflow_id, agent_name)
            result = AgentResult(agent=agent_name, status="failed", error=str(exc), output={})
        logger.info("[%s] agent %s -> %s", self.state.workflow_id, agent_name, result.status)
        if result.status != "completed":
            self._emit("agent_status", {"agent": agent_name, "status": "failed"})
        return result

    async def _run_review_loop(self, ctx: AgentContext) -> None:
        self._emit("activity", {"level": "ok", "message": "All primary agents completed", "agent": "manager"})
        cycles = 0
        reviewer = ReviewerAgent(llm=self.llm)

        while cycles <= config.MAX_REVIEW_CYCLES:
            if cycles > 0:
                self.state.retry_count = cycles
            self._update_state(agent="reviewer", task="Reviewing agent results", message=f"Review round {cycles + 1}")
            self._emit("activity", {"level": "ok", "message": f"Reviewer inspecting results (round {cycles + 1})", "agent": "reviewer"})

            review_ctx = AgentContext(
                question=self.question,
                task="Review results",
                df=self.df,
                profile=self.profile,
                profile_text=self.profile_text,
                workflow_id=self.state.workflow_id,
                prior_results=dict(self.state.agent_results),
                critic_notes="",
                emit=self._agent_emit,
            )
            review_result = await reviewer.run(review_ctx)
            self.state.agent_results["reviewer"] = review_result
            decision = review_result.output.get("decision", "NEEDS_CORRECTION")

            if decision == "APPROVED":
                self.state.review_status = "approved"
                self._emit("activity", {"level": "ok", "message": "Reviewer approved the analysis", "agent": "reviewer"})
                return

            if cycles >= config.MAX_REVIEW_CYCLES:
                self.state.review_status = "cycle_limit"
                self.state.error = (
                    f"Review could not be fully resolved within {config.MAX_REVIEW_CYCLES} cycles."
                )
                self._emit("activity", {"level": "warn", "message": "Review cycle limit reached", "agent": "reviewer"})
                return

            cycles += 1
            self._emit("activity", {"level": "warn", "message": f"Reviewer requested corrections (round {cycles})", "agent": "reviewer"})

            # route corrections through Manager
            directives = review_result.output.get("directives", [])
            if not directives:
                # no actionable directive: treat as approved to avoid a loop
                self.state.review_status = "approved"
                self._emit("activity", {"level": "ok", "message": "Reviewer had no directives; approving", "agent": "manager"})
                return

            manager = ManagerAgent(llm=self.llm)
            correction_tasks = manager.correction_tasks(directives)
            for task in correction_tasks[:3]:
                agent_name = task["agent"]
                self._bump_steps(agent_name)
                self._update_state(agent=agent_name, task=task["task"], message=f"Re-working: {task['task']}")
                self._emit("activity", {"level": "warn", "message": f"{agent_name} correcting work", "agent": agent_name})
                fix_ctx = AgentContext(
                    question=self.question,
                    task=task["task"],
                    df=self.df,
                    profile=self.profile,
                    profile_text=self.profile_text,
                    workflow_id=self.state.workflow_id,
                    prior_results=dict(self.state.agent_results),
                    critic_notes=task["task"],
                    emit=self._agent_emit,
                )
                result = await self._execute_agent(agent_name, fix_ctx)
                self.state.agent_results[agent_name] = result

            self._emit("state", self.state.model_dump())

        self.state.review_status = "cycle_limit"

    # ------------------------------------------------------------------ limits
    def _bump_steps(self, agent: str) -> None:
        self.state.total_steps += 1
        if self.state.total_steps > config.MAX_AGENT_STEPS:
            raise WorkflowLimitError(
                f"Maximum agent steps ({config.MAX_AGENT_STEPS}) exceeded."
            )

    # ------------------------------------------------------------------ helpers
    def _make_context(self) -> AgentContext:
        return AgentContext(
            question=self.question,
            task="Create plan",
            df=self.df,
            profile=self.profile,
            profile_text=self.profile_text,
            workflow_id=self.state.workflow_id,
            prior_results=dict(self.state.agent_results),
            emit=self._agent_emit,
        )

    def _collect_charts(self) -> list[dict]:
        viz = self.state.agent_results.get("visualizer")
        if viz and viz.status == "completed":
            return viz.output.get("charts", [])
        return []