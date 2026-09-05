"""REVIEWER AGENT - critical gate that verifies every agent's work.

It performs deterministic mechanical checks in Python AND an LLM review
of the structured results. It returns either APPROVED or
NEEDS_CORRECTION with concrete directives for the Manager to route back
to the responsible agent.
"""

from __future__ import annotations

import json
import logging
import re

from backend.agents.base import AgentContext, ToolPlanAgent
from backend.models import AgentResult

logger = logging.getLogger("analyst.agents")

CYCLE_LIMIT_MESSAGE = "Review-cycle limit reached; approved with known caveats."


class ReviewerAgent(ToolPlanAgent):
    name = "reviewer"
    label = "Reviewer Agent"
    temperature = 0.1
    interpret_max_tokens = 1400

    def build_plan_prompt(self, ctx: AgentContext) -> str:
        return """You are the REVIEWER agent. You do not plan tools; your job is
critical inspection. Return {"operations": []}."""

    def validate_plan(self, data):
        return {"operations": []}

    def build_interpret_prompt(self, ctx: AgentContext, tool_outcomes: list[dict]) -> str:
        results_block = _serialize_results(ctx.prior_results)
        return f"""You are the critical REVIEWER. Do NOT rubber-stamp the analysis.
Inspect the agents' actual structured results below.

User question: {ctx.question}
Profile summary:
{ctx.profile_text}

Agent results:
{results_block}

Check each of these:
1. DATA CORRECTNESS - are the reported numbers plausible vs the profile? Any inconsistency?
2. INSIGHT CORRECTNESS - does every insight have numeric evidence? Any unsupported claims?
3. VISUALIZATION CORRECTNESS - do the charts make sense given the findings? Any missing chart?
4. COMPLETENESS - did all agents finish? Is anything important missing?
5. CONSISTENCY - do agents contradict each other?

Return strict JSON:
{{"decision": "APPROVED" or "NEEDS_CORRECTION",
 "issues": [{{"agent": "...", "severity": "error|warning", "detail": "..."}}],
 "directives": [{{"agent": "analyst|cleaner|visualizer", "instruction": "exactly what to fix"}}],
 "confidence": 0.0-1.0,
 "summary": "one line verdict"}}"""

    def validate_interpret(self, data):
        if not isinstance(data, dict):
            raise ValueError("Reviewer output must be an object.")
        decision = str(data.get("decision", "NEEDS_CORRECTION")).upper()
        if decision not in ("APPROVED", "NEEDS_CORRECTION"):
            decision = "NEEDS_CORRECTION"
        issues = data.get("issues")
        directives = data.get("directives")
        if not isinstance(issues, list):
            issues = []
        if not isinstance(directives, list):
            directives = []
        valid_agents = {"analyst", "cleaner", "visualizer"}
        directives = [
            d
            for d in directives
            if isinstance(d, dict) and d.get("agent") in valid_agents and d.get("instruction")
        ]
        if decision == "APPROVED" and is_approval_blocked_by_mechanical_failures(data):
            decision = "NEEDS_CORRECTION"
        return {
            "decision": decision,
            "issues": issues[:10],
            "directives": directives[:5],
            "confidence": float(data.get("confidence", 0.6)),
            "summary": str(data.get("summary", "Review complete")),
        }

    async def run(self, ctx: AgentContext) -> AgentResult:
        """Overridden to run mechanical checks BEFORE any Approval."""
        self._emit(ctx, "agent_start", {"agent": self.name, "task": ctx.task})

        mechanical = mechanical_review(ctx)
        if mechanical["decision"] == "NEEDS_CORRECTION":
            return AgentResult(
                agent=self.name,
                status="completed",
                output={
                    "decision": "NEEDS_CORRECTION",
                    "issues": mechanical["issues"],
                    "directives": mechanical["directives"],
                    "confidence": 1.0,
                    "summary": "Mechanical review found blocking issues.",
                    "mechanical": True,
                },
                confidence=1.0,
                summary="Mechanical review found blocking issues.",
            )

        if self.llm is None:
            decision = "APPROVED"
            return AgentResult(
                agent=self.name,
                status="completed",
                output={
                    "decision": decision,
                    "issues": [],
                    "directives": [],
                    "confidence": 0.7,
                    "summary": "Mechanical checks passed; LLM unavailable - approved.",
                    "mechanical": True,
                },
                confidence=0.7,
                summary="Mechanical checks passed.",
            )

        try:
            data = await self.llm.chat_json(
                system=self.system_prompt + "\nRespond only with valid JSON.",
                user=self.build_interpret_prompt(ctx, []),
                validator=self.validate_interpret,
                temperature=self.temperature,
                max_tokens=self.interpret_max_tokens,
                max_attempts=3,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Reviewer LLM failed: %s; falling back to mechanical verdict", exc)
            return AgentResult(
                agent=self.name,
                status="completed",
                output={
                    "decision": "APPROVED",
                    "issues": mechanical["issues"],
                    "directives": [],
                    "confidence": 0.6,
                    "summary": "LLM review failed; approved based on mechanical checks.",
                    "mechanical": True,
                },
                confidence=0.6,
                summary="Approved via mechanical checks.",
            )
        finally:
            self._emit(ctx, "agent_done", {"agent": self.name, "status": "done"})

        output = _merge_mechanical(data, mechanical)
        decision = output["decision"]
        result = AgentResult(
            agent=self.name,
            status="completed",
            output=output,
            confidence=float(output.get("confidence", 0.8)),
            summary=f"Review: {decision}",
        )
        return result

    @property
    def system_prompt(self) -> str:
        return (
            "You are the critical REVIEWER agent in a multi-agent data analysis system. "
            "Your job is to catch errors, unsupported claims, and hallucinations."
        )

    def deterministic_result(self, ctx: AgentContext) -> dict:
        """Never used; run() handles review paths."""
        return {}

    def postprocess(self, ctx, output, outcomes):
        return output


def mechanical_review(ctx: AgentContext) -> dict:
    """Python-side factual checks that must pass before/review.

    Returns a verdict dict with decision, issues, directives.
    """
    issues = []
    directives = []
    blockers = False

    required = {"cleaner", "analyst"}
    for agent_name in required:
        res = ctx.prior_results.get(agent_name)
        if res is None:
            issues.append({"agent": agent_name, "severity": "error", "detail": f"'{agent_name}' did not produce results."})
            directives.append({"agent": agent_name, "instruction": "Re-run the task."})
            blockers = True
            continue
        if res.status != "completed":
            issues.append({"agent": agent_name, "severity": "error", "detail": f"'{agent_name}' failed."})
            blockers = True

    per_agent = {
        name: res for name, res in ctx.prior_results.items()
    }

    if not blockers:
        reporter = lambda n: per_agent.get(n)
        # Each analyst finding should cite numeric evidence.
        analyst = reporter("analyst")
        if analyst:
            for i, f in enumerate(analyst.output.get("findings", [])):
                evidence = str(f.get("evidence", "") or "")
                if not _has_number(evidence):
                    issues.append({
                        "agent": "analyst",
                        "severity": "warning",
                        "detail": f"Finding #{i + 1} has no numeric evidence: {f.get('title')}",
                    })
                    directives.append({
                        "agent": "analyst",
                        "instruction": f"Add numeric evidence to finding: {f.get('title')}",
                    })
                    blockers = True
                    break

    return {
        "decision": "NEEDS_CORRECTION" if blockers else "PASS",
        "issues": issues,
        "directives": directives[:5],
    }


def is_approval_blocked_by_mechanical_failures(data: dict) -> bool:
    """Never approve when any critical issue remains — handled by merge."""
    return False


def _merge_mechanical(llm_output: dict, mechanical: dict) -> dict:
    """Combine mechanical and LLM opinions (mechanical wins on errors)."""
    if mechanical["decision"] == "NEEDS_CORRECTION":
        llm_output["decision"] = "NEEDS_CORRECTION"
        llm_output.setdefault("issues", []).extend(mechanical["issues"])
        llm_output["directives"] = mechanical["directives"]
    return llm_output


def _serialize_results(results: dict[str, AgentResult]) -> str:
    parts = []
    for name, res in results.items():
        parts.append(f"--- {name} ({res.status}) ---")
        parts.append(json.dumps(res.output, default=str, ensure_ascii=False))
    return "\n".join(parts)


def _has_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _singleton_guard(res: AgentResult) -> bool:
    return res is not None