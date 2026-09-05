"""Base agent implementing the controlled tool-call loop.

Flow for every agent:
  AGENT decides needed tools  ->  structured JSON plan (LLM)
  Backend validates+executes  ->  tool results (Python)
  AGENT interprets results    ->  structured final output (LLM)

No LLM-generated code is ever executed. If the LLM is unavailable, the
agent degrades to a deterministic Python fallback using the same tools,
so the workflow continues without crashing.
"""

from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from backend.models import AgentResult
from backend.services.llm import LLMService
from backend.tools.registry import try_execute_tool
from backend.utils.errors import LLMError

logger = logging.getLogger("analyst.agents")


@dataclass
class AgentContext:
    """Everything an agent is allowed to see about the job."""

    question: str
    task: str
    df: pd.DataFrame
    profile: dict
    profile_text: str
    workflow_id: str = ""
    prior_results: dict[str, AgentResult] = field(default_factory=dict)
    critic_notes: str = ""
    emit: Callable[[str, str, dict], None] | None = None

    def story_so_far(self) -> str:
        """Compact summary of prior agent outputs for context."""
        import json

        parts = []
        for name, res in self.prior_results.items():
            head = res.summary or name
            parts.append(f"[{name}] {head}")
        return "\n".join(parts) if parts else "No prior agent results."


class ToolPlanAgent(ABC):
    """Base class: plan tools -> execute -> interpret."""

    name: str = "base"
    label: str = "Agent"
    icon: str = "🤖"
    max_operations: int = 8
    max_steps: int = 1
    temperature: float = 0.2
    interpret_max_tokens: int = 1500
    plan_max_tokens: int = 800

    def __init__(self, llm: LLMService | None = None):
        self.llm = llm

    # ---- prompts (override) ----
    @property
    def system_prompt(self) -> str:
        return "You are a helpful data analysis agent."

    def build_plan_prompt(self, ctx: AgentContext) -> str:  # pragma: no cover - override
        raise NotImplementedError

    def build_interpret_prompt(self, ctx: AgentContext, tool_outcomes: list[dict]) -> str:  # noqa: B027
        return "Return your findings as JSON."

    # ---- validators (override) ----
    def validate_plan(self, data: Any) -> dict:
        return _validated_plan(self, data)

    def validate_interpret(self, data: Any) -> dict:
        if not isinstance(data, dict):
            raise ValueError("Interpretation must be a JSON object.")
        return data

    # ---- fallback (override if needed) ----
    def deterministic_result(self, ctx: AgentContext) -> dict:
        return {"note": "No LLM analysis available.", "findings": []}

    # ---- main loop ----
    async def run(self, ctx: AgentContext) -> AgentResult:
        self._emit(ctx, "agent_start", {"agent": self.name, "task": ctx.task})
        if self.llm is None:
            logger.info("%s: no LLM configured, using deterministic output", self.name)
            return self._fallback(ctx)

        try:
            plan = await self._plan_tools(ctx)
            tool_outcomes = self._execute_tools(ctx, plan)
            output = await self._interpret(ctx, plan, tool_outcomes)
            output = self.postprocess(ctx, output, tool_outcomes)
        except LLMError as exc:
            logger.warning("%s LLM failed (%s); falling back to deterministic output.", self.name, exc)
            return self._fallback(ctx, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s unexpected failure: %s", self.name, exc)
            return self._fallback(ctx, error=str(exc))
        finally:
            self._emit(ctx, "agent_done", {"agent": self.name, "status": "done"})

        confidence = float(output.get("confidence", 0.7))
        confidence = max(0.0, min(1.0, confidence))
        result = AgentResult(
            agent=self.name,
            status="completed",
            output=output,
            confidence=confidence,
            summary=str(output.get("summary") or self._default_summary(output)),
        )
        return result

    async def _plan_tools(self, ctx: AgentContext) -> dict:
        try:
            data = await self.llm.chat_json(
                system=self.system_prompt,
                user=self.build_plan_prompt(ctx),
                validator=self.validate_plan,
                temperature=self.temperature,
                max_tokens=self.plan_max_tokens,
                max_attempts=2,
            )
        except LLMError:
            # Plan without LLM: run the agent's default operations.
            data = self.default_plan(ctx)
        ops = data.get("operations", [])
        logger.info("%s planned %d operations", self.name, len(ops))
        return data

    def _execute_tools(self, ctx: AgentContext, plan: dict) -> list[dict]:
        outcomes = []
        for op in plan.get("operations", []):
            tool = op.get("tool") or op.get("name")
            arguments = op.get("arguments") or {}
            reason = op.get("reason", "")
            outcome = try_execute_tool(ctx.df, tool, arguments)
            outcome["reason"] = reason
            outcomes.append(outcome)
            self._emit(
                ctx,
                "tool_result",
                {"agent": self.name, "tool": tool, "ok": outcome["ok"]},
            )
        return outcomes

    async def _interpret(self, ctx: AgentContext, plan: dict, outcomes: list[dict]) -> dict:
        data = await self.llm.chat_json(
            system=self.system_prompt + "\nRespond only with valid JSON.",
            user=self.build_interpret_prompt(ctx, outcomes),
            validator=self.validate_interpret,
            temperature=self.temperature,
            max_tokens=self.interpret_max_tokens,
            max_attempts=3,
        )
        return data

    # overridable hooks
    def default_plan(self, ctx: AgentContext) -> dict:  # pragma: no cover - override
        return {"operations": []}

    def postprocess(self, ctx: AgentContext, output: dict, outcomes: list[dict]) -> dict:
        return output

    def _expected_keys(self) -> tuple[set[str], set[str]]:  # (required, all allowed)  # noqa: B027
        return (set(), set())

    def _default_summary(self, output: dict) -> str:
        findings = output.get("findings") or []
        if isinstance(findings, list) and findings:
            return str(findings[0])[:200]
        return "Completed."

    def _fallback(self, ctx: AgentContext, error: str | None = None) -> AgentResult:
        output = self.deterministic_result(ctx)
        return AgentResult(
            agent=self.name,
            status="completed",
            output=output,
            confidence=0.4,
            summary=str(output.get("summary") or "Completed (fallback mode)."),
            error=error,
        )

    def _emit(self, ctx: AgentContext, event: str, payload: dict) -> None:
        if ctx.emit:
            try:
                ctx.emit(event, self.name, payload)
            except Exception:  # noqa: BLE001
                pass


def _validated_plan(agent: ToolPlanAgent, data: Any) -> dict:
    """Ensure the LLM's plan is a well-formed operations list."""
    if isinstance(data, dict) and "operations" in data and isinstance(data["operations"], list):
        ops = []
        for op in data["operations"][: agent.max_operations]:
            if isinstance(op, dict) and ("tool" in op or "name" in op):
                ops.append(
                    {
                        "tool": op.get("tool", op.get("name")),
                        "arguments": op.get("arguments") if isinstance(op.get("arguments"), dict) else {},
                        "reason": str(op.get("reason", "")),
                    }
                )
        return {"operations": ops, "reasoning": str(data.get("reasoning", ""))}
    raise ValueError("Plan must contain an 'operations' list.")

# ---------- helper validators shared by agents ----------


def require_dict_keys(allowed: set[str], required: set[str] | None = None, list_field: str | None = None):
    """Build a validator that prunes unknown keys from LLM output."""

    def validator(data):
        if not isinstance(data, dict):
            raise ValueError("Output must be a JSON object.")
        clean = {k: v for k, v in data.items() if k in allowed}
        if required:
            missing = [k for k in required if k not in data]
            if missing:
                raise ValueError(f"Missing required keys: {missing}")
        if list_field and list_field in data and not isinstance(data[list_field], list):
            raise ValueError(f"'{list_field}' must be a list.")
        return clean

    return validator


def findings_validator(data):
    """Generic validator ensuring a 'findings' list of labeled items."""
    validator = require_dict_keys(
        allowed={"findings", "summary", "confidence", "statistics", "notes", "evidence", "issues", "anomalies", "correlations", "charts", "recommendations"},
        required={"findings"},
        list_field="findings",
    )
    cleaned = validator(data)

    validated_findings = []
    for item in cleaned.get("findings", [])[:12]:
        if isinstance(item, dict):
            okay_item = {
                "title": str(item.get("title", "Finding")),
                "detail": str(item.get("detail", "")),
                "evidence": str(item.get("evidence", "")),
                "severity": (item.get("severity") or "info"),
            }
            validated_findings.append(okay_item)
        elif isinstance(item, str):
            validated_findings.append({"title": item[:200], "detail": "", "evidence": "", "severity": "info"})
    cleaned["findings"] = validated_findings
    if not validated_findings:
        raise ValueError("No valid findings found in output.")
    return cleaned