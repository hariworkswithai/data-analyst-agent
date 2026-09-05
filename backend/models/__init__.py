"""Pydantic models for agent results, workflow state, and API payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentResult(BaseModel):
    """Structured output produced by an agent."""

    agent: str
    status: str  # completed | failed | skipped
    output: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    error: str | None = None
    summary: str = ""


class WorkflowState(BaseModel):
    """Explicit workflow agent-state model.

    state  = current workflow status (what is happening now)
    We also accept a small 'memory' dict for reusable context.
    """

    workflow_id: str
    filename: str = ""
    question: str = ""
    status: str = "pending"  # pending|running|reviewing|completed|failed|partial
    current_agent: str = ""
    current_task: str = ""
    current_message: str = ""
    completed_tasks: list[str] = Field(default_factory=list)
    failed_tasks: list[str] = Field(default_factory=list)
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)
    review_status: str = "pending"  # pending|approved|needs_correction|cycle_limit
    retry_count: int = 0
    total_steps: int = 0
    memory: dict[str, Any] = Field(default_factory=dict)
    charts: list[dict[str, Any]] = Field(default_factory=list)
    final_report: str = ""
    report_path: str = ""
    json_path: str = ""
    error: str | None = None


class DatasetUploadInfo(BaseModel):
    """Metadata about an uploaded dataset (never the raw data)."""

    filename: str
    upload_path: str
    rows: int
    columns: int


class UploadResponse(BaseModel):
    survey_id: str
    upload: DatasetUploadInfo
    question: str = ""
    quality_score: int = 0


class AnalysisPlan(BaseModel):
    """The Manager's plan of tasks delegated to specialized agents."""

    objective: str = ""
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    notes: str = ""


class ToolRequest(BaseModel):
    """Structured tool-call request from an agent (LLM)."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class ToolCallout(BaseModel):
    """The LLM's planned operations block."""

    operations: list[ToolRequest] = Field(default_factory=list)
    reasoning: str = ""


class FinalReport(BaseModel):
    """Output of the Report Agent."""

    markdown: str
    sections: dict[str, Any] = Field(default_factory=dict)