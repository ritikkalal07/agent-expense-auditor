"""
Expense Audit Environment — Pydantic Models.

Defines the typed Action, Observation, and supporting models
for the corporate expense report auditing environment.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ── Supporting Models ──────────────────────────────────────────────────────

class ExpenseItem(BaseModel):
    """A single line-item on an expense report."""

    item_id: str = Field(description="Unique identifier for this expense item")
    date: str = Field(description="Date the expense was incurred (ISO format)")
    category: str = Field(
        description="Expense category: meals, travel, lodging, office_supplies, entertainment, misc"
    )
    vendor: str = Field(description="Vendor / merchant name")
    amount: float = Field(description="Expense amount")
    currency: str = Field(default="USD", description="Currency code")
    description: str = Field(description="Description of the expense")
    receipt_present: bool = Field(description="Whether a receipt is attached")
    receipt_description: str = Field(
        default="", description="OCR-like text describing the receipt contents"
    )


class PolicyRule(BaseModel):
    """A single company expense policy rule."""

    rule_id: str = Field(description="Unique rule identifier")
    category: str = Field(description="Which expense category this rule applies to, or 'all'")
    description: str = Field(description="Human-readable rule text")
    limit: Optional[float] = Field(default=None, description="Dollar limit if applicable")


class ExpenseReport(BaseModel):
    """A complete expense report submitted by an employee."""

    report_id: str = Field(description="Unique report identifier")
    employee_name: str = Field(description="Name of the employee")
    employee_id: str = Field(description="Employee ID")
    department: str = Field(description="Department name")
    submission_date: str = Field(description="Date the report was submitted (ISO)")
    business_purpose: str = Field(description="Stated business purpose")
    expenses: List[ExpenseItem] = Field(description="Line items in this report")
    total_amount: float = Field(description="Sum of all expense amounts")


# ── Observation ────────────────────────────────────────────────────────────

class AuditObservation(BaseModel):
    """Observation returned to the agent at each step."""

    # Episode context
    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: Optional[float] = Field(default=None, description="Reward from last action")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    # Current report under review
    current_report: Optional[ExpenseReport] = Field(
        default=None, description="The expense report currently being audited"
    )

    # Policy reference
    company_policy: List[PolicyRule] = Field(
        default_factory=list, description="Company expense policy rules"
    )

    # Audit progress
    audit_history: List[str] = Field(
        default_factory=list, description="Log of actions taken so far in this episode"
    )
    flagged_items: List[str] = Field(
        default_factory=list,
        description="Item IDs flagged as violations on the current report",
    )
    approved_items: List[str] = Field(
        default_factory=list,
        description="Item IDs approved on the current report",
    )
    reports_remaining: int = Field(default=0, description="Reports left to audit")
    reports_completed: int = Field(default=0, description="Reports fully audited")
    current_step: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=50, description="Maximum steps allowed")

    # Feedback from previous action
    feedback: str = Field(
        default="", description="Feedback message about the last action taken"
    )


# ── Action ─────────────────────────────────────────────────────────────────

class AuditAction(BaseModel):
    """Action the agent can take during an audit."""

    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    action_type: Literal[
        "flag_violation",
        "approve_item",
        "approve_report",
        "reject_report",
        "request_info",
        "next_report",
    ] = Field(description="Type of audit action to perform")

    item_id: Optional[str] = Field(
        default=None,
        description="Expense item ID (required for flag_violation and approve_item)",
    )

    reason: str = Field(
        default="",
        description="Agent's justification or reasoning for the action",
    )

    violation_type: Optional[Literal[
        "over_limit",
        "missing_receipt",
        "wrong_category",
        "duplicate",
        "policy_violation",
        "suspicious_vendor",
        "personal_expense",
        "split_transaction",
        "date_mismatch",
    ]] = Field(
        default=None,
        description="Type of violation (required for flag_violation)",
    )
