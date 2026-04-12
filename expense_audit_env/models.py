"""
Expense Audit Environment — Pydantic Models.

Defines the typed Action, Observation, and supporting models
for the corporate expense report auditing environment.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict
from openenv.core import Action, Observation


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

class AuditObservation(Observation):
    """
    Observation returned to the agent at each step.
    Extends base Observation with environment-specific results.
    """
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    # Episode context
    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: Optional[float] = Field(default=None, description="Reward from last action")
    # Use metadata as the container for environment-specific data 
    # as per strict Observation schema requirements.


# ── Action ─────────────────────────────────────────────────────────────────

class AuditAction(Action):
    """
    Action parameters for the audit environment.
    Extends base Action with audited business logic.
    """
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
    )
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
