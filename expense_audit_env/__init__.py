"""
Expense Audit Environment — AI agent environment for corporate expense report auditing.

This environment simulates the real-world task of auditing employee expense reports
for policy compliance. Agents must review line items, flag violations, and issue verdicts.

Example:
    >>> from expense_audit_env import ExpenseAuditEnv, AuditAction
    >>>
    >>> async with ExpenseAuditEnv(base_url="http://localhost:8000") as env:
    ...     result = await env.reset()
    ...     result = await env.step(AuditAction(
    ...         action_type="flag_violation",
    ...         item_id="EXP-001",
    ...         violation_type="over_limit",
    ...         reason="Meal exceeds $75 policy limit"
    ...     ))
"""

from models import AuditAction, AuditObservation, ExpenseItem, ExpenseReport, PolicyRule
from client import ExpenseAuditEnv

__all__ = [
    "ExpenseAuditEnv",
    "AuditAction",
    "AuditObservation",
    "ExpenseItem",
    "ExpenseReport",
    "PolicyRule",
]
