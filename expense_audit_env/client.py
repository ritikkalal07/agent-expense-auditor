"""
Expense Audit Environment Client.

This module provides the client for connecting to an Expense Audit Environment server.
"""

from openenv.core.env_client import EnvClient
from models import AuditAction, AuditObservation


class ExpenseAuditEnv(EnvClient):
    """
    Client for the Expense Audit Environment.

    Example:
        >>> async with ExpenseAuditEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset()
        ...     result = await env.step(AuditAction(
        ...         action_type="flag_violation",
        ...         item_id="EXP-001",
        ...         violation_type="over_limit",
        ...         reason="Meal exceeds $75 limit"
        ...     ))

    Example with Docker:
        >>> env = await ExpenseAuditEnv.from_docker_image("expense-audit-env:latest")
        >>> result = await env.reset()
    """

    pass  # EnvClient provides all needed functionality
