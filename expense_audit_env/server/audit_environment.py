"""
Expense Audit Environment — Core environment implementation.

Implements the OpenEnv Environment interface for corporate expense report auditing.
"""

import sys
import os
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

from models import AuditAction, AuditObservation, ExpenseReport, PolicyRule
from server.scenarios import ReportAnnotation, generate_scenario
from server.graders import grade_audit


class AuditEnvironment(Environment):
    """
    Corporate Expense Report Auditing Environment.

    The agent receives expense reports and must:
    1. Review each line item for policy compliance
    2. Flag violations with correct type
    3. Approve clean items
    4. Issue approve/reject verdicts on each report

    Supports 3 tasks: basic_audit, standard_audit, forensic_audit.
    """

    def __init__(self):
        """Initialize the audit environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_name: str = "basic_audit"
        self._seed: int = 42

        # Scenario data
        self._reports: List[ExpenseReport] = []
        self._annotations: List[ReportAnnotation] = []
        self._policy: List[PolicyRule] = []
        self._max_steps: int = 30

        # Agent tracking
        self._current_report_idx: int = 0
        self._agent_flags: Dict[str, Dict[str, str]] = {}  # {report_id: {item_id: violation_type}}
        self._agent_verdicts: Dict[str, str] = {}  # {report_id: "approve"|"reject"}
        self._agent_approvals: Dict[str, Set[str]] = {}  # {report_id: {item_ids}}
        self._audit_history: List[str] = []
        self._episode_rewards: List[float] = []
        self._done: bool = False
        self._last_feedback: str = ""

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment for a new audit episode.

        Args:
            seed: Random seed for reproducibility
            episode_id: Custom episode ID
            task: Task name — "basic_audit", "standard_audit", or "forensic_audit"

        Returns:
            Initial observation with the first expense report to audit
        """
        self._seed = seed if seed is not None else 42
        self._task_name = task or kwargs.get("task", "basic_audit")
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )

        # Generate scenario
        self._reports, self._annotations, self._policy, self._max_steps = (
            generate_scenario(self._task_name, self._seed)
        )

        # Reset tracking
        self._current_report_idx = 0
        self._agent_flags = {}
        self._agent_verdicts = {}
        self._agent_approvals = {}
        self._audit_history = []
        self._episode_rewards = []
        self._done = False
        self._last_feedback = f"Welcome to the {self._task_name} audit task. You have {len(self._reports)} expense reports to review. Please examine each report's line items, flag any policy violations, approve clean items, and issue approve/reject verdicts."

        return self._make_observation(reward=0.0)

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute an audit action.

        Args:
            action: AuditAction to execute
            timeout_s: Optional timeout (unused)

        Returns:
            Observation after the action
        """
        self._state.step_count += 1

        if self._done:
            return self._make_observation(reward=0.0)

        # Parse action
        if isinstance(action, dict):
            try:
                audit_action = AuditAction(**action)
            except Exception as e:
                self._last_feedback = f"Invalid action format: {e}"
                self._episode_rewards.append(-0.02)
                return self._make_observation(reward=-0.02)
        elif isinstance(action, AuditAction):
            audit_action = action
        else:
            # Try to construct from action data
            try:
                action_data = action.model_dump() if hasattr(action, 'model_dump') else dict(action)
                # Remove metadata before parsing
                action_data.pop('metadata', None)
                audit_action = AuditAction(**action_data)
            except Exception as e:
                self._last_feedback = f"Could not parse action: {e}"
                self._episode_rewards.append(-0.02)
                return self._make_observation(reward=-0.02)

        # Check step limit
        if self._state.step_count > self._max_steps:
            self._done = True
            return self._finalize_episode()

        # Check if there are reports to audit
        if self._current_report_idx >= len(self._reports):
            self._done = True
            return self._finalize_episode()

        current_report = self._reports[self._current_report_idx]
        current_annotation = self._annotations[self._current_report_idx]
        rid = current_report.report_id

        # Initialize tracking for this report if needed
        if rid not in self._agent_flags:
            self._agent_flags[rid] = {}
        if rid not in self._agent_approvals:
            self._agent_approvals[rid] = set()

        reward = 0.0

        # ── Handle each action type ───────────────────────────────────────

        if audit_action.action_type == "flag_violation":
            reward = self._handle_flag(audit_action, current_report, current_annotation, rid)

        elif audit_action.action_type == "approve_item":
            reward = self._handle_approve_item(audit_action, current_report, current_annotation, rid)

        elif audit_action.action_type == "approve_report":
            reward = self._handle_verdict(rid, "approve", current_annotation)
            self._advance_report()

        elif audit_action.action_type == "reject_report":
            reward = self._handle_verdict(rid, "reject", current_annotation)
            self._advance_report()

        elif audit_action.action_type == "request_info":
            # Informational action — small cost to prevent spamming
            self._last_feedback = self._provide_info(audit_action, current_report)
            reward = -0.01
            self._audit_history.append(
                f"Step {self._state.step_count}: Requested info on {audit_action.item_id or 'report'} — {audit_action.reason}"
            )

        elif audit_action.action_type == "next_report":
            # Skip to next report without verdict — penalize
            if rid not in self._agent_verdicts:
                self._agent_verdicts[rid] = "none"
                self._last_feedback = "Moved to next report without issuing a verdict. This counts as an incorrect verdict."
                reward = -0.10
            self._advance_report()

        else:
            self._last_feedback = f"Unknown action type: {audit_action.action_type}"
            reward = -0.02

        self._episode_rewards.append(reward)

        # Check if episode is done
        if self._current_report_idx >= len(self._reports):
            self._done = True
            return self._finalize_episode()

        return self._make_observation(reward=reward)

    def _handle_flag(
        self,
        action: AuditAction,
        report: ExpenseReport,
        annotation: ReportAnnotation,
        rid: str,
    ) -> float:
        """Handle a flag_violation action."""
        item_id = action.item_id
        violation_type = action.violation_type or "policy_violation"

        if not item_id:
            self._last_feedback = "flag_violation requires an item_id."
            return -0.02

        # Check item exists
        valid_ids = {item.item_id for item in report.expenses}
        if item_id not in valid_ids:
            self._last_feedback = f"Item {item_id} not found in report {rid}."
            return -0.02

        # Check if already flagged
        if item_id in self._agent_flags[rid]:
            self._last_feedback = f"Item {item_id} was already flagged."
            return -0.01

        # Record the flag
        self._agent_flags[rid][item_id] = violation_type

        # Check against ground truth
        true_violation_ids = {v.item_id for v in annotation.violations}
        if item_id in true_violation_ids:
            # Correct flag!
            true_type = None
            for v in annotation.violations:
                if v.item_id == item_id:
                    true_type = v.violation_type
                    break
            if violation_type == true_type:
                self._last_feedback = f"Correctly flagged {item_id} as {violation_type}. Good work!"
                reward = 0.15
            else:
                self._last_feedback = f"Correctly identified a violation in {item_id}, but the type '{violation_type}' may not be the best classification."
                reward = 0.10
        else:
            # False positive
            self._last_feedback = f"Item {item_id} appears to be compliant. False positive flag."
            reward = -0.10

        self._audit_history.append(
            f"Step {self._state.step_count}: Flagged {item_id} as {violation_type} — {action.reason} (reward: {reward:+.2f})"
        )
        return reward

    def _handle_approve_item(
        self,
        action: AuditAction,
        report: ExpenseReport,
        annotation: ReportAnnotation,
        rid: str,
    ) -> float:
        """Handle an approve_item action."""
        item_id = action.item_id

        if not item_id:
            self._last_feedback = "approve_item requires an item_id."
            return -0.02

        valid_ids = {item.item_id for item in report.expenses}
        if item_id not in valid_ids:
            self._last_feedback = f"Item {item_id} not found in report {rid}."
            return -0.02

        if item_id in self._agent_approvals[rid]:
            self._last_feedback = f"Item {item_id} was already approved."
            return -0.01

        self._agent_approvals[rid].add(item_id)

        # Check against ground truth
        true_violation_ids = {v.item_id for v in annotation.violations}
        if item_id in true_violation_ids:
            # Approved a violation — bad!
            self._last_feedback = f"Item {item_id} actually has a policy violation that was missed."
            reward = -0.15
        else:
            self._last_feedback = f"Item {item_id} correctly approved."
            reward = 0.05

        self._audit_history.append(
            f"Step {self._state.step_count}: Approved {item_id} — {action.reason} (reward: {reward:+.2f})"
        )
        return reward

    def _handle_verdict(
        self,
        rid: str,
        verdict: str,
        annotation: ReportAnnotation,
    ) -> float:
        """Handle an approve_report or reject_report action."""
        self._agent_verdicts[rid] = verdict
        expected = "reject" if annotation.should_reject else "approve"

        if verdict == expected:
            self._last_feedback = f"Report {rid}: Correct verdict — {verdict}."
            reward = 0.10
        else:
            self._last_feedback = f"Report {rid}: Incorrect verdict. You chose '{verdict}' but it should have been '{expected}'."
            reward = -0.10

        self._audit_history.append(
            f"Step {self._state.step_count}: Verdict on {rid}: {verdict} (expected: {expected}, reward: {reward:+.2f})"
        )
        return reward

    def _advance_report(self):
        """Move to the next report."""
        self._current_report_idx += 1

    def _provide_info(self, action: AuditAction, report: ExpenseReport) -> str:
        """Provide additional information about an item or report."""
        if action.item_id:
            for item in report.expenses:
                if item.item_id == action.item_id:
                    return (
                        f"Item {item.item_id}: {item.category} — {item.vendor} — "
                        f"${item.amount:.2f} on {item.date}. "
                        f"Receipt: {'Present' if item.receipt_present else 'Missing'}. "
                        f"Receipt details: {item.receipt_description or 'N/A'}"
                    )
            return f"Item {action.item_id} not found in current report."
        return (
            f"Report {report.report_id}: {report.employee_name} ({report.department}). "
            f"Purpose: {report.business_purpose}. Total: ${report.total_amount:.2f}. "
            f"Items: {len(report.expenses)}."
        )

    def _finalize_episode(self) -> Observation:
        """Compute final score and return terminal observation."""
        # Penalize reports without verdicts
        for ann in self._annotations:
            rid = ann.report_id
            if rid not in self._agent_verdicts:
                self._agent_verdicts[rid] = "none"

        # Run grader
        grade_result = grade_audit(
            annotations=self._annotations,
            agent_flags=self._agent_flags,
            agent_verdicts=self._agent_verdicts,
            agent_approvals=self._agent_approvals,
            steps_taken=self._state.step_count,
            max_steps=self._max_steps,
        )

        self._last_feedback = (
            f"Audit complete! Final score: {grade_result['score']:.4f}. "
            f"Detection F1: {grade_result['f1']:.4f}. "
            f"Verdict accuracy: {grade_result['verdict_score']:.4f}. "
            f"Correct flags: {grade_result['correct_flags']}/{grade_result['total_violations']}. "
            f"False positives: {grade_result['false_positives']}."
        )

        final_reward = grade_result["score"]
        self._episode_rewards.append(final_reward)

        obs = self._make_observation(reward=final_reward)
        obs.metadata["grade_result"] = grade_result
        return obs

    def _make_observation(self, reward: float) -> Observation:
        """Construct an observation from current state."""
        current_report = None
        if not self._done and self._current_report_idx < len(self._reports):
            current_report = self._reports[self._current_report_idx]

        rid = current_report.report_id if current_report else ""

        obs = AuditObservation(
            done=self._done,
            reward=reward,
            metadata={
                "task": self._task_name,
                "seed": self._seed,
                "episode_id": self._state.episode_id,
            },
            current_report=current_report,
            company_policy=self._policy,
            audit_history=self._audit_history[-10:],  # Last 10 entries
            flagged_items=list(self._agent_flags.get(rid, {}).keys()) if rid else [],
            approved_items=list(self._agent_approvals.get(rid, set())) if rid else [],
            reports_remaining=len(self._reports) - self._current_report_idx - (0 if self._done else 1),
            reports_completed=self._current_report_idx if not self._done else len(self._reports),
            current_step=self._state.step_count,
            max_steps=self._max_steps,
            feedback=self._last_feedback,
        )
        # Return as base Observation type for serialization
        return Observation(
            done=obs.done,
            reward=obs.reward,
            metadata={
                **obs.metadata,
                "current_report": obs.current_report.model_dump() if obs.current_report else None,
                "company_policy": [r.model_dump() for r in obs.company_policy],
                "audit_history": obs.audit_history,
                "flagged_items": obs.flagged_items,
                "approved_items": obs.approved_items,
                "reports_remaining": obs.reports_remaining,
                "reports_completed": obs.reports_completed,
                "current_step": obs.current_step,
                "max_steps": obs.max_steps,
                "feedback": obs.feedback,
            },
        )

    @property
    def state(self) -> State:
        """Get the current environment state."""
        return self._state

    def close(self) -> None:
        """Clean up resources."""
        pass
