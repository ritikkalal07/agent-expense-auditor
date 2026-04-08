"""
Graders — Scoring logic for each audit task.

Computes precision/recall-based scores over violation detection
and report-level verdict accuracy.
"""

from typing import Dict, List, Set

from server.scenarios import ReportAnnotation


def grade_audit(
    annotations: List[ReportAnnotation],
    agent_flags: Dict[str, Dict[str, str]],
    agent_verdicts: Dict[str, str],
    agent_approvals: Dict[str, Set[str]],
    steps_taken: int,
    max_steps: int,
) -> Dict:
    """
    Grade an agent's audit performance.

    Args:
        annotations: Ground truth for all reports
        agent_flags: {report_id: {item_id: violation_type}} — items flagged by agent
        agent_verdicts: {report_id: "approve" | "reject"} — report-level verdicts
        agent_approvals: {report_id: {item_ids}} — items explicitly approved
        steps_taken: Number of steps the agent took
        max_steps: Maximum allowed steps

    Returns:
        Dict with score (0.0-1.0), breakdown, and details
    """
    total_true_violations = 0
    total_correct_flags = 0
    total_false_positives = 0
    total_missed_violations = 0
    total_correct_approvals = 0
    total_correct_verdicts = 0
    total_reports = len(annotations)
    report_details = []

    for ann in annotations:
        rid = ann.report_id

        # Ground truth
        true_violation_items = {v.item_id for v in ann.violations}
        true_clean_items = set(ann.clean_items)
        total_true_violations += len(true_violation_items)

        # Agent actions for this report
        flagged = agent_flags.get(rid, {})
        flagged_items = set(flagged.keys())
        verdict = agent_verdicts.get(rid, "none")
        approved = agent_approvals.get(rid, set())

        # Item-level scoring
        correct_flags = flagged_items & true_violation_items
        false_positives = flagged_items - true_violation_items
        missed = true_violation_items - flagged_items
        correct_approvals_for_report = approved & true_clean_items

        total_correct_flags += len(correct_flags)
        total_false_positives += len(false_positives)
        total_missed_violations += len(missed)
        total_correct_approvals += len(correct_approvals_for_report)

        # Check violation type accuracy (bonus)
        type_matches = 0
        for item_id in correct_flags:
            true_type = None
            for v in ann.violations:
                if v.item_id == item_id:
                    true_type = v.violation_type
                    break
            agent_type = flagged.get(item_id, "")
            if true_type and agent_type == true_type:
                type_matches += 1

        # Report-level verdict
        expected_verdict = "reject" if ann.should_reject else "approve"
        verdict_correct = verdict == expected_verdict
        if verdict_correct:
            total_correct_verdicts += 1

        report_details.append({
            "report_id": rid,
            "true_violations": len(true_violation_items),
            "correct_flags": len(correct_flags),
            "false_positives": len(false_positives),
            "missed_violations": len(missed),
            "type_matches": type_matches,
            "verdict": verdict,
            "expected_verdict": expected_verdict,
            "verdict_correct": verdict_correct,
        })

    # ── Compute composite score ────────────────────────────────────────────
    # Weight breakdown:
    #   - Violation detection (precision + recall): 50%
    #   - Report verdict accuracy: 30%
    #   - Efficiency: 10%
    #   - Violation type accuracy: 10%

    # Violation detection score (F1-like)
    if total_true_violations > 0:
        recall = total_correct_flags / total_true_violations
    else:
        recall = 1.0

    if total_correct_flags + total_false_positives > 0:
        precision = total_correct_flags / (total_correct_flags + total_false_positives)
    else:
        precision = 1.0 if total_true_violations == 0 else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    detection_score = f1  # 0.0 - 1.0

    # Verdict accuracy
    verdict_score = total_correct_verdicts / total_reports if total_reports > 0 else 0.0

    # Efficiency bonus
    if max_steps > 0:
        efficiency = max(0.0, 1.0 - (steps_taken / max_steps))
    else:
        efficiency = 0.0

    # Type accuracy bonus
    if total_correct_flags > 0:
        type_accuracy = sum(
            d["type_matches"] for d in report_details
        ) / total_correct_flags
    else:
        type_accuracy = 0.0

    # Weighted composite
    final_score = (
        0.50 * detection_score
        + 0.30 * verdict_score
        + 0.10 * efficiency
        + 0.10 * type_accuracy
    )

    # Clamp to [0.0, 1.0]
    final_score = max(0.0, min(1.0, final_score))

    return {
        "score": round(final_score, 4),
        "detection_score": round(detection_score, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "verdict_score": round(verdict_score, 4),
        "efficiency": round(efficiency, 4),
        "type_accuracy": round(type_accuracy, 4),
        "total_violations": total_true_violations,
        "correct_flags": total_correct_flags,
        "false_positives": total_false_positives,
        "missed_violations": total_missed_violations,
        "total_reports": total_reports,
        "correct_verdicts": total_correct_verdicts,
        "steps_taken": steps_taken,
        "max_steps": max_steps,
        "report_details": report_details,
    }
