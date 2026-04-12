"""
Baseline Inference Script for Expense Audit Environment.

Runs an LLM agent against all 3 audit tasks and produces reproducible scores.
Uses the OpenAI API client with structured [START]/[STEP]/[END] logging.

Required environment variables:
    API_BASE_URL  — The API endpoint for the LLM
    MODEL_NAME    — The model identifier to use
    HF_TOKEN      — Your Hugging Face / API key

Usage:
    python inference.py
"""

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Configuration ──────────────────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
IMAGE_NAME = os.environ.get("IMAGE_NAME", "expense-audit-env:latest")
BENCHMARK = "expense_audit_env"

# Per-task configuration
TASK_CONFIGS = {
    "basic_audit": {"max_steps": 30, "max_total_reward": 3.0, "success_threshold": 0.5},
    "standard_audit": {"max_steps": 50, "max_total_reward": 5.0, "success_threshold": 0.4},
    "forensic_audit": {"max_steps": 80, "max_total_reward": 8.0, "success_threshold": 0.3},
}

SPACE_URL = os.environ.get("SPACE_URL", "http://localhost:7860")


# ── Logging helpers ────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    # Ensure booleans are lowercase
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    # Ensure booleans are lowercase
    success_str = "true" if success else "false"
    # Format rewards as comma-separated list
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(
        f"[END] success={success_str} steps={steps} score={score:.2f} "
        f"rewards={rewards_str}",
        flush=True,
    )


# ── System prompt ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert corporate expense auditor. You review employee expense reports for policy compliance.

For each expense report, you must:
1. Review each line item against the company policy rules provided
2. Flag any items that violate policy using the "flag_violation" action
3. Approve clean items using the "approve_item" action
4. Issue a final verdict on the report: "approve_report" (if clean) or "reject_report" (if violations found)

Available actions (respond with valid JSON):
- {"action_type": "flag_violation", "item_id": "<ID>", "violation_type": "<type>", "reason": "<why>"}
  violation_types: over_limit, missing_receipt, wrong_category, duplicate, policy_violation, suspicious_vendor, personal_expense, split_transaction, date_mismatch
- {"action_type": "approve_item", "item_id": "<ID>", "reason": "<why>"}
- {"action_type": "approve_report", "reason": "<why>"}
- {"action_type": "reject_report", "reason": "<why>"}
- {"action_type": "request_info", "item_id": "<ID>", "reason": "<what info needed>"}
- {"action_type": "next_report", "reason": "<why>"}

Strategy:
- Carefully check each item amount against category limits
- Verify receipts are present for expenses over $25
- Look for category mismatches (e.g., restaurant meals categorized as office supplies)
- Watch for suspicious vendors or personal expenses
- After reviewing items, issue approve_report or reject_report
- Be precise with violation types to maximize your score

RESPOND WITH ONLY A SINGLE JSON ACTION OBJECT. No other text."""


# ── LLM interaction ───────────────────────────────────────────────────────

def format_observation_for_llm(obs_data: Dict[str, Any]) -> str:
    """Format the observation data into a readable prompt for the LLM."""
    metadata = obs_data.get("metadata", {})
    report = metadata.get("current_report")
    policy = metadata.get("company_policy", [])
    feedback = metadata.get("feedback", "")
    history = metadata.get("audit_history", [])
    flagged = metadata.get("flagged_items", [])
    approved = metadata.get("approved_items", [])
    remaining = metadata.get("reports_remaining", 0)
    completed = metadata.get("reports_completed", 0)
    step = metadata.get("current_step", 0)
    max_steps = metadata.get("max_steps", 50)

    parts = []
    parts.append(f"=== Step {step}/{max_steps} | Reports completed: {completed} | Remaining: {remaining} ===")

    if feedback:
        parts.append(f"\nFeedback from last action: {feedback}")

    if policy:
        parts.append("\n--- COMPANY POLICY RULES ---")
        for rule in policy:
            limit_str = f" (Limit: ${rule['limit']:.2f})" if rule.get('limit') else ""
            parts.append(f"  [{rule['rule_id']}] ({rule['category']}): {rule['description']}{limit_str}")

    if report:
        parts.append(f"\n--- CURRENT REPORT: {report['report_id']} ---")
        parts.append(f"Employee: {report['employee_name']} ({report['department']})")
        parts.append(f"Purpose: {report['business_purpose']}")
        parts.append(f"Submitted: {report['submission_date']}")
        parts.append(f"Total: ${report['total_amount']:.2f}")
        parts.append(f"\nLine Items ({len(report['expenses'])}):")
        for item in report["expenses"]:
            receipt_status = "✓ Receipt" if item["receipt_present"] else "✗ NO RECEIPT"
            flag_status = " [FLAGGED]" if item["item_id"] in flagged else ""
            appr_status = " [APPROVED]" if item["item_id"] in approved else ""
            parts.append(
                f"  {item['item_id']}: {item['date']} | {item['category']:16s} | "
                f"{item['vendor']:25s} | ${item['amount']:>8.2f} | {receipt_status}{flag_status}{appr_status}"
            )
            parts.append(f"           Desc: {item['description']}")
            if item.get("receipt_description"):
                parts.append(f"           Receipt: {item['receipt_description']}")
    else:
        parts.append("\nNo more reports to review.")

    if flagged:
        parts.append(f"\nCurrently flagged items: {', '.join(flagged)}")
    if approved:
        parts.append(f"Currently approved items: {', '.join(approved)}")

    return "\n".join(parts)


def get_model_action(
    client: OpenAI,
    step: int,
    observation_text: str,
    last_reward: float,
    history: List[str],
) -> Dict[str, Any]:
    """Get the next action from the LLM."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Add recent history for context
    if history:
        context = "\n".join(history[-5:])
        messages.append({"role": "user", "content": f"Recent audit history:\n{context}"})

    messages.append({
        "role": "user",
        "content": f"Last reward: {last_reward:+.4f}\n\n{observation_text}\n\nWhat is your next action? Respond with ONLY a JSON object.",
    })

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=300,
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON from response (handle markdown code blocks)
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1])

        action_data = json.loads(content)
        return action_data

    except json.JSONDecodeError:
        print(f"[DEBUG] Failed to parse LLM response as JSON: {content[:200]}", flush=True)
        # Fallback: try to extract JSON from the text
        try:
            start = content.index("{")
            end = content.rindex("}") + 1
            action_data = json.loads(content[start:end])
            return action_data
        except (ValueError, json.JSONDecodeError):
            return {"action_type": "next_report", "reason": "Could not parse action"}

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "next_report", "reason": "Model error"}


# ── HTTP-based environment interaction ─────────────────────────────────────

import httpx


async def run_task(task_name: str, client: OpenAI) -> Dict[str, Any]:
    """Run a single task and return results."""
    config = TASK_CONFIGS[task_name]
    max_steps = config["max_steps"]
    max_total_reward = config["max_total_reward"]
    success_threshold = config["success_threshold"]

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        async with httpx.AsyncClient(timeout=60.0) as http_client:
            # Reset environment
            reset_resp = await http_client.post(
                f"{SPACE_URL}/reset",
                json={"seed": 42, "task": task_name},
            )
            reset_data = reset_resp.json()
            obs_data = reset_data
            last_reward = 0.0

            for step in range(1, max_steps + 1):
                if obs_data.get("done", False):
                    break

                # Format observation for LLM
                obs_text = format_observation_for_llm(obs_data)

                # Get action from LLM
                action_data = get_model_action(client, step, obs_text, last_reward, history)

                # Execute step
                step_resp = await http_client.post(
                    f"{SPACE_URL}/step",
                    json={"action": action_data},
                )
                step_data = step_resp.json()
                obs_data = step_data

                reward = step_data.get("reward", 0.0) or 0.0
                done = step_data.get("done", False)

                rewards.append(reward)
                steps_taken = step
                last_reward = reward

                action_str = json.dumps(action_data)
                log_step(step=step, action=action_str, reward=reward, done=done, error=None)

                history.append(f"Step {step}: {action_str} -> reward {reward:+.4f}")

                if done:
                    break

        # Compute final score
        score = sum(rewards) / max_total_reward if max_total_reward > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= success_threshold

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_name,
        "score": score,
        "success": success,
        "steps": steps_taken,
        "rewards": rewards,
    }


async def main() -> None:
    """Run baseline inference on all 3 tasks."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print("=" * 60, flush=True)
    print("Expense Audit Environment — Baseline Inference", flush=True)
    print(f"Model: {MODEL_NAME}", flush=True)
    print(f"API: {API_BASE_URL}", flush=True)
    print(f"Space: {SPACE_URL}", flush=True)
    print("=" * 60, flush=True)

    all_results = []
    for task_name in ["basic_audit", "standard_audit", "forensic_audit"]:
        print(f"\n{'-' * 40}", flush=True)
        print(f"Running task: {task_name}", flush=True)
        print(f"{'-' * 40}", flush=True)
        result = await run_task(task_name, client)
        all_results.append(result)
        print(f"\nTask {task_name}: score={result['score']:.4f} success={result['success']}", flush=True)

    # Summary
    print(f"\n{'=' * 60}", flush=True)
    print("RESULTS SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    for r in all_results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  {r['task']:20s} | Score: {r['score']:.4f} | Steps: {r['steps']:3d} | {status}", flush=True)

    avg_score = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n  Average Score: {avg_score:.4f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
