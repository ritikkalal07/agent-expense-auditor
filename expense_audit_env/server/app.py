"""
FastAPI application for the Expense Audit Environment.

Provides both the OpenEnv-standard endpoints and a custom stateful HTTP API
that maintains environment state across reset/step calls.

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import sys
import os
from typing import Any, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, Body
from pydantic import BaseModel, Field

from server.audit_environment import AuditEnvironment
from models import AuditAction


# ── Request/Response models ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    seed: Optional[int] = Field(default=None, description="Random seed")
    episode_id: Optional[str] = Field(default=None, description="Episode ID")
    task: Optional[str] = Field(default="basic_audit", description="Task name")


class StepRequest(BaseModel):
    action: Dict[str, Any] = Field(..., description="Action to execute")
    timeout_s: Optional[float] = Field(default=None, description="Timeout")


class EnvResponse(BaseModel):
    observation: Dict[str, Any] = Field(default_factory=dict)
    reward: Optional[float] = None
    done: bool = False


# ── Application ────────────────────────────────────────────────────────────

app = FastAPI(
    title="Expense Audit Environment",
    description="OpenEnv environment for corporate expense report auditing",
    version="0.1.0",
)

# Stateful environment instance (per-server singleton for HTTP API)
# For production multi-user, use WebSocket sessions instead.
_env: Optional[AuditEnvironment] = None


def _get_env() -> AuditEnvironment:
    global _env
    if _env is None:
        _env = AuditEnvironment()
    return _env


def _obs_to_dict(obs) -> Dict[str, Any]:
    """Convert Observation to a serializable dict."""
    if hasattr(obs, 'model_dump'):
        return obs.model_dump()
    return {"done": getattr(obs, 'done', False), "reward": getattr(obs, 'reward', None), "metadata": getattr(obs, 'metadata', {})}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset(request: ResetRequest = Body(default_factory=ResetRequest)):
    global _env
    _env = AuditEnvironment()
    obs = _env.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        task=request.task,
    )
    d = _obs_to_dict(obs)
    return EnvResponse(
        observation=d.get("metadata", d),
        reward=d.get("reward"),
        done=d.get("done", False),
    )


@app.post("/step")
async def step(request: StepRequest):
    env = _get_env()
    action_data = request.action

    # Remove metadata key if present (base Action field)
    action_data.pop("metadata", None)

    try:
        action = AuditAction(**action_data)
    except Exception as e:
        return EnvResponse(
            observation={"error": f"Invalid action: {e}", "feedback": f"Invalid action: {e}"},
            reward=-0.02,
            done=False,
        )

    obs = env.step(action)
    d = _obs_to_dict(obs)
    return EnvResponse(
        observation=d.get("metadata", d),
        reward=d.get("reward"),
        done=d.get("done", False),
    )


@app.get("/state")
async def state():
    env = _get_env()
    s = env.state
    return {
        "episode_id": s.episode_id,
        "step_count": s.step_count,
    }


@app.get("/schema")
async def schema():
    return {
        "action": AuditAction.model_json_schema(),
        "observation": {
            "description": "Audit observation with current report, policy, and feedback",
        },
    }


@app.get("/metadata")
async def metadata():
    return {
        "name": "expense_audit_env",
        "description": "Corporate expense report auditing environment for AI agents",
        "version": "0.1.0",
        "tasks": ["basic_audit", "standard_audit", "forensic_audit"],
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
