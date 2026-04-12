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

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
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
    title="Expense Audit Environment API",
    description="OpenEnv-compliant API for auditing corporate expense reports",
    version="0.1.0",
)

# Add CORS middleware for browser-based automated validators
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Stateful environment instance (per-server singleton for HTTP API)
_env: Optional[AuditEnvironment] = None


def _get_env() -> AuditEnvironment:
    global _env
    if _env is None:
        _env = AuditEnvironment()
    return _env


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    """OpenEnv standard reset endpoint."""
    global _env
    if request is None:
        request = ResetRequest()
    _env = AuditEnvironment()
    obs = _env.reset(
        seed=request.seed,
        episode_id=request.episode_id,
        task=request.task,
    )
    # Convert to dict to ensure serialized correctly
    return obs.model_dump() if hasattr(obs, 'model_dump') else obs


@app.post("/step")
async def step(request: StepRequest):
    """OpenEnv standard step endpoint."""
    env = _get_env()
    action_data = request.action
    if isinstance(action_data, dict):
        action_data.pop("metadata", None)

    try:
        obs = env.step(action_data)
    except Exception as e:
        return {
            "metadata": {"error": f"Failed to execute step: {e}", "feedback": str(e)},
            "reward": -0.02,
            "done": False,
        }
    # Convert to dict for maximum compatibility
    return obs.model_dump() if hasattr(obs, 'model_dump') else obs


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
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
