"""
FastAPI application for the Expense Audit Environment.

Uses the standard openenv.core.env_server factory to ensure full 
spec compliance for the Reset, Step, and State endpoints.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app
from server.audit_environment import AuditEnvironment
from models import AuditAction, AuditObservation

# Create the spec-compliant FastAPI application
# create_fastapi_app takes a factory function for the environment
app = create_fastapi_app(
    env=lambda: AuditEnvironment(),
    action_cls=AuditAction,
    observation_cls=AuditObservation,
)

# Add custom utility routes
@app.get("/metadata")
async def metadata():
    return {
        "name": "expense_audit_env",
        "description": "Corporate expense report auditing environment for AI agents",
        "version": "0.1.0",
        "tasks": ["basic_audit", "standard_audit", "forensic_audit"],
    }

@app.get("/schema")
async def schema():
    return {
        "action_schema": AuditAction.model_json_schema(),
        "observation_schema": AuditObservation.model_json_schema(),
    }

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
