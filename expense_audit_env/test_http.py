"""Test HTTP endpoints."""
import httpx
import json

BASE = "http://localhost:7860"

# Health check
r = httpx.get(f"{BASE}/health")
print(f"Health: {r.status_code} {r.json()}")

# Reset
r = httpx.post(f"{BASE}/reset", json={"seed": 42, "task": "basic_audit"})
print(f"\nReset: {r.status_code}")
data = r.json()
print(f"Done: {data.get('done')}")
print(f"Reward: {data.get('reward')}")
# In OpenEnv standard, the observation data is in 'metadata'
obs = data.get('metadata', {})
print(f"Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")

meta = obs
print(f"Feedback: {str(meta.get('feedback', ''))[:100]}")

# Step
# Try to find a real item id from the observation
expenses = obs.get("current_report", {}).get("expenses", [])
item_id = expenses[0].get("item_id", "EXP-000") if expenses else "EXP-000"

action = {
    "action_type": "flag_violation",
    "item_id": item_id,
    "violation_type": "over_limit",
    "reason": "Amount exceeds limit",
}
r = httpx.post(f"{BASE}/step", json={"action": action})
print(f"\nStep: {r.status_code}")
step_data = r.json()
print(f"Step response keys: {list(step_data.keys())}")
print(f"Done: {step_data.get('done')}")
print(f"Reward: {step_data.get('reward')}")

print("\n✓ HTTP endpoint tests complete!")
