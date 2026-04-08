"""Test HTTP endpoints."""
import httpx
import json

BASE = "http://localhost:8000"

# Health check
r = httpx.get(f"{BASE}/health")
print(f"Health: {r.status_code} {r.json()}")

# Reset
r = httpx.post(f"{BASE}/reset", json={"seed": 42, "task": "basic_audit"})
print(f"\nReset: {r.status_code}")
data = r.json()
print(f"Done: {data.get('done')}")
print(f"Reward: {data.get('reward')}")
obs = data.get('observation', {})
print(f"Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")
# Check if data is directly the observation or nested
if 'metadata' in data:
    meta = data['metadata']
elif 'observation' in data and isinstance(data['observation'], dict):
    meta = data['observation'].get('metadata', data['observation'])
else:
    meta = data
print(f"Feedback: {str(meta.get('feedback', ''))[:100]}")

# Step
action = {
    "action_type": "flag_violation",
    "item_id": "EXP-000",
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
