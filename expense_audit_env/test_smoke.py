"""Quick smoke test for the environment."""
import sys
sys.path.insert(0, '.')

from server.audit_environment import AuditEnvironment
from openenv.core.env_server.types import Action

# Test reset
env = AuditEnvironment()
obs = env.reset(task='basic_audit', seed=42)
print(f"Reset OK: done={obs.done}, reward={obs.reward}")
print(f"Metadata keys: {list(obs.metadata.keys())}")
feedback = obs.metadata.get('feedback', '')
print(f"Feedback: {feedback[:120]}")

report = obs.metadata.get('current_report')
if report:
    print(f"\nCurrent report: {report['report_id']}")
    print(f"Employee: {report['employee_name']} ({report['department']})")
    print(f"Items: {len(report['expenses'])}")
    for item in report['expenses']:
        print(f"  {item['item_id']}: {item['category']} | {item['vendor']} | ${item['amount']:.2f} | receipt={item['receipt_present']}")

policy = obs.metadata.get('company_policy', [])
print(f"\nPolicy rules: {len(policy)}")

# Test a flag_violation step
print("\n--- Testing flag_violation ---")
first_item_id = report['expenses'][0]['item_id'] if report else 'EXP-000'
action_data = {
    "action_type": "flag_violation",
    "item_id": first_item_id,
    "violation_type": "over_limit",
    "reason": "Testing flag",
}

# The environment expects an Action-like object
from models import AuditAction
action = AuditAction(**action_data)
obs2 = env.step(action)
print(f"Step OK: done={obs2.done}, reward={obs2.reward}")
print(f"Feedback: {obs2.metadata.get('feedback', '')[:120]}")

# Test approve_report
print("\n--- Testing approve_report ---")
action2 = AuditAction(action_type="approve_report", reason="Report looks clean overall")
obs3 = env.step(action2)
print(f"Step OK: done={obs3.done}, reward={obs3.reward}")
print(f"Feedback: {obs3.metadata.get('feedback', '')[:120]}")

# Check state
state = env.state
print(f"\nState: episode_id={state.episode_id}, step_count={state.step_count}")

print("\n✓ All smoke tests passed!")
