# Demo: CI Gate Failure

This fixture demonstrates a real regression that the watchdog catches.

**What changed:** The `check_eligibility` node's policy shifted from "full refund within 30 days" to "refunds only for defective products with photographic evidence." This causes `process_refund` to deny a refund that was previously approved.

## Steps to record the GIF

```bash
# 1. Pin the baseline
langgraph-replay baseline set demo-baseline-001

# 2. Run watchdog (exact mode) — should fail with exit 1
langgraph-replay watchdog watch demo-regressed-001 \
  --baseline demo-baseline-001 \
  --output demo_report.json

# 3. Optional: show upstream causes
langgraph-replay watchdog watch demo-regressed-001 \
  --baseline demo-baseline-001 \
  --upstream \
  --output demo_upstream_report.json
```

## Expected output

**Step 2** should show:
- `parse_ticket` — unchanged
- `check_eligibility` — REGRESSION (policy changed)
- `process_refund` — REGRESSION (denied instead of approved)
- `send_notification` — unchanged
- Exit code: 1

**Step 3** should additionally show:
- Upstream cause: `check_eligibility` tool output changed
- Exit code: 1
