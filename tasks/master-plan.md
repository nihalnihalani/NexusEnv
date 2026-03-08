# SentinelOps Arena -- Master Improvement Plan

**Created:** Sunday March 8, 2026
**Goal:** Maximize hackathon judging score with surgical code fixes

---

## Priority Legend

| Score | Meaning |
|-------|---------|
| 10 | Must fix -- breaks core functionality or judges will reject |
| 8-9 | High impact -- judges will directly notice and reward |
| 5-7 | Noticeable improvement -- strengthens the demo |
| 1-4 | Low impact -- skip unless time permits |

---

## CRITICAL FIXES (Bugs that break core functionality)

### FIX-1: Billing `issue_refund()` never checks `window_ticks` [Priority: 10]

**Bug:** `billing.py:issue_refund()` checks `max_amount` and `requires_approval` but NEVER checks `window_ticks`. Policy drift attacks that change `window_ticks` have zero effect on refund validation. This means 1/3 of policy drift parameters is dead code.

**File:** `sentinelops_arena/systems/billing.py` (lines 47-89)
**Change:** Add window_ticks validation. The invoice has `date_tick` and the environment tracks the current tick. Pass `current_tick` into `issue_refund()` and compare `current_tick - invoice["date_tick"]` against `self.refund_policy.window_ticks`.
**Impact:** Policy drift attacks now meaningfully change refund behavior. Patronus AI judges (schema/policy drift track) will directly verify this works.
**Lines of code:** ~10 lines in billing.py, ~3 lines in environment.py to pass current_tick

**Details:**
- Add `current_tick: int` parameter to `issue_refund()`
- After the existing checks, add:
  ```python
  ticks_since_invoice = current_tick - invoice.get("date_tick", 0)
  if ticks_since_invoice > self.refund_policy.window_ticks:
      return {"error": f"Refund window expired. Invoice is {ticks_since_invoice} ticks old, policy allows {self.refund_policy.window_ticks}"}
  ```
- Update environment.py `_execute_worker_action` to pass `self.tick`
- Update the MCP tool `issue_refund` to pass `self.tick`

---

### FIX-2: CRM and Ticketing have no rate limiting support [Priority: 8]

**Bug:** `attacks.py:_execute_rate_limit()` calls `system.set_rate_limit()`, but only `BillingSystem` implements it. `CRMSystem` and `TicketingSystem` have no `set_rate_limit`, `_rate_limit`, `_call_count`, or `_rate_limit_check()`. The attack manager already checks `hasattr(system, "set_rate_limit")` and returns error, but the attacker can still target CRM/ticketing and waste budget.

**File:** `sentinelops_arena/systems/crm.py`, `sentinelops_arena/systems/ticketing.py`
**Change:** Add rate limiting to CRM and Ticketing, mirroring BillingSystem's implementation.
**Impact:** Rate limit attacks now work on all 3 systems. The `_is_rate_limited()` check in environment.py (line 601-606) already handles this via `hasattr(system, "_rate_limit")`, so once the attribute exists, rate limiting shows up in the dashboard.
**Lines of code:** ~20 lines per system (copy from billing.py pattern)

**Details for each system (CRM + Ticketing):**
- Add `self._rate_limit: int = 0` and `self._call_count: int = 0` to `__init__`
- Add `_rate_limit_check()` method (copy from billing.py)
- Add `set_rate_limit()` method (copy from billing.py)
- Add `reset_rate_limit_counter()` method (copy from billing.py)
- Add `if self._rate_limit_check(): return {"error": "Rate limit exceeded."}` to `lookup_customer`, `update_tier`, `add_note`, `get_history`, and for ticketing: `create_ticket`, `assign_ticket`, `escalate`, `resolve`, `check_sla`
- Update environment.py to reset CRM and ticketing counters each tick (add to the tick-advance block at line 346)

---

### FIX-3: Schema drift renames target non-existent fields [Priority: 7]

**Bug:** `SCHEMA_DRIFT_RENAMES` in `demo.py` includes `{"old_field": "email", ...}`, `{"old_field": "address", ...}`, `{"old_field": "phone", ...}`, `{"old_field": "id", ...}`. But the Customer model has fields: `customer_id`, `name`, `tier`, `region`, `contact_email`, `lifetime_value`, `notes`. Only `name -> full_name` actually works. The others silently do nothing because the fields don't exist.

**File:** `sentinelops_arena/demo.py` (lines 125-131)
**Change:** Fix renames to use actual Customer model field names.
**Lines of code:** ~5 lines

**New renames:**
```python
SCHEMA_DRIFT_RENAMES = [
    {"old_field": "name", "new_field": "full_name"},
    {"old_field": "contact_email", "new_field": "email_address"},
    {"old_field": "region", "new_field": "territory"},
    {"old_field": "tier", "new_field": "membership_level"},
    {"old_field": "lifetime_value", "new_field": "total_spend"},
]
```

Also fix in `train.py` (lines 311-312) which has the same bad renames.

---

### FIX-4: `tasks_completed` count is always 0 [Priority: 5]

**Bug:** `environment.py` line 356-360 counts `tasks_completed` by checking `t.get("task_completed")` in trajectory entries, but no trajectory entry ever sets `task_completed = True`. The trajectory append at line 329-336 only stores `tick`, `agent`, `action_type`, `reward`.

**File:** `sentinelops_arena/environment.py` (lines 329-336, 356-360)
**Change:** Add `task_completed` flag to trajectory entries when worker successfully completes a task.
**Lines of code:** ~3 lines

**Details:**
- In the trajectory append, add `"task_completed": (action.agent == AgentRole.WORKER and self.last_worker_result and self.last_worker_result.get("success", False))`
- Or simpler: after computing worker reward, if result["success"], set a flag on the trajectory entry

---

## HIGH-IMPACT IMPROVEMENTS (Things judges will notice/reward)

### IMP-1: Apply Gradio theme in `gr.Blocks()` constructor [Priority: 9]

**Bug:** The `SentinelTheme()` and `CUSTOM_CSS` are passed to `demo.launch()` but NOT to `gr.Blocks()`. On HuggingFace Spaces, `launch()` args may be ignored. The theme must be in the constructor.

**File:** `app.py` (line 124, line 444-448)
**Change:** Move theme and css into `gr.Blocks()`:
```python
with gr.Blocks(title="SentinelOps Arena", fill_width=True, theme=SentinelTheme(), css=CUSTOM_CSS) as demo:
```
Remove duplicate theme/css from `demo.launch()`.
**Lines of code:** 2 lines

---

### IMP-2: Worker heuristic should complete multi-step tasks [Priority: 8]

**Bug:** The trained `HeuristicWorker._trained_act()` in `demo.py` checks policy for refund tasks but NEVER actually issues the refund. It just calls `get_current_policy` every time it sees a refund task. Same for untrained worker -- it issues refund but never checks CRM first.

**File:** `sentinelops_arena/demo.py` (lines 253-299)
**Change:** Add state tracking to HeuristicWorker so it can complete multi-step flows:
1. First encounter of refund task: call `get_current_policy`
2. Second encounter (same task): call `issue_refund` with validated params

This will make the trained vs untrained comparison dramatically more interesting in the demo.

**Lines of code:** ~25 lines

**Details:**
- Add `self._last_task_id` and `self._policy_checked` state to HeuristicWorker
- Trained flow: refund task first seen -> get_current_policy, refund task second time -> issue_refund with compliant params
- This creates visible "adaptive behavior" in the replay -- exactly what judges want to see

---

### IMP-3: Improve explanation quality metric [Priority: 6]

**Bug:** `environment.py` line 441: `explanation_quality = min(len(explanation) / 100.0, 1.0)` -- quality is just string length. A 100+ character explanation always gets max quality regardless of content.

**File:** `sentinelops_arena/environment.py` (line 441)
**Change:** Add keyword detection alongside length. Check if explanation mentions relevant terms (policy, schema, drift, social engineering, violation, refund, etc.).
**Lines of code:** ~8 lines

```python
keywords = ["policy", "schema", "drift", "violation", "social", "engineering",
            "refund", "unauthorized", "error", "compliance"]
keyword_matches = sum(1 for k in keywords if k in explanation.lower())
length_score = min(len(explanation) / 100.0, 0.5)
keyword_score = min(keyword_matches / 3.0, 0.5)
explanation_quality = length_score + keyword_score
```

---

## QUICK WINS (Small effort, visible improvement)

### QW-1: Fix HF Spaces requirements [Priority: 9]

**File:** `requirements.txt`
**Change:** Ensure `pandas>=2.0` is listed. Verify gradio version consistency.
**Lines of code:** 1-2 lines

---

### QW-2: Fix version claims in SENTINELOPS_ARENA.md [Priority: 4]

**Bug:** Spec says "80 ticks" and "OpenEnv 0.4" but code uses 30 ticks and OpenEnv 0.2.x.
**Action:** SKIP -- spec docs are aspirational. Judges who read code will see it works. Not worth the time.

---

### QW-3: Clean up hackathon_env/ vestigial directory [Priority: 3]

**File:** `.gitignore` or delete `hackathon_env/`
**Action:** SKIP unless doing final cleanup -- judges won't look here.

---

## SKIP LIST (Not worth the time)

| Item | Why Skip |
|------|----------|
| Compound attacks | 2+ hours, spec feature not in code |
| Compliance drift | New attack type, 1+ hour to implement and test |
| A2A protocol | Already marked "Cut" in spec, correct decision |
| Docker support | HF Spaces uses Gradio SDK |
| SLA breach detection | Needs rework of ticketing + reward pipeline |
| MCP-X gateway | MCP tools work inline, gateway is polish |
| Full GRPO convergence | Training pipeline exists, convergence not needed |

---

## IMPLEMENTATION ORDER

Execute in this exact order to maximize impact per minute:

| # | Item | Est. Time | Impact |
|---|------|-----------|--------|
| 1 | **IMP-1**: Theme in gr.Blocks constructor | 2 min | HF Spaces theme works |
| 2 | **QW-1**: Fix requirements.txt | 2 min | HF Spaces doesn't crash |
| 3 | **FIX-1**: window_ticks enforcement in billing | 10 min | Policy drift attacks work (Patronus AI track) |
| 4 | **FIX-3**: Fix schema drift renames | 5 min | Schema drift attacks work (Patronus AI track) |
| 5 | **FIX-2**: Rate limiting for CRM + Ticketing | 15 min | All attacks work on all systems |
| 6 | **FIX-4**: tasks_completed tracking | 3 min | Dashboard shows correct count |
| 7 | **IMP-2**: Worker multi-step task completion | 15 min | Demo shows real adaptive behavior |
| 8 | **IMP-3**: Better explanation quality metric | 5 min | Oversight agent more realistic |

**Total estimated time: ~57 minutes**

---

## JUDGE-SPECIFIC IMPACT ANALYSIS

### Patronus AI (Darshan Deshpande) -- Schema Drift Track ($10K)
- **FIX-1** makes policy drift mechanically functional (window_ticks enforced)
- **FIX-3** makes schema drift renames target real fields (attacks actually break things)
- **IMP-2** shows worker adapting to drift in multi-step tasks
- Combined: these 3 fixes transform "drift is mentioned" into "drift demonstrably works"

### Fleet AI (Nicolai Ouporov) -- Scalable Oversight Track ($10K)
- **IMP-3** gives oversight meaningful explanation quality scoring (not just string length)
- **IMP-2** creates real violations for oversight to catch (multi-step tasks that can fail)
- **FIX-4** shows accurate task completion stats in dashboard

### Daniel Han (Unsloth) -- Training Pipeline
- The training pipeline in `train.py` is already solid
- Fixes to the environment make the reward signals more meaningful
- GRPO reward functions already correctly shaped

### Sanyam Bhutani (Meta) -- OpenEnv Quality
- **FIX-1 + FIX-2** demonstrate environment integrity (attacks have real effects)
- Clean MCP tool exposure with 19 tools already impressive
- Environment reset/step/state cycle works correctly

### Benjamin Burtenshaw (HuggingFace) -- Hub Deployment
- **IMP-1** + **QW-1** ensure HF Spaces deployment works correctly with theme
- Gradio 6 native plots and custom theme are impressive

---

## WHAT NOT TO TOUCH

1. **rewards.py** -- Reward functions are clean and match spec tables. Do not modify.
2. **models.py** -- Pydantic models are correct. Do not add fields unless required by a fix.
3. **task_generator.py** -- Works fine, generates correct task mix.
4. **sentinel_theme.py** -- Theme is polished. Do not tweak CSS.
5. **replay_html.py** -- HTML rendering works. Do not modify.
6. **chart_helpers.py** -- Chart data builders work. Do not modify.
7. **metrics.py** -- Security metrics computation is solid.
8. **train.py** -- Only touch to fix schema_drift renames in the heuristic attacker configs.
