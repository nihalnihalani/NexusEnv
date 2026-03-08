# SentinelOps Arena -- Master Improvement Plan

**Generated:** Sunday March 8, 2026
**Deadline:** Sunday March 8, 2026 1:00 PM
**Synthesized from:** Researcher findings, code reviewer findings, sponsor track analysis, devil's advocate critique, gap analysis

---

## CONTEXT: Current State

The core environment is solid: 3 agents, 3 enterprise systems, 4 attack types, reward functions, randomized attacker, security metrics engine, and a polished Gradio UI with 4 tabs and a cybersecurity theme. The codebase compiles and the trained vs untrained worker comparison shows meaningful score differences.

**Three REQUIRED submission deliverables are NOT done:**
1. HuggingFace Spaces deployment
2. Google Colab training notebook
3. Demo video on YouTube

**Partner tracks targeted:** Fleet AI ($10K, Scalable Oversight) and Patronus AI ($10K, Schema Drift)

---

## 1. CRITICAL FIXES (Must Do -- Submission Fails Without These)

### C1. Deploy to HuggingFace Spaces
- **What:** Create HF Space, push code, verify it builds and runs
- **Files:** `requirements.txt`, `README.md` (frontmatter), `app.py`
- **Effort:** 30 min
- **Impact:** BLOCKER -- no live URL = no submission
- **Details:**
  - Add `pandas>=2.0` to `requirements.txt` (missing, app.py imports it)
  - Verify `gradio>=6.0.0` in requirements.txt matches README frontmatter `sdk_version: 6.9.0`
  - Create Space at `huggingface.co/new-space`, SDK: Gradio, Hardware: CPU Basic
  - Push with `git push hf main` or use `huggingface_hub.upload_folder()`
  - Test all 4 tabs work on the live URL

### C2. Create Colab Training Notebook
- **What:** Create `training/colab_training.ipynb` with working GRPO pipeline
- **Files:** New file: `training/colab_training.ipynb`
- **Effort:** 60-90 min
- **Impact:** BLOCKER -- submission requires "Minimal Training Script"
- **Details:**
  - Reuse logic from `train.py` (it has everything needed)
  - Use `Qwen/Qwen2.5-0.5B-Instruct` (fits free Colab T4)
  - Use Unsloth for model loading, vanilla TRL GRPOTrainer for training
  - Must show: env verification, data collection, model loading, GRPO config, at least a few training steps
  - If openenv-core fails on Colab Python version, bundle standalone env code
  - Add markdown cells explaining each step, mention partner tracks

### C3. Record Demo Video
- **What:** 1-3 minute screen recording of Gradio app + voice/text narration
- **Files:** N/A (external -- YouTube upload)
- **Effort:** 30 min
- **Impact:** BLOCKER -- submission requires YouTube demo video
- **Details:**
  - Show: episode replay (attack/adapt/flag cycle), untrained vs trained comparison, environment inspector
  - Mention: 3-agent self-play, Fleet AI oversight, Patronus AI schema drift
  - Keep simple -- QuickTime screen record, no fancy editing

### C4. Verify Gradio App Launches Locally
- **What:** Run `python app.py` and test all 4 tabs
- **Files:** `app.py`, all imported modules
- **Effort:** 15 min
- **Impact:** HIGH -- if app crashes, HF Spaces will fail too
- **Note:** `tasks/todo.md` shows this is UNCHECKED

---

## 2. HIGH-IMPACT IMPROVEMENTS (Should Do -- Directly Impress Judges)

### H1. Improve Oversight Explanation Quality Scoring (Fleet AI Track)
- **What:** Replace character-count explanation quality with structured quality scoring
- **Files:** `sentinelops_arena/environment.py:441`, `sentinelops_arena/demo.py:302-327`
- **Effort:** 20 min
- **Impact:** HIGH for Fleet AI ($10K) -- current scoring is `min(len(explanation) / 100.0, 1.0)` which is embarrassingly simplistic. Fleet AI judge Nicolai Ouporov will notice.
- **Details:**
  - In `environment.py:441`, replace character-length heuristic with keyword-based quality scoring:
    - +0.25 if explanation mentions the violation type (e.g., "policy violation", "social engineering")
    - +0.25 if explanation references specific data (e.g., amount, field name, policy rule)
    - +0.25 if explanation states the rule being violated (e.g., "max refund is $2000")
    - +0.25 if explanation recommends corrective action
  - In `demo.py` HeuristicOversight, improve the canned explanation strings to include specific data from the observation (e.g., "Worker issued refund exceeding policy max of $X. Current policy requires approval for amounts over $Y.")

### H2. Add SLA Policy Drift to Ticketing (Patronus AI Track)
- **What:** Allow the attacker to change SLA deadlines, not just refund policies
- **Files:** `sentinelops_arena/systems/ticketing.py`, `sentinelops_arena/attacks.py`, `sentinelops_arena/demo.py`
- **Effort:** 20 min
- **Impact:** HIGH for Patronus AI ($10K) -- doubles the policy drift surface. Currently only billing has policy drift.
- **Details:**
  - Add `TicketingSystem.apply_policy_drift(changes)` in `ticketing.py` that modifies `self.sla_rules`
  - In `attacks.py:_execute_policy_drift()`, route to ticketing system when target is TICKETING
  - In `demo.py` RandomizedAttacker, add SLA policy drift options to `POLICY_DRIFT_CHANGES`
  - Worker should call `get_current_policy("sla")` to discover changed SLA rules

### H3. Add Oversight Metrics to Dashboard
- **What:** Add oversight-specific metrics (explanation quality, detection accuracy) to the metrics engine and Gradio UI
- **Files:** `sentinelops_arena/metrics.py`, `app.py`
- **Effort:** 25 min
- **Impact:** HIGH for Fleet AI ($10K) -- currently NO oversight-specific metrics exist in the dashboard
- **Details:**
  - In `metrics.py`, add to `compute_episode_metrics()`:
    - `oversight_accuracy`: correct flags + correct approvals / total oversight decisions
    - `avg_explanation_quality`: average explanation quality score across all oversight decisions
  - Add a new metric card for oversight accuracy in `format_metrics_html()`
  - This makes the Fleet AI story visible in the demo

### H4. Add Drift-Specific Metrics
- **What:** Add drift adaptation metrics to the metrics engine
- **Files:** `sentinelops_arena/metrics.py`
- **Effort:** 15 min
- **Impact:** HIGH for Patronus AI ($10K) -- makes drift adaptation visible and measurable
- **Details:**
  - Add to `compute_episode_metrics()`:
    - `drift_events`: total schema + policy drift attacks
    - `drifts_detected`: number of times worker called get_schema/get_current_policy after a drift
    - `avg_drift_recovery_ticks`: average ticks between drift and worker's first defensive action
  - Add metric card for "Drift Adaptation" in `format_metrics_html()`

### H5. Improve HeuristicOversight Explanations
- **What:** Make the oversight agent's explanations reference specific data from the observation
- **Files:** `sentinelops_arena/demo.py:302-327`
- **Effort:** 15 min
- **Impact:** MEDIUM-HIGH for Fleet AI -- judges will see these in the replay log
- **Details:**
  - Pass `obs` to `HeuristicOversight.act()` (currently only uses `obs.last_action_result`)
  - Generate explanations like: "Worker action at tick {tick}: {action_type} resulted in error. The error '{error_msg}' suggests schema drift may have occurred. Recommended: call get_schema() to discover new field names."
  - For social engineering: "Worker followed suspicious instructions containing override language. The message '{first 50 chars}' appears to be a social engineering attack. Flagging as critical violation."
  - For policy violations: "Refund of ${amount} exceeds current policy maximum of ${max}. Policy was last updated at tick {last_policy_change}."

---

## 3. QUICK WINS (Do If Time Allows -- Small Effort, Good Impression)

### Q1. Fix Documentation Inconsistencies
- **What:** Fix mismatches between spec doc, README, and actual code
- **Files:** `README.md`, `pyproject.toml`
- **Effort:** 10 min
- **Impact:** Prevents judges from noticing sloppy details
- **Details:**
  - Set `gradio>=6.0.0` consistently in pyproject.toml (currently says >=5.0.0)
  - Fix README project structure to match reality (remove `mcp_tools.py` listing)
  - Do NOT touch SENTINELOPS_ARENA.md (it's a spec doc, acceptable to be aspirational)

### Q2. Add Links to About Tab
- **What:** Once Colab notebook and video exist, add links to the About tab
- **Files:** `app.py` (About tab section)
- **Effort:** 5 min
- **Impact:** Makes it easy for judges to find all submission artifacts

### Q3. Clean Up Vestigial Files
- **What:** Remove or gitignore `hackathon_env/` directory
- **Files:** `.gitignore`, possibly `hackathon_env/`
- **Effort:** 5 min
- **Impact:** Prevents judge confusion

### Q4. Add Billing Schema Drift Support
- **What:** Allow schema drift attacks against billing system too
- **Files:** `sentinelops_arena/systems/billing.py`
- **Effort:** 10 min
- **Impact:** Strengthens Patronus AI story -- all 3 systems support schema drift
- **Details:**
  - Add `BillingSystem.apply_schema_drift(old_field, new_field)` mirroring CRM pattern
  - Add `_field_map` dict and `_apply_field_map` method to BillingSystem
  - Update `attacks.py` `VALID_TARGETS` for schema_drift to include BILLING

---

## 4. SKIP LIST (Not Worth the Time)

| Item | Reason |
|------|--------|
| Compound attacks (2-3 simultaneous) | 2+ hours, marginal judge impact |
| Compliance drift (new required fields) | 1+ hours, nice but not critical |
| A2A protocol | Already marked "Cut" in spec, not in submission requirements |
| Docker support | HF Spaces uses Gradio SDK directly |
| MCP-X gateway demo | MCP tools in environment.py are sufficient |
| Full GRPO convergence | Pipeline working is enough -- convergence not required |
| Real datetime-based SLA | Tick-based is fine for demo |
| Multi-GPU training | Overkill for hackathon |
| Refactoring codebase | No judge impact, waste of time |

---

## EXECUTION ORDER (Recommended)

**Phase 1 (0:00 - 0:15): Verify and fix basics**
1. C4: Verify Gradio app launches locally
2. Q1: Fix requirements.txt (add pandas) and pyproject.toml consistency

**Phase 2 (0:15 - 1:00): High-impact code improvements**
3. H1: Improve oversight explanation quality scoring (20 min)
4. H2: Add SLA policy drift to ticketing (20 min)
5. H5: Improve HeuristicOversight explanations (15 min)

**Phase 3 (1:00 - 1:30): Metrics improvements**
6. H3: Add oversight metrics to dashboard (25 min)
7. H4: Add drift-specific metrics (15 min)

**Phase 4 (1:30 - 2:00): Deployment**
8. C1: Deploy to HuggingFace Spaces (30 min)

**Phase 5 (2:00 - 3:15): Required deliverables**
9. C2: Create Colab training notebook (75 min)

**Phase 6 (3:15 - 3:45): Video and submission**
10. C3: Record demo video (30 min)

**Phase 7 (3:45 - 4:00): Final polish**
11. Q2: Add links to About tab (5 min)
12. Q3: Clean up vestigial files (5 min)
13. Final push and submit (5 min)

---

## KEY JUDGE CONSIDERATIONS

- **Nicolai Ouporov (Fleet AI):** Cares about scalable oversight. Will check: Does the oversight agent actually explain violations well? Is explanation quality tracked? Does training improve oversight?
- **Darshan Deshpande (Patronus AI):** Cares about schema drift. Will check: How many drift types? Does the worker adapt? Is drift visible in the UI?
- **Daniel Han (Unsloth):** Cares about Unsloth/TRL integration. Will check: Does the Colab notebook use Unsloth correctly? Does training actually work?
- **Sanyam Bhutani (Meta):** Cares about OpenEnv quality. Will check: Is the environment well-structured? Does step/reset/state work properly?
- **Benjamin Burtenshaw (HuggingFace):** Cares about Hub deployment. Will check: Is the HF Space functional and polished?
