# SentinelOps Arena -- Build Plan

## Overview

14-hour hackathon build plan for a multi-agent self-play RL environment on OpenEnv 0.2.1. Solo developer. Deadline: Sunday March 8, 2026 at 1:00 PM.

**KEY INSIGHT:** Innovation (40%) + Storytelling (30%) = 70% of judging is NON-code. Allocate time accordingly.

## Revised Phase Summary

| Phase | File | Time | Cumulative | What |
|-------|------|------|------------|------|
| 0 | (inline) | 0.5h | 0-0.5h | Test H100/Northflank, write 60s video script |
| 1 | [phase-1-models-and-systems.md](phase-1-models-and-systems.md) | 3.5h | 0.5-4h | Pydantic models + enterprise system simulators |
| 2 | [phase-2-environment-core.md](phase-2-environment-core.md) | 2h | 4-6h | SentinelOpsArena(MCPEnvironment), rewards, turn management |
| 3 | [phase-3-mcp-and-server.md](phase-3-mcp-and-server.md) | 0.5h | 6-6.5h | MCP tools via MCPEnvironment + HTTP server |
| 4 | [phase-4-demo-and-ui.md](phase-4-demo-and-ui.md) | 2h | 6.5-8.5h | Demo script, Gradio app (1 tab), HF Spaces deploy |
| 5 | [phase-5-training.md](phase-5-training.md) | 2h | 8.5-10.5h | Colab notebook, GRPO pipeline (fall back to SFT at 1.5h) |
| 6 | [phase-6-polish-and-submit.md](phase-6-polish-and-submit.md) | 3.5h | 10.5-14h | Polish, video recording, submission |

**Total: 14 hours**

## Phase 0: Pre-Flight (Hour 0-0.5)

Before writing any code:
1. **Test H100 via Northflank** -- verify access, note available VRAM. If no H100, lock to Qwen2.5-1.5B.
2. **Write 60-second video script** -- forces clarity on what to demo. Script drives the build.
3. **Set up repo structure** -- create directories, pyproject.toml

## Dependencies

```
Phase 0 (Pre-Flight)
    |
    v
Phase 1 (Models & Systems)
    |
    v
Phase 2 (Environment Core)  -- CHECKPOINT 1 (Hour 6): Minimum Viable
    |
    v
Phase 3 (MCP + Server)      -- MCPEnvironment handles this almost free
    |
    v
Phase 4 (Demo & UI)         -- CHECKPOINT 2 (Hour 8.5): Deploy to HF Spaces
    |
    v
Phase 5 (Training)          -- CHECKPOINT 3 (Hour 10.5): Strong Submission
    |
    v
Phase 6 (Polish & Submit)   -- CHECKPOINT 4 (Hour 14): Full Submission
```

## Stop-and-Submit Checkpoints

**Hour 6 (after Phase 2):** Environment works with random agents. Submit with basic demo + placeholder training notebook. Minimum viable.

**Hour 8.5 (after Phase 4):** Environment + MCP tools + Gradio demo deployed on HF Spaces. Good submission. **INSURANCE SUBMISSION** -- deploy to HF Spaces here.

**Hour 10.5 (after Phase 5):** Everything above + working Colab training pipeline with visible reward improvement. Strong submission.

**Hour 14 (after Phase 6):** Polished demo, training curves, video, stretch goals. Full submission.

## Scoring Priorities

| Criterion | Weight | Primary Phase | Time Allocated |
|-----------|--------|---------------|----------------|
| Innovation | 40% | Phases 1-2 (3-agent self-play architecture) | 5.5h |
| Storytelling | 30% | Phase 4 + 6 (Gradio demo + video) | 5.5h |
| Training Script | 20% | Phase 5 (Colab GRPO notebook) | 2h |
| Pipeline | 10% | Phase 3 (MCP integration) | 0.5h |

## Key Technical Decisions

- **OpenEnv version:** 0.2.1 (stable, `openenv-core[core]>=0.2.0`)
- **Base class:** `MCPEnvironment` (NOT raw `Environment`) -- auto-routes `ListToolsAction`/`CallToolAction` to FastMCP server. Gives MCP tool discovery for free.
- **MCP-X gateway:** CUT -- MCPEnvironment already handles MCP tool exposure. Per-agent isolation is nice-to-have, not needed.
- **Action pattern:** `Action(extra='forbid')` -- all agent-specific fields must be Optional with defaults, or use separate action classes per role
- **Server:** `create_app()` from `openenv.core.env_server.http_server`
- **Training:** Unsloth for model loading only, vanilla TRL `GRPOTrainer` with `rollout_func`. Fall back to SFT if GRPO fails at 1.5h.
- **Model:** Qwen2.5-1.5B for Colab (5GB VRAM), Qwen2.5-7B if H100 available
- **Demo:** Gradio on HuggingFace Spaces
- **Episode scope:** 30 ticks, 15 customers, 15 invoices, 10 tickets, 30 tasks
- **Attack types:** 4 (schema drift, policy drift, social engineering, rate limiting)
- **Reserved tool names:** `reset`, `step`, `state`, `close` CANNOT be used as MCP tool names

## File Structure

```
sentinelops_arena/
  __init__.py
  models.py              # Pydantic models (enums, data, action/observation/state)
  systems/
    __init__.py
    crm.py               # CRM simulator
    billing.py           # Billing simulator
    ticketing.py         # Ticketing simulator
  attacks.py             # Attack mechanics (4 types)
  rewards.py             # Reward functions (3 agents)
  task_generator.py      # Task generation
  environment.py         # SentinelOpsArena(MCPEnvironment) -- MCP tools defined here
  server.py              # create_app() HTTP server

training/
  colab_training.ipynb   # Colab GRPO notebook (REQUIRED)
  env_standalone.py      # Standalone env for Colab (no openenv dependency)

app.py                   # HF Spaces Gradio entry point
pyproject.toml
README.md
```

**NOTE:** No separate `mcp_tools.py` -- MCP tools are defined inside `environment.py` using FastMCP, and `MCPEnvironment` auto-routes them.

**NOTE:** No `mcp-x/` directory -- MCP-X gateway is CUT from the plan.

## Partner Track Alignment

- **Fleet AI** (Scalable Oversight): The Oversight agent monitors, analyzes, and explains behavior of Worker agent
- **Patronus AI** (Schema Drift): Schema drift and policy drift are core attack types in the environment
