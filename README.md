# SentinelOps Arena

Multi-agent self-play RL environment for enterprise security training, built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv) for the [OpenEnv Hackathon SF](https://cerebralvalley.ai/e/openenv-hackathon-sf) (March 7-8, 2026).

Three AI agents compete in a simulated enterprise environment:
- **RED TEAM (Attacker)** — Launches schema drift, policy drift, social engineering, and rate limiting attacks
- **BLUE TEAM (Worker)** — Handles customer requests across CRM, Billing, and Ticketing systems
- **AUDITOR (Oversight)** — Monitors worker actions and flags policy violations

Through adversarial self-play with GRPO training, all three agents improve simultaneously.

## Quick Start

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run Gradio demo
python app.py

# Run HTTP server
python -m sentinelops_arena.server --port 8000

# Run demo script
python -m sentinelops_arena.demo
```

## Project Structure

```
NexusEnv/
├── sentinelops_arena/
│   ├── models.py              # Action, Observation, State, data models
│   ├── environment.py         # SentinelOpsArena (MCPEnvironment) — core env
│   ├── systems/
│   │   ├── crm.py             # CRM simulator
│   │   ├── billing.py         # Billing simulator
│   │   └── ticketing.py       # Ticketing simulator
│   ├── attacks.py             # 4 attack types (schema/policy drift, social eng, rate limit)
│   ├── rewards.py             # Reward functions for all 3 agents
│   ├── task_generator.py      # Customer task generation
│   ├── demo.py                # Heuristic agents + episode runner
│   ├── server.py              # HTTP/WebSocket server
│   ├── test_phase1.py         # Unit tests
│   └── test_environment.py    # Integration tests
├── app.py                     # Gradio UI (HuggingFace Spaces)
├── train.py                   # GRPO training script (Unsloth + TRL)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Architecture

**3 Agents, 3 Systems, 30 Ticks per Episode**

Each tick: Attacker acts → Worker acts → Oversight acts

### Attack Types
1. **Schema Drift** — Renames fields across all records. Worker must detect KeyError, call `get_schema()`, and adapt.
2. **Policy Drift** — Changes business rules (refund windows, approval requirements). Worker must call `get_current_policy()`.
3. **Social Engineering** — Injects fake authority messages. Worker must resist manipulation.
4. **Rate Limiting** — Throttles API calls. Worker must handle gracefully.

### MCP Tools
19 tools exposed via FastMCP, organized by agent role:
- **Worker**: lookup_customer, check_balance, issue_refund, create_ticket, get_schema, get_current_policy, etc.
- **Attacker**: launch_attack, get_attack_budget
- **Oversight**: flag_action, get_trajectory

## Training

Uses GRPO (Group Relative Policy Optimization) with Unsloth + TRL:

```bash
# Train with Unsloth (recommended, 2x faster)
python train.py --use_unsloth --model_name unsloth/Qwen2.5-0.5B-Instruct

# Train without Unsloth
python train.py --model_name Qwen/Qwen2.5-0.5B-Instruct
```

See `train.py` for the full training pipeline.

## Partner Tracks

- **Fleet AI** — Scalable Oversight: the Oversight agent monitors and explains Worker behavior
- **Patronus AI** — Schema Drift: schema and policy drift are core attack types

## Tech Stack

- **OpenEnv** 0.2.x — Environment framework
- **FastMCP** — MCP tool server
- **Gradio** — Demo UI
- **HuggingFace TRL** — GRPO training
- **Unsloth** — Fast fine-tuning (2x speed, 70% less VRAM)
- **Pydantic** — Data validation

## Tests

```bash
python sentinelops_arena/test_phase1.py
python sentinelops_arena/test_environment.py
```
