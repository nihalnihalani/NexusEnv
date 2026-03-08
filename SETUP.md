# SentinelOps Arena — Complete Setup Guide

## 1. Local Dev Environment

### Python Version
- **Required:** Python 3.14 (system) or 3.12+ (venv)
- **Current venv:** Python 3.14.2 in `hackathon_env/.venv/` (created by uv)
- **Root venv:** Python 3.12.12 in `.venv/` (created by uv)
- **OpenEnv 0.2.1** requires `>=3.10`, works fine on 3.14
- **Tool manager:** `uv` 0.9.26 (installed at `/Users/nihalnihalani/.local/bin/uv`)

### Existing Environment State
The `hackathon_env/` directory already has a working OpenEnv echo environment with:
- `openenv-core==0.2.1` installed in `hackathon_env/.venv/`
- Working `Environment` subclass pattern (see `server/hackathon_env_environment.py`)
- Working `create_app()` HTTP server (see `server/app.py`)
- Working `EnvClient` subclass with `_step_payload()` and `_parse_result()` (see `client.py`)
- Working Dockerfile for HF Spaces deployment
- `openenv.yaml` spec file

### CRITICAL: The venv has a broken interpreter path
The `hackathon_env/.venv/bin/openenv` script points to `/Users/nihalnihalani/Desktop/Github/openev/hackathon_env/.venv/bin/python` (note `openev` not `NexusEnv`). This means the venv was created in a different directory and moved. The Python binary itself works fine, but CLI entry points are broken.

**Fix:** Recreate the venv from `hackathon_env/`:
```bash
cd /Users/nihalnihalani/Desktop/Github/NexusEnv/hackathon_env
uv venv .venv --python 3.14
uv sync
```

### Dependencies — pyproject.toml for SentinelOps

The project needs a **root-level** `pyproject.toml` for the SentinelOps Arena package. The `hackathon_env/pyproject.toml` only covers the echo env template.

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sentinelops-arena"
version = "0.1.0"
description = "Multi-agent self-play training environment for enterprise AI security"
requires-python = ">=3.10"
dependencies = [
    # Core OpenEnv runtime
    "openenv-core[core]>=0.2.1",
    # MCP tool server
    "mcp>=1.26.0",
    "fastmcp>=2.14.5",
    # HTTP server
    "fastapi>=0.115.0",
    "uvicorn>=0.24.0",
    # MCP-X gateway dependencies
    "PyJWT>=2.0",
    "toml>=0.10.2",
    "httpx>=0.27",
    # Gradio for HF Spaces demo UI
    "gradio>=5.0.0",
    # Data handling
    "pydantic>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=4.0.0",
]
training = [
    # These are for local training only, NOT for HF Spaces
    "trl>=0.15.0",
    "transformers>=4.40.0",
    "torch>=2.0.0",
    "accelerate>=0.30.0",
    "datasets>=2.18.0",
    "peft>=0.10.0",
]

[project.scripts]
server = "sentinelops_arena.server:main"

[tool.setuptools]
include-package-data = true
packages = ["sentinelops_arena", "sentinelops_arena.systems"]
```

### Pinned Dependency Versions (from envbeats reference)
| Package | Min Version | Source |
|---------|-------------|--------|
| openenv-core | 0.2.1 | hackathon_env/pyproject.toml |
| mcp | 1.26.0 | eb_assessor/pyproject.toml |
| fastmcp | 2.14.5 | mcp-x/pyproject.toml |
| fastapi | 0.128.6+ | mcp-x/pyproject.toml |
| PyJWT | 2.0+ | mcp-x/pyproject.toml |
| toml | 0.10.2+ | mcp-x/pyproject.toml |
| httpx | 0.27+ | mcp-x/pyproject.toml |
| uvicorn | 0.24.0+ | hackathon_env/server/requirements.txt |
| pydantic | 2.0+ | transitive via openenv-core |
| gradio | 5.0+ | for HF Spaces demo UI |

---

## 2. Infrastructure Setup

### Northflank H100
- Each team gets H100 GPU access via Northflank
- Used for **training only** (not deployment)
- Request at hackathon check-in or via organizer Slack
- Configure: SSH access, install Python 3.10+, CUDA drivers
- **Not required for MVP** — can use Colab free tier for training demo

### HuggingFace
- **Account:** Already have (nihalnihalani)
- **Join openenv-community:** Required for $30 compute credits — join org at huggingface.co
- **Create Space:** `nihalnihalani/sentinelops-arena`
  - SDK: Docker (custom Dockerfile) or Gradio
  - Hardware: CPU Basic (free) or CPU Upgrade ($0.03/hr from credits)
- **Push command:** `openenv push --space nihalnihalani/sentinelops-arena` OR manual git push to HF repo

### Google Colab
- Training notebook: `training/colab_training.ipynb`
- Runtime: T4 GPU (free tier) or A100 if credits available
- Key concern: Colab runs Python 3.10-3.11, but openenv-core requires >=3.10 (should work)
- **Fallback:** Bundle standalone env code in notebook without openenv import (for Python compat)

### YouTube
- Account for demo video upload
- Video length: **1 minute** (per spec, NOT 3-5 minutes as in the build plan)
- Screen record: Gradio demo + training signal
- Upload as unlisted, share link in submission

---

## 3. Repository Structure

### Target File Tree
```
NexusEnv/
├── .git/
├── .gitignore
├── .venv/                          # Root venv (Python 3.12)
├── CLAUDE.md                       # Claude Code rules
├── README.md                       # Project README (update for submission)
├── SENTINELOPS_ARENA.md            # Full spec document
├── SETUP.md                        # This file
├── pyproject.toml                  # Root project config (NEW)
├── app.py                          # HF Spaces entry point — Gradio app (NEW)
├── sentinelops_arena/              # Core package (NEW)
│   ├── __init__.py
│   ├── models.py                   # Pydantic models: Action, Observation, State, data models
│   ├── systems/
│   │   ├── __init__.py
│   │   ├── crm.py                  # CRM simulator
│   │   ├── billing.py              # Billing simulator
│   │   └── ticketing.py            # Ticketing simulator
│   ├── attacks.py                  # Attack mechanics (4 types)
│   ├── rewards.py                  # Reward functions (3 agents)
│   ├── task_generator.py           # Customer task generation
│   ├── environment.py              # SentinelOpsArena(Environment)
│   ├── mcp_tools.py                # FastMCP tool definitions
│   ├── server.py                   # create_app() HTTP server
│   └── demo.py                     # Demo script with heuristic agents
├── mcp_x/                          # MCP-X gateway (adapted from envbeats) (NEW)
│   ├── mcp_x.py                    # Gateway server (copy+adapt)
│   └── config.toml                 # Per-agent tool ACLs
├── training/                       # Training deliverables (NEW)
│   ├── colab_training.ipynb        # REQUIRED Colab notebook
│   └── rollout.py                  # rollout_func for GRPOTrainer
├── envbeats/                       # Reference implementation (existing, read-only)
│   ├── eb_assessor/
│   ├── eb_assessee_gym/
│   └── mcp-x/
├── hackathon_env/                  # Original echo env template (existing, reference)
│   ├── ...
│   └── server/
│       ├── Dockerfile              # Reference Dockerfile
│       └── app.py                  # Reference create_app() usage
└── train.py                        # Existing training script (update or replace)
```

### Key Files to Create (in build order)
1. `pyproject.toml` — root project config
2. `sentinelops_arena/__init__.py`
3. `sentinelops_arena/models.py` — all Pydantic models
4. `sentinelops_arena/systems/__init__.py`
5. `sentinelops_arena/systems/crm.py`
6. `sentinelops_arena/systems/billing.py`
7. `sentinelops_arena/systems/ticketing.py`
8. `sentinelops_arena/attacks.py`
9. `sentinelops_arena/rewards.py`
10. `sentinelops_arena/task_generator.py`
11. `sentinelops_arena/environment.py`
12. `sentinelops_arena/mcp_tools.py`
13. `sentinelops_arena/server.py`
14. `sentinelops_arena/demo.py`
15. `app.py` — Gradio HF Spaces entry point
16. `mcp_x/mcp_x.py` + `mcp_x/config.toml`
17. `training/colab_training.ipynb`

---

## 4. Deployment Config

### HuggingFace Spaces — Two Options

#### Option A: Gradio SDK (Simpler, Recommended)
HF Spaces README.md header:
```yaml
---
title: SentinelOps Arena
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
license: mit
---
```

No Dockerfile needed. HF auto-installs from `requirements.txt`:

**requirements.txt** (for HF Spaces):
```
openenv-core[core]>=0.2.1
mcp>=1.26.0
fastmcp>=2.14.5
fastapi>=0.115.0
uvicorn>=0.24.0
PyJWT>=2.0
toml>=0.10.2
httpx>=0.27
gradio>=5.0.0
pydantic>=2.0
```

#### Option B: Docker (If Gradio SDK fails)
Use adapted Dockerfile from `hackathon_env/server/Dockerfile`.

HF Spaces README.md header:
```yaml
---
title: SentinelOps Arena
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit
---
```

**Dockerfile:**
```dockerfile
FROM python:3.14-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Gradio uses port 7860 on HF Spaces
EXPOSE 7860
CMD ["python", "app.py"]
```

### Deployment Commands
```bash
# Option 1: Using openenv CLI
cd sentinelops_arena
openenv push --space nihalnihalani/sentinelops-arena

# Option 2: Manual HF push
# Create space on huggingface.co first, then:
git remote add hf https://huggingface.co/spaces/nihalnihalani/sentinelops-arena
git push hf main

# Option 3: Using huggingface_hub Python API
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path=".", repo_id="nihalnihalani/sentinelops-arena", repo_type="space")
```

---

## 5. Submission Checklist

Every field required in the submission form:

| Field | Value | Status |
|-------|-------|--------|
| **Team Name** | TBD (e.g., "NexusEnv" or "SentinelOps") | Need to decide |
| **Project Description** | Multi-agent self-play RL environment where 3 AI agents (Attacker, Worker, Oversight) interact with simulated enterprise systems. Through adversarial dynamics, agents learn to attack, defend, and audit enterprise operations. | Draft ready |
| **HuggingFace Spaces Link** | `https://huggingface.co/spaces/nihalnihalani/sentinelops-arena` | Need to create |
| **Demo Video (YouTube)** | 1-minute screencast of Gradio demo + training | Need to record |
| **Minimal Training Script** | Colab notebook link (`training/colab_training.ipynb`) | Need to build |
| **Partner Tracks** | Fleet AI (Scalable Oversight), Patronus AI (Schema Drift) | Selected |

### Submission Deadline
**Sunday, March 8th, 2026 at 1:00 PM**

---

## 6. Pre-flight Checks

### Before Writing Any Code
- [x] Python 3.14 available (system)
- [x] `uv` installed and working
- [x] OpenEnv 0.2.1 installed in `hackathon_env/.venv/`
- [x] OpenEnv Environment/Action/Observation/State APIs understood
- [x] EnvBeats patterns analyzed (create_app, MCP-X, client patterns)
- [x] Git repo initialized, on `main` branch
- [ ] Create `nihal` branch (per CLAUDE.md push rules)
- [ ] Create root `pyproject.toml`
- [ ] Set up new venv with all dependencies: `uv venv .venv && uv sync`
- [ ] Verify imports: `python -c "from openenv.core.env_server.interfaces import Environment; print('OK')"`
- [ ] Create HF Space (can be empty placeholder)
- [ ] HuggingFace: Join openenv-community org for $30 credits

### Critical API Patterns (from hackathon_env reference)

**Environment class:**
```python
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import Action, Observation, State

class MyEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def reset(self, seed=None, episode_id=None, **kwargs) -> MyObservation:
        ...

    def step(self, action: MyAction, timeout_s=None, **kwargs) -> MyObservation:
        ...

    @property
    def state(self) -> State:  # NOTE: property, not method
        ...
```

**Action class:**
```python
class MyAction(Action):
    # extra='forbid' inherited from Action base
    field: str = Field(..., description="...")
```

**Observation class:**
```python
class MyObservation(Observation):
    # Inherits: done (bool), reward (float|None), metadata (dict)
    my_field: str = Field(default="", description="...")
```

**HTTP Server:**
```python
from openenv.core.env_server.http_server import create_app
app = create_app(MyEnv, MyAction, MyObservation, env_name="my_env")
# Run: uvicorn module:app --host 0.0.0.0 --port 8000
```

**Client:**
```python
from openenv.core import EnvClient
class MyClient(EnvClient[MyAction, MyObservation]):
    def _step_payload(self, action: MyAction) -> Dict:
        return action.model_dump()
    def _parse_result(self, payload: Dict) -> StepResult[MyObservation]:
        ...
```

### Known Gotchas
1. `Action` has `extra='forbid'` — SentinelAction must not have extra fields
2. `state` is a `@property` not a method — use `env.state` not `env.state()`
3. `create_app()` returns an ASGI app — use `uvicorn.run(app)` not `app.run()`
4. Observation `reward` field type is `bool | int | float | None` (allows bool)
5. The hackathon_env venv has broken CLI entry points (moved from different path)
6. CLAUDE.md says push to `nihal` branch, not `main`
7. Demo video must be **1 minute**, not 3-5 minutes (spec says 1 minute)

### OpenEnv Version Note
The spec says "OpenEnv 0.4" but OpenEnv 0.4 does NOT exist. The stable version is **0.2.1**. The SENTINELOPS_ARENA.md references "0.4" but the actual codebase and all dependencies use 0.2.1. Build against 0.2.1.

---

## 7. Quick Start Commands

```bash
# 1. Create nihal branch
cd /Users/nihalnihalani/Desktop/Github/NexusEnv
git checkout -b nihal

# 2. Create root pyproject.toml (see Section 1)

# 3. Set up venv
uv venv .venv --python 3.14
uv sync

# 4. Verify setup
.venv/bin/python -c "from openenv.core.env_server.interfaces import Environment; print('OpenEnv OK')"
.venv/bin/python -c "from mcp.server.fastmcp import FastMCP; print('FastMCP OK')"
.venv/bin/python -c "import gradio; print('Gradio OK')"

# 5. Start building sentinelops_arena/models.py
```
