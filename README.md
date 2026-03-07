# OpenEnv Hackathon Project

Built for the [OpenEnv Hackathon](https://cerebralvalley.ai/e/openenv-hackathon-sf) (March 7-8, 2026)

## Quick Start

```bash
# Setup
python3.12 -m venv .venv
source .venv/bin/activate
pip install "openenv-core[core]>=0.2.1"

# Run environment locally
cd hackathon_env
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
openev/
├── hackathon_env/           # OpenEnv environment
│   ├── models.py            # Action/Observation data models
│   ├── client.py            # Environment client
│   ├── server/
│   │   ├── hackathon_env_environment.py  # Core environment logic
│   │   ├── app.py           # FastAPI server
│   │   └── Dockerfile       # Container config
│   ├── openenv.yaml         # OpenEnv spec
│   └── pyproject.toml       # Dependencies
├── train.py                 # Training script (TRL + GRPO)
└── README.md
```

## Deployment

### HuggingFace Spaces

```bash
# Build & push to HF Spaces
cd hackathon_env
openenv push --space <your-hf-username>/hackathon-env
```

### Local Docker

```bash
cd hackathon_env
docker build -t hackathon-env:latest -f server/Dockerfile .
docker run -p 8000:8000 hackathon-env:latest
```

## Training

See `train.py` for the minimal training script using HF TRL's GRPOTrainer with OpenEnv integration.

## Tech Stack

- **OpenEnv** 0.2.1 - Environment framework
- **HuggingFace TRL** - RL training (GRPO)
- **Unsloth** - Fast fine-tuning (2x speed, 70% less VRAM)
