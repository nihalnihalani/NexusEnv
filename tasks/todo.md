# SentinelOps Arena — Winning Hackathon Implementation Plan

## Gap Analysis (from codebase audit)

| Gap | Description | Priority |
|-----|-------------|----------|
| Scripted attacker | `HeuristicAttacker` fires at fixed ticks (7/14/20/25) — not adaptive | CRITICAL |
| No key metrics | ASR, Benign Task Success, FPR, MTTD not computed | CRITICAL |
| No metrics in Gradio | Dashboard shows scores but not security-specific metrics | HIGH |
| About tab outdated | Doesn't reflect the full narrative | MEDIUM |

## Implementation Tasks

### Task 1: Randomized Adaptive Attacker
- [x] Replace `HeuristicAttacker.ATTACK_SCHEDULE` with budget-based random strategy
- [x] Random attack type selection weighted by past success
- [x] Random timing (not fixed ticks)
- [x] Random target system selection
- [x] Varying social engineering messages (not just one template)
- [x] Keep budget constraint (10.0, cost 0.3 per attack)

### Task 2: Key Metrics Engine
- [x] Create `sentinelops_arena/metrics.py`
- [x] Compute from episode log:
  - Attack Success Rate (ASR) = attacks that caused worker failure / total attacks
  - Benign Task Success = successful tasks / total tasks attempted
  - False Positive Rate (FPR) = false flags / total oversight flags
  - Mean Time to Detect (MTTD) = avg ticks between attack and first detection

### Task 3: Metrics in Gradio Dashboard
- [x] Add metrics panel to Run Episode tab
- [x] Add metrics to Before/After comparison tab
- [x] Styled HTML cards matching the cybersecurity theme

### Task 4: Update About Tab
- [x] Full narrative matching the vision document
- [x] Key metrics definitions
- [x] Self-play explanation

## Verification
- [x] `python -c "from sentinelops_arena.demo import run_episode; run_episode()"` works
- [x] `python -c "from sentinelops_arena.metrics import compute_episode_metrics; print('OK')"` works
- [ ] Gradio app launches without errors
- [x] Randomized attacker produces different attack patterns across seeds
  - Seed 42: 10 attacks at ticks [1,2,4,11,13,17,18,19,21,27]
  - Seed 99: 10 attacks at ticks [1,2,5,12,20,23,25,27,28,29]
  - Seed 7: 12 attacks at ticks [1,2,3,4,5,7,9,12,14,20,25,28]
- [x] Metrics compute correctly (ASR, Benign Success, FPR, MTTD)
- [x] Trained worker outperforms untrained (30.0 vs 25.0 worker score)
