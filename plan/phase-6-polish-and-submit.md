# Phase 6: Polish, Video, and Submit

**Time:** 4 hours (Hours 10-14)
**Priority:** CRITICAL -- this is when everything comes together
**Depends on:** All previous phases

---

## Breakdown

| Task | Est. Time |
|------|-----------|
| Polish demo quality (before/after, visuals) | 1h (Hours 10-11) |
| Stretch goals (if time) | 1h (Hours 11-12) |
| Final deployment + verification | 1h (Hours 12-13) |
| Video script + recording + upload | 45 min (Hours 13-13:45) |
| Submission form | 15 min (Hours 13:45-14) |

---

## Step-by-Step Instructions

### Hour 10-11: Polish Demo Quality

**Improve Gradio app:**
- Add attack timeline visualization (which attacks at which ticks)
- Add color-coded severity indicators for oversight flags
- Run 5 episodes, show aggregate statistics (avg scores)
- Improve HTML formatting (better colors, icons, spacing)
- Add episode statistics panel (tasks completed, attacks survived, violations caught)

**Improve before/after comparison:**
- Show specific moments where trained worker outperforms untrained
- Highlight "key moments" in the replay (attack launched, error recovered, social eng resisted)
- Add score differential chart

**Optional: MCP-X Demo Tab**
If MCP-X is working:
- Add a tab showing per-agent tool lists
- Demonstrate tool isolation (worker can't call launch_attack)
- Show JWT-based authentication in action

### Hour 11-12: Stretch Goals (Pick Based on Time)

**Priority order:**
1. **Compound attacks** -- 2 simultaneous attacks (schema drift + social engineering)
2. **More task variety** -- additional customer scenarios for richer demos
3. **Better training** -- run more epochs, capture better curves
4. **Episode replay export** -- JSON format for external analysis
5. **Richer prompt dataset** -- 50+ diverse enterprise scenarios

### Hour 12-13: Final Deployment + Verification

**Deploy checklist:**
```bash
# 1. Final push to HF Spaces
cd sentinelops_arena
git add -A
git commit -m "Final submission build"
# Push to HF Spaces repo

# 2. Verify HF Spaces
# - Navigate to public URL
# - Run Episode tab works
# - Comparison tab works
# - Inspector tab works
# - No errors in Space logs

# 3. Verify Colab notebook
# - Open fresh Colab instance
# - Run all cells from scratch
# - Verify model loads
# - Verify training starts
# - Capture training curves screenshot

# 4. Final code cleanup
# - Remove debug prints
# - Check all imports work
# - Verify pyproject.toml is correct
# - README has clear setup instructions
```

**Final smoke test:**
```bash
# Local verification
python -m sentinelops_arena.demo
python app.py  # Gradio loads
uvicorn sentinelops_arena.server:app --port 8000  # HTTP API works
curl http://localhost:8000/schema  # Schema endpoint returns
```

### Hour 13-13:45: Demo Video

**Video Script (aim for 1-3 minutes):**

```
[SLIDE 1: Title - 5 seconds]
"SentinelOps Arena: Multi-Agent Self-Play for Enterprise Security"

[SCREEN: Gradio app - 15 seconds]
"SentinelOps Arena is a multi-agent self-play training environment
built on OpenEnv. Three AI agents -- Attacker, Worker, and
Oversight -- interact with simulated enterprise systems."

[SCREEN: Run Episode tab - 20 seconds]
"Let me show you an episode. The attacker launches schema drift
at tick 7 -- renaming customer_id to account_id. Watch what
happens when the untrained worker hits this."
[Click Run Episode with trained=False]
"The worker crashes on the schema change. It doesn't know how
to recover."

[SCREEN: Comparison tab - 20 seconds]
"Now let's see the trained worker handle the same attacks."
[Click Run Comparison]
"The trained worker detects the KeyError, calls get_schema to
discover the new field name, and continues serving customers.
Score improvement is clear."

[SCREEN: Inspector tab - 10 seconds]
"Under the hood, we have 15 customers, 15 invoices, 10 tickets,
and 30 customer tasks per episode. Four attack types: schema
drift, policy drift, social engineering, and rate limiting."

[SCREEN: Colab notebook - 15 seconds]
"Training uses GRPO with Unsloth and TRL. The environment
provides reward signals directly to the training loop. Here
you can see the reward improving over training steps."
[Show training curves]

[SLIDE 2: Partner Tracks - 10 seconds]
"We target two partner tracks:
Fleet AI -- our Oversight agent monitors and explains Worker behavior
Patronus AI -- schema and policy drift are core attack types"

[SLIDE 3: Architecture - 10 seconds]
"Built on OpenEnv with MCP tools and an MCP-X gateway for
per-agent tool isolation. Three agents, three systems,
self-play training via GRPO."

[END - 5 seconds]
"SentinelOps Arena. Try it on HuggingFace Spaces."
```

**Recording instructions:**
1. Open Gradio app in browser
2. Use screen recording tool (OBS, QuickTime, or Loom)
3. Follow the script above
4. Keep pacing steady -- don't rush
5. Total target: 1-3 minutes (max 5)

**Upload to YouTube:**
- Title: "SentinelOps Arena -- OpenEnv Hackathon SF 2026"
- Description: Link to HF Spaces + Colab notebook
- Set as "Unlisted" (or public)
- Copy the YouTube URL for submission

### Hour 13:45-14: Submission

**Submission form fields:**

| Field | Value |
|-------|-------|
| Team Name | (your team name) |
| Project Description | SentinelOps Arena is a multi-agent self-play RL environment built on OpenEnv where three AI agents -- Attacker (red team), Worker (blue team), and Oversight (auditor) -- interact with simulated enterprise systems (CRM, Billing, Ticketing). The Attacker launches schema drift, policy drift, and social engineering attacks. The Worker must detect disruptions, adapt, and continue serving customers. The Oversight agent monitors worker actions and flags policy violations. Through adversarial self-play with GRPO training, all three agents improve simultaneously -- creating an autocurriculum that produces hardened enterprise AI agents. |
| HuggingFace Spaces Link | https://huggingface.co/spaces/nihalnihalani/sentinelops-arena |
| Demo Video (YouTube) | (YouTube URL from above) |
| Minimal Training Script | (Colab notebook URL) |
| Partner Tracks | Fleet AI (Scalable Oversight), Patronus AI (Schema Drift) |

---

## VERIFY

### Final Verification Checklist

```
BEFORE SUBMITTING, verify ALL of these:

[ ] HF Spaces URL loads (not erroring)
[ ] Run Episode produces replay with scores
[ ] Comparison shows trained > untrained
[ ] YouTube video plays (not processing)
[ ] YouTube video is < 5 minutes
[ ] YouTube video shows: Gradio demo, attack/adapt cycle, training curves
[ ] Colab notebook URL is accessible
[ ] Colab notebook: Cell 1 installs succeed
[ ] Colab notebook: Model loads
[ ] Colab notebook: Training starts (at least 1 step)
[ ] Submission form: all fields filled
[ ] Submission form: partner tracks selected
[ ] All links work when opened in incognito browser
```

---

## DEBUG: Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| YouTube video "processing" | Just uploaded | Wait 5-10 min, YouTube processes in background |
| HF Spaces down at submission time | Spaces overloaded | Have local demo.py as backup, record video from local |
| Colab notebook won't open | Sharing permissions | Set sharing to "Anyone with the link can view" |
| Video too long | Over-explaining | Cut to key moments, skip setup/install footage |
| Submission form rejects URL | Wrong format | Ensure full URL with https:// |
| Spaces error after deploy | Missing dependency | Check Space build logs, add to requirements.txt |
| Video quality poor | Screen recording settings | Record at 1080p, use high bitrate |

---

## EXIT CRITERIA

- [ ] HF Spaces URL is publicly accessible and working
- [ ] Demo video uploaded to YouTube and accessible
- [ ] Demo video shows: Gradio app, attack/adapt/flag cycle, training curves
- [ ] Colab notebook URL accessible and runnable
- [ ] Submission form submitted with ALL required fields
- [ ] All links verified in incognito browser

---

## ROLLBACK PLAN

If Phase 6 takes longer than expected:
1. **Cut polish** -- submit with whatever Gradio app you have from Phase 4
2. **Simplify video** -- screen record just the "Run Episode" tab, narrate over it. 60 seconds.
3. **Skip stretch goals** -- go straight to deployment + video
4. **Emergency video** -- record terminal running `demo.py`, narrate the output. No Gradio needed.
5. **Absolute minimum** -- submit HF Spaces link + Colab link + 30-second video showing it works

**Deadline priority:**
- DO NOT miss the 1:00 PM Sunday deadline
- Submit at LEAST 30 minutes early (12:30 PM) to account for form issues
- If at hour 13 things aren't working, submit what you have. A working partial submission beats a broken full submission.

---

## Video Script Alternative (60-second version)

If short on time, use this minimal script:

```
[SCREEN: Gradio app, 10 sec]
"SentinelOps Arena -- three AI agents compete in a simulated enterprise environment."

[SCREEN: Run Episode, 20 sec]
"The attacker launches schema drift and policy drift attacks.
The trained worker detects and adapts. The oversight agent flags violations."
[Show replay scrolling]

[SCREEN: Comparison, 15 sec]
"Trained worker significantly outperforms untrained."
[Show score comparison]

[SCREEN: Colab, 10 sec]
"Training uses GRPO with Unsloth and TRL on OpenEnv."
[Show training curves]

[END, 5 sec]
"Built for Fleet AI and Patronus AI partner tracks."
```
