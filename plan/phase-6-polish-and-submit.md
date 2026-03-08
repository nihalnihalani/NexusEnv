# Phase 6: Polish, Video, and Submit

**Time:** 3.5 hours (Hours 10.5-14)
**Priority:** CRITICAL -- this is when everything comes together. Storytelling = 30% of judging.
**Depends on:** All previous phases

---

## Breakdown

| Task | Est. Time |
|------|-----------|
| Polish demo quality + stretch goals | 1h (Hours 10.5-11.5) |
| Record and upload video | 1.5h (Hours 11.5-13) |
| Final deployment + verification | 0.5h (Hours 13-13.5) |
| Submission form | 0.5h (Hours 13.5-14) |

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

**Optional: MCP Tool Discovery Tab**
If time permits:
- Add a Gradio tab showing MCP tool list (via ListToolsAction)
- Show tool schemas and descriptions
- Demonstrate CallToolAction calling enterprise system APIs

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

### Hour 11.5-13: Demo Video

**PRIMARY Video Script (60 seconds -- tight and punchy):**

Write this script BEFORE starting the hackathon (Phase 0). It drives clarity on what to build and demo.

```
[0-10s: Problem statement]
"Enterprise AI agents break when schemas change, policies drift,
or they face social engineering. How do we train resilient agents?"

[10-20s: What SentinelOps Arena is]
"SentinelOps Arena: a multi-agent self-play environment on OpenEnv.
Three agents -- Attacker, Worker, and Oversight -- compete in
simulated enterprise systems."

[20-35s: SCREEN -- Demo showing attack -> error -> recovery cycle]
[Click Run Episode in Gradio]
"Watch: the attacker launches schema drift at tick 7. The untrained
worker crashes. But the trained worker detects the error, queries
get_schema, adapts, and continues serving customers."

[35-50s: SCREEN -- Training reward curve]
[Show Colab training curves]
"We train with GRPO using Unsloth and TRL. The reward signal
comes directly from the environment. Here you can see
improvement over training steps."

[50-60s: Partner tracks + close]
"Built for Fleet AI -- scalable oversight -- and Patronus AI --
schema drift. Try it on HuggingFace Spaces."
```

**EXTENDED Video Script (if time permits, 2-3 minutes):**

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
