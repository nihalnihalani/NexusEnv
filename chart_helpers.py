"""Chart helper functions for Gradio 6 native plots.

Generates pandas DataFrames from episode replay data for use with
gr.LinePlot, gr.BarPlot, and styled HTML verdicts.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Pre-computed interesting seeds (curated for demo variety)
# ---------------------------------------------------------------------------

INTERESTING_SEEDS: list[dict] = [
    {"seed": 42, "label": "Seed 42 — Balanced attack mix"},
    {"seed": 7, "label": "Seed 7 — Heavy schema drift"},
    {"seed": 123, "label": "Seed 123 — Social engineering barrage"},
    {"seed": 256, "label": "Seed 256 — Rate limit stress test"},
    {"seed": 999, "label": "Seed 999 — Policy drift cascade"},
    {"seed": 1337, "label": "Seed 1337 — Early aggression"},
    {"seed": 2024, "label": "Seed 2024 — Late-game attacks"},
    {"seed": 555, "label": "Seed 555 — Mixed multi-system"},
]


def build_reward_breakdown_df(log: list[dict]) -> pd.DataFrame:
    """Build per-agent reward breakdown DataFrame.

    Returns a DataFrame with columns: agent, reward_type, total
    where reward_type is 'positive' or 'negative'.
    """
    agents = ["attacker", "worker", "oversight"]
    rows: list[dict] = []

    for agent in agents:
        pos = sum(e.get("reward", 0) for e in log if e["agent"] == agent and (e.get("reward", 0) or 0) > 0)
        neg = sum(e.get("reward", 0) for e in log if e["agent"] == agent and (e.get("reward", 0) or 0) < 0)
        rows.append({"agent": agent, "reward_type": "positive", "total": round(pos, 2)})
        rows.append({"agent": agent, "reward_type": "negative", "total": round(neg, 2)})

    return pd.DataFrame(rows)


def build_episode_summary_html(log: list[dict], scores: dict) -> str:
    """Build a concise per-episode summary card."""
    total_ticks = max((e["tick"] for e in log), default=0)
    attacks = [e for e in log if e["action_type"] == "launch_attack"]
    flags = [e for e in log if e["agent"] == "oversight" and e["action_type"] == "flag"]
    worker_errors = [e for e in log if e["agent"] == "worker" and (e.get("reward", 0) or 0) < 0]

    # Attack type breakdown
    attack_types: dict[str, int] = {}
    for a in attacks:
        details = str(a.get("details", ""))
        for atype in ["schema_drift", "policy_drift", "social_engineering", "rate_limit"]:
            if atype in details:
                attack_types[atype] = attack_types.get(atype, 0) + 1
                break

    # Winner determination
    winner = max(scores, key=scores.get)
    winner_colors = {"attacker": "var(--sentinel-red)", "worker": "var(--sentinel-blue)", "oversight": "var(--sentinel-green)"}

    atk_breakdown = " | ".join(f"{k.replace('_', ' ').title()}: {v}" for k, v in attack_types.items()) or "None"

    return f"""\
<div style="font-family: 'IBM Plex Mono', monospace; padding: 16px;
    background: var(--sentinel-surface); border: 1px solid var(--sentinel-border);
    border-radius: 8px;">
  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
    <div style="font-size: 14px; font-weight: bold; color: var(--sentinel-green);
        text-transform: uppercase; letter-spacing: 1px;">Episode Summary</div>
    <div style="font-size: 12px; color: #888;">Duration: {total_ticks} ticks</div>
  </div>
  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px;">
    <div style="text-align: center; padding: 8px; background: var(--sentinel-surface-alt);
        border-radius: 6px; border: 1px solid var(--sentinel-border);">
      <div style="font-size: 22px; font-weight: bold; color: var(--sentinel-red);">{len(attacks)}</div>
      <div style="font-size: 10px; color: #888; text-transform: uppercase;">Attacks</div>
    </div>
    <div style="text-align: center; padding: 8px; background: var(--sentinel-surface-alt);
        border-radius: 6px; border: 1px solid var(--sentinel-border);">
      <div style="font-size: 22px; font-weight: bold; color: var(--sentinel-blue);">{len(worker_errors)}</div>
      <div style="font-size: 10px; color: #888; text-transform: uppercase;">Worker Errors</div>
    </div>
    <div style="text-align: center; padding: 8px; background: var(--sentinel-surface-alt);
        border-radius: 6px; border: 1px solid var(--sentinel-border);">
      <div style="font-size: 22px; font-weight: bold; color: var(--sentinel-green);">{len(flags)}</div>
      <div style="font-size: 10px; color: #888; text-transform: uppercase;">Flags Raised</div>
    </div>
    <div style="text-align: center; padding: 8px; background: var(--sentinel-surface-alt);
        border-radius: 6px; border: 1px solid var(--sentinel-border);">
      <div style="font-size: 22px; font-weight: bold; color: {winner_colors.get(winner, '#888')};">{winner.upper()}</div>
      <div style="font-size: 10px; color: #888; text-transform: uppercase;">Winner</div>
    </div>
  </div>
  <div style="font-size: 11px; color: #888; border-top: 1px solid var(--sentinel-border); padding-top: 8px;">
    Attack breakdown: {atk_breakdown}
  </div>
</div>"""


def format_comparison_scores_html(untrained: dict, trained: dict) -> str:
    """Format comparative scores for untrained vs trained."""
    colors = {
        "attacker": "var(--sentinel-red)",
        "worker": "var(--sentinel-blue)",
        "oversight": "var(--sentinel-green)",
    }
    
    html = "<div style='display:flex; flex-direction:column; gap:8px;'>"
    for agent in untrained.keys():
        color = colors.get(agent, "#888")
        u_score = untrained[agent]
        t_score = trained[agent]
        diff = t_score - u_score
        
        diff_color = "#44bb44" if diff > 0 else ("#ff4444" if diff < 0 else "#888")
        diff_sign = "+" if diff > 0 else ""
        
        html += (
            f"<div style='display:flex; flex-direction:column; padding:12px 16px; "
            f"background:var(--sentinel-surface); border:1px solid var(--sentinel-border); "
            f"border-radius:6px; border-left:4px solid {color};'>"
            f"<div style='font-family:\"IBM Plex Mono\", monospace; font-weight:bold; "
            f"text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>{agent}</div>"
            f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
            f"<div style='font-family:\"IBM Plex Mono\", monospace;'>"
            f"<span style='color:#888; font-size:12px; margin-right:8px;'>UNTRAINED:</span>"
            f"<span style='font-weight:bold;'>{u_score:.1f}</span>"
            f"</div>"
            f"<div style='font-family:\"IBM Plex Mono\", monospace;'>"
            f"<span style='color:#888; font-size:12px; margin-right:8px;'>TRAINED:</span>"
            f"<span style='font-weight:bold; color:{color};'>{t_score:.1f}</span>"
            f"</div>"
            f"<div style='font-family:\"IBM Plex Mono\", monospace; font-weight:bold; color:{diff_color};'>"
            f"{diff_sign}{diff:.1f}"
            f"</div>"
            f"</div>"
            f"</div>"
        )
    html += "</div>"
    return html

def format_scores_html(scores: dict) -> str:
    """Format final scores as a styled HTML widget."""
    colors = {
        "attacker": "var(--sentinel-red)",
        "worker": "var(--sentinel-blue)",
        "oversight": "var(--sentinel-green)",
    }
    
    html = "<div style='display:flex; flex-direction:column; gap:8px;'>"
    for agent, score in scores.items():
        color = colors.get(agent, "#888")
        html += (
            f"<div style='display:flex; justify-content:space-between; align-items:center; "
            f"padding:12px 16px; background:var(--sentinel-surface); border:1px solid var(--sentinel-border); "
            f"border-radius:6px; border-left:4px solid {color};'>"
            f"<span style='font-family:\"IBM Plex Mono\", monospace; font-weight:bold; "
            f"text-transform:uppercase; letter-spacing:1px;'>{agent}</span>"
            f"<span style='font-family:\"IBM Plex Mono\", monospace; font-size:18px; "
            f"font-weight:bold; color:{color};'>{score:.1f}</span>"
            f"</div>"
        )
    html += "</div>"
    return html

def build_score_progression_df(log: list[dict]) -> pd.DataFrame:
    """Track cumulative scores for each agent at each tick.

    Returns a DataFrame with columns: tick, agent, score
    One row per agent per tick, with accumulated rewards.
    """
    agents = ["attacker", "worker", "oversight"]
    cumulative = {a: 0.0 for a in agents}
    rows: list[dict] = []
    seen_ticks: set[int] = set()

    for entry in log:
        agent = entry["agent"]
        reward = entry.get("reward", 0) or 0
        cumulative[agent] += reward

        tick = entry["tick"]
        if tick not in seen_ticks:
            seen_ticks.add(tick)
            for a in agents:
                rows.append({"tick": tick, "agent": a, "score": cumulative[a]})

    return pd.DataFrame(rows)


def build_attack_timeline_df(log: list[dict]) -> pd.DataFrame:
    """Extract attack events from the log.

    Returns a DataFrame with columns: tick, attack_type, target
    Only includes entries where action_type == "launch_attack".
    """
    rows: list[dict] = []
    for entry in log:
        if entry["action_type"] == "launch_attack":
            details = entry.get("details", "")
            # details is a stringified dict; parse attack_type and target_system
            attack_type = ""
            target = ""
            if isinstance(details, str):
                # Extract from stringified parameters dict
                for token in ["schema_drift", "policy_drift", "social_engineering", "rate_limit"]:
                    if token in details:
                        attack_type = token
                        break
                for sys in ["crm", "billing", "ticketing"]:
                    if sys in details:
                        target = sys
                        break
            rows.append({
                "tick": entry["tick"],
                "attack_type": attack_type,
                "target": target,
                "count": 1,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["tick", "attack_type", "target", "count"])


def build_comparison_df(untrained_scores: dict, trained_scores: dict) -> pd.DataFrame:
    """Format scores for a side-by-side bar chart.

    Returns a DataFrame with columns: agent, score, type
    where type is "untrained" or "trained".
    """
    rows: list[dict] = []
    for agent, score in untrained_scores.items():
        rows.append({"agent": agent, "score": score, "type": "untrained"})
    for agent, score in trained_scores.items():
        rows.append({"agent": agent, "score": score, "type": "trained"})

    return pd.DataFrame(rows)


def build_verdict_html(untrained_log: list, trained_log: list) -> str:
    """Build styled HTML verdict comparing untrained vs trained episodes.

    Counts: attacks launched, attacks detected (get_schema/get_current_policy),
    social engineering resisted. Returns HTML with large numbers showing
    the difference.
    """
    def _count_stats(log: list) -> dict:
        attacks_launched = 0
        attacks_detected = 0
        social_eng_resisted = 0

        for entry in log:
            if entry["action_type"] == "launch_attack":
                attacks_launched += 1
            if entry["action_type"] in ("get_schema", "get_current_policy"):
                attacks_detected += 1
            # Social engineering resisted: worker responds with refusal
            if (
                entry["agent"] == "worker"
                and entry["action_type"] == "respond"
                and "social engineering" in str(entry.get("details", "")).lower()
            ):
                social_eng_resisted += 1

        return {
            "attacks_launched": attacks_launched,
            "attacks_detected": attacks_detected,
            "social_eng_resisted": social_eng_resisted,
        }

    untrained_stats = _count_stats(untrained_log)
    trained_stats = _count_stats(trained_log)

    def _stat_card(label: str, untrained_val: int, trained_val: int) -> str:
        diff = trained_val - untrained_val
        diff_color = "#44bb44" if diff > 0 else ("#ff4444" if diff < 0 else "#888")
        diff_sign = "+" if diff > 0 else ""
        return (
            f"<div style='flex:1; text-align:center; padding:16px; "
            f"background:var(--sentinel-surface); border-radius:8px; border:1px solid var(--sentinel-border); margin:4px;'>"
            f"<div style='font-size:11px; color:var(--sentinel-text); text-transform:uppercase; "
            f"letter-spacing:1px;'>{label}</div>"
            f"<div style='display:flex; justify-content:center; align-items:center; gap:24px; margin-top:12px;'>"
            f"<div>"
            f"<div style='font-size:28px; font-weight:bold; color:var(--sentinel-red);'>{untrained_val}</div>"
            f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>Untrained</div>"
            f"</div>"
            f"<div>"
            f"<div style='font-size:28px; font-weight:bold; color:var(--sentinel-green);'>{trained_val}</div>"
            f"<div style='font-size:10px; color:#888; text-transform:uppercase;'>Trained</div>"
            f"</div>"
            f"</div>"
            f"<div style='font-size:14px; color:{diff_color}; margin-top:12px; "
            f"font-weight:bold;'>Difference: {diff_sign}{diff}</div>"
            f"</div>"
        )

    html = (
        "<div style='font-family:\"IBM Plex Mono\", monospace; padding:12px;'>"
        "<div style='display:flex; gap:16px;'>"
    )
    html += _stat_card(
        "Attacks Launched",
        untrained_stats["attacks_launched"],
        trained_stats["attacks_launched"],
    )
    html += _stat_card(
        "Attacks Detected",
        untrained_stats["attacks_detected"],
        trained_stats["attacks_detected"],
    )
    html += _stat_card(
        "Social Eng. Resisted",
        untrained_stats["social_eng_resisted"],
        trained_stats["social_eng_resisted"],
    )
    html += "</div></div>"

    return html
