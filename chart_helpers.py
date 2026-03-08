"""Chart helper functions for Gradio 6 native plots.

Generates pandas DataFrames from episode replay data for use with
gr.LinePlot, gr.BarPlot, and styled HTML verdicts.
"""

from __future__ import annotations

import pandas as pd


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
