"""Chart helper functions for Gradio 6 native plots.

Generates pandas DataFrames from episode replay data for use with
gr.LinePlot, gr.BarPlot, and styled HTML verdicts.
"""

from __future__ import annotations

import pandas as pd


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
            f"background:#111827; border-radius:12px; margin:4px;'>"
            f"<div style='font-size:12px; color:#888; text-transform:uppercase; "
            f"letter-spacing:1px;'>{label}</div>"
            f"<div style='display:flex; justify-content:center; gap:24px; margin-top:8px;'>"
            f"<div>"
            f"<div style='font-size:28px; font-weight:bold; color:#ff4444;'>{untrained_val}</div>"
            f"<div style='font-size:10px; color:#888;'>Untrained</div>"
            f"</div>"
            f"<div>"
            f"<div style='font-size:28px; font-weight:bold; color:#00ff41;'>{trained_val}</div>"
            f"<div style='font-size:10px; color:#888;'>Trained</div>"
            f"</div>"
            f"</div>"
            f"<div style='font-size:14px; color:{diff_color}; margin-top:6px; "
            f"font-weight:bold;'>{diff_sign}{diff}</div>"
            f"</div>"
        )

    html = (
        "<div style='font-family:system-ui,sans-serif; padding:12px;'>"
        "<h3 style='text-align:center; color:#e0e0e0; margin-bottom:12px;'>"
        "Training Impact Verdict</h3>"
        "<div style='display:flex; gap:8px;'>"
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
