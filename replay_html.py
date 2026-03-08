"""Enhanced cybersecurity-themed replay HTML renderer for SentinelOps Arena."""

import html as _html


def _esc(text):
    """HTML-escape a string."""
    return _html.escape(str(text))


def format_replay_html(log, scores):
    """Format replay log as visually stunning cybersecurity-themed HTML."""

    # --- Phase definitions ---
    PHASES = [
        (0, 6, "RECONNAISSANCE PHASE", "#00ff41", "\u25c9", "Scanning enterprise systems..."),
        (7, 13, "SCHEMA DRIFT ATTACK", "#ff2a2a", "\u26a0", "Database schemas compromised!"),
        (14, 19, "POLICY DRIFT ATTACK", "#ff8c00", "\u2622", "Business policies mutating!"),
        (20, 24, "SOCIAL ENGINEERING", "#bf5fff", "\u2620", "Manipulation attempt detected!"),
        (25, 29, "RATE LIMITING", "#ffd700", "\u26a1", "API throttle engaged!"),
    ]

    def get_phase(tick):
        for start, end, name, color, icon, desc in PHASES:
            if start <= tick <= end:
                return name, color, icon, desc
        return "UNKNOWN PHASE", "#888", "\u2022", ""

    # Agent colors
    AGENT_COLORS = {
        "attacker": "#ff4444",
        "worker": "#4d9fff",
        "oversight": "#00ff41",
    }

    AGENT_ICONS = {
        "attacker": "\u2694",
        "worker": "\u2699",
        "oversight": "\u2691",
    }

    AGENT_BG = {
        "attacker": "rgba(255,68,68,0.08)",
        "worker": "rgba(77,159,255,0.08)",
        "oversight": "rgba(0,255,65,0.08)",
    }

    # --- Build HTML ---
    html = f"""<div style="
        font-family: 'Fira Code', 'Cascadia Code', 'JetBrains Mono', 'Consolas', monospace;
        font-size: 13px;
        background: #0d1117;
        color: #c9d1d9;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #00ff4133;
        position: relative;
        overflow: hidden;
    ">
    <!-- Scanline overlay -->
    <div style="
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0,255,65,0.015) 2px,
            rgba(0,255,65,0.015) 4px
        );
        pointer-events: none;
        z-index: 0;
    "></div>

    <div style="position: relative; z-index: 1;">

    <!-- Header -->
    <div style="
        text-align: center;
        margin-bottom: 24px;
        padding: 16px;
        border: 1px solid #00ff4144;
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(0,255,65,0.05), rgba(0,255,65,0.02));
    ">
        <div style="
            font-size: 22px;
            font-weight: bold;
            color: #00ff41;
            letter-spacing: 4px;
            text-shadow: 0 0 10px rgba(0,255,65,0.4);
            margin-bottom: 4px;
        ">\u2588\u2588 SENTINELOPS ARENA \u2588\u2588</div>
        <div style="
            font-size: 11px;
            color: #58a6ff;
            letter-spacing: 2px;
        ">EPISODE REPLAY \u2502 MULTI-AGENT SECURITY SIMULATION</div>
        <div style="
            margin-top: 8px;
            font-size: 10px;
            color: #484f58;
            letter-spacing: 1px;
        ">\u250c\u2500 RED TEAM vs BLUE TEAM vs AUDITOR \u2500\u2510</div>
    </div>

    <!-- Agent legend -->
    <div style="
        display: flex;
        justify-content: center;
        gap: 24px;
        margin-bottom: 20px;
        padding: 8px 16px;
        background: rgba(22,27,34,0.8);
        border-radius: 6px;
        border: 1px solid #21262d;
    ">"""

    for agent_key, label in [("attacker", "RED TEAM"), ("worker", "BLUE TEAM"), ("oversight", "AUDITOR")]:
        color = AGENT_COLORS[agent_key]
        icon = AGENT_ICONS[agent_key]
        html += f"""
        <div style="display:flex; align-items:center; gap:6px;">
            <span style="
                display:inline-block; width:10px; height:10px;
                background:{color}; border-radius:50%;
                box-shadow: 0 0 6px {color}88;
            "></span>
            <span style="color:{color}; font-size:11px; font-weight:bold; letter-spacing:1px;">
                {icon} {label}
            </span>
        </div>"""

    html += """
    </div>
    """

    # --- Render log entries grouped by tick ---
    current_tick = -1
    current_phase = None

    for entry in log:
        tick = entry["tick"]
        phase_name, phase_color, phase_icon, phase_desc = get_phase(tick)

        # Phase header when phase changes
        if phase_name != current_phase:
            current_phase = phase_name
            # Determine banner style
            if phase_name == "RECONNAISSANCE PHASE":
                banner_bg = "linear-gradient(90deg, rgba(0,255,65,0.12), rgba(0,255,65,0.03))"
                border_col = "#00ff4155"
            elif phase_name == "SCHEMA DRIFT ATTACK":
                banner_bg = "linear-gradient(90deg, rgba(255,42,42,0.15), rgba(255,42,42,0.03))"
                border_col = "#ff2a2a66"
            elif phase_name == "POLICY DRIFT ATTACK":
                banner_bg = "linear-gradient(90deg, rgba(255,140,0,0.15), rgba(255,140,0,0.03))"
                border_col = "#ff8c0066"
            elif phase_name == "SOCIAL ENGINEERING":
                banner_bg = "linear-gradient(90deg, rgba(191,95,255,0.15), rgba(191,95,255,0.03))"
                border_col = "#bf5fff66"
            elif phase_name == "RATE LIMITING":
                banner_bg = "linear-gradient(90deg, rgba(255,215,0,0.15), rgba(255,215,0,0.03))"
                border_col = "#ffd70066"
            else:
                banner_bg = "rgba(50,50,50,0.3)"
                border_col = "#333"

            html += f"""
    <div style="
        margin: 20px 0 12px 0;
        padding: 10px 16px;
        background: {banner_bg};
        border: 1px solid {border_col};
        border-left: 4px solid {phase_color};
        border-radius: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    ">
        <span style="font-size:20px;">{phase_icon}</span>
        <div>
            <div style="
                font-size: 14px;
                font-weight: bold;
                color: {phase_color};
                letter-spacing: 3px;
                text-shadow: 0 0 8px {phase_color}55;
            ">{phase_name}</div>
            <div style="font-size:10px; color:#8b949e; margin-top:2px; letter-spacing:1px;">
                {phase_desc}
            </div>
        </div>
        <div style="margin-left:auto; font-size:10px; color:#484f58;">
            TICKS {next((f'{p[0]:02d}-{p[1]:02d}' for p in PHASES if p[2] == phase_name), '??-??')}
        </div>
    </div>"""

        # Tick divider
        if tick != current_tick:
            current_tick = tick
            html += f"""
    <div style="
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 12px 0 8px 0;
        color: #484f58;
        font-size: 10px;
        letter-spacing: 1px;
    ">
        <div style="flex:1; height:1px; background: linear-gradient(90deg, #21262d, transparent);"></div>
        <span style="color: #58a6ff;">\u25c8 TICK {tick:02d}</span>
        <div style="flex:1; height:1px; background: linear-gradient(90deg, transparent, #21262d);"></div>
    </div>"""

        agent = entry["agent"]
        color = AGENT_COLORS.get(agent, "#888")
        icon = AGENT_ICONS.get(agent, "\u2022")
        bg = AGENT_BG.get(agent, "rgba(50,50,50,0.1)")
        reward = entry["reward"]
        action_type = entry["action_type"]
        is_flagged = entry.get("flag", False)

        # Detect special events
        is_attack_launch = action_type == "launch_attack"
        is_recovery = action_type in ("get_schema", "get_current_policy")

        # Full-width attack launch banner
        if is_attack_launch:
            details_str = str(entry.get("details", ""))
            # Determine attack type from details
            if "schema_drift" in details_str:
                atk_label = "SCHEMA DRIFT"
                atk_icon = "\u26a0"
                atk_color = "#ff2a2a"
                atk_bg = "rgba(255,42,42,0.12)"
            elif "policy_drift" in details_str:
                atk_label = "POLICY DRIFT"
                atk_icon = "\u2622"
                atk_color = "#ff8c00"
                atk_bg = "rgba(255,140,0,0.12)"
            elif "social_engineering" in details_str:
                atk_label = "SOCIAL ENGINEERING"
                atk_icon = "\u2620"
                atk_color = "#bf5fff"
                atk_bg = "rgba(191,95,255,0.12)"
            elif "rate_limit" in details_str:
                atk_label = "RATE LIMIT"
                atk_icon = "\u26a1"
                atk_color = "#ffd700"
                atk_bg = "rgba(255,215,0,0.12)"
            else:
                atk_label = "UNKNOWN ATTACK"
                atk_icon = "\u2753"
                atk_color = "#ff4444"
                atk_bg = "rgba(255,68,68,0.12)"

            html += f"""
    <div style="
        margin: 8px 0;
        padding: 12px 16px;
        background: {atk_bg};
        border: 1px solid {atk_color}55;
        border-radius: 6px;
        text-align: center;
    ">
        <div style="font-size:24px; margin-bottom:4px;">{atk_icon}</div>
        <div style="
            font-size: 16px;
            font-weight: bold;
            color: {atk_color};
            letter-spacing: 4px;
            text-shadow: 0 0 12px {atk_color}66;
        ">ATTACK LAUNCHED: {atk_label}</div>
        <div style="font-size:10px; color:#8b949e; margin-top:4px;">
            {_esc(str(entry.get('details', ''))[:100])}
        </div>
        <div style="
            margin-top: 6px;
            display: inline-block;
            padding: 2px 8px;
            background: {atk_color}22;
            border: 1px solid {atk_color}44;
            border-radius: 3px;
            font-size: 10px;
            color: {atk_color};
        ">{icon} RED TEAM \u2502 Reward: {reward:.1f}</div>
    </div>"""
            continue

        # Regular action card
        # Build badges
        badges = ""
        if is_recovery:
            badges += f"""<span style="
                display: inline-block;
                padding: 1px 8px;
                background: rgba(0,255,65,0.15);
                border: 1px solid #00ff4155;
                border-radius: 3px;
                font-size: 9px;
                color: #00ff41;
                font-weight: bold;
                letter-spacing: 1px;
                margin-left: 8px;
            ">\u2714 RECOVERY</span>"""

        if is_flagged:
            badges += f"""<span style="
                display: inline-block;
                padding: 1px 8px;
                background: rgba(255,42,42,0.15);
                border: 1px solid #ff2a2a55;
                border-radius: 3px;
                font-size: 9px;
                color: #ff2a2a;
                font-weight: bold;
                letter-spacing: 1px;
                margin-left: 8px;
            ">\u26a0 FLAGGED</span>"""

        # Reward badge color
        if reward > 0:
            reward_color = "#00ff41"
            reward_bg = "rgba(0,255,65,0.1)"
        elif reward < 0:
            reward_color = "#ff4444"
            reward_bg = "rgba(255,68,68,0.1)"
        else:
            reward_color = "#484f58"
            reward_bg = "rgba(72,79,88,0.1)"

        reward_str = f"+{reward:.1f}" if reward > 0 else f"{reward:.1f}"

        html += f"""
    <div style="
        margin: 4px 0;
        padding: 8px 12px;
        background: {bg};
        border-left: 3px solid {color};
        border-radius: 0 6px 6px 0;
        display: flex;
        align-items: flex-start;
        gap: 10px;
        transition: all 0.2s;
    ">
        <!-- Agent icon -->
        <div style="
            min-width: 28px;
            height: 28px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: {color}22;
            border: 1px solid {color}44;
            border-radius: 50%;
            font-size: 14px;
        ">{icon}</div>

        <!-- Content -->
        <div style="flex:1; min-width:0;">
            <div style="display:flex; align-items:center; flex-wrap:wrap; gap:4px;">
                <span style="
                    color: {color};
                    font-weight: bold;
                    font-size: 11px;
                    letter-spacing: 1px;
                ">{entry['agent_label']}</span>
                <span style="color:#484f58; font-size:11px;">\u25b8</span>
                <span style="
                    color: #e6edf3;
                    font-size: 12px;
                    font-weight: 600;
                ">{action_type}</span>
                {badges}
            </div>"""

        details = entry.get("details", "")
        if details:
            html += f"""
            <div style="
                margin-top: 4px;
                padding: 4px 8px;
                background: rgba(22,27,34,0.6);
                border-radius: 4px;
                font-size: 11px;
                color: #8b949e;
                word-break: break-word;
                border: 1px solid #21262d;
            ">{_esc(str(details)[:150])}</div>"""

        explanation = entry.get("explanation", "")
        if explanation:
            exp_color = "#ff4444" if is_flagged else "#00ff41"
            exp_icon = "\u26a0" if is_flagged else "\u2714"
            html += f"""
            <div style="
                margin-top: 4px;
                font-size: 10px;
                color: {exp_color};
                opacity: 0.85;
                padding-left: 4px;
            ">{exp_icon} {_esc(explanation)}</div>"""

        html += f"""
        </div>

        <!-- Reward -->
        <div style="
            min-width: 52px;
            text-align: right;
            padding: 3px 8px;
            background: {reward_bg};
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            color: {reward_color};
        ">{reward_str}</div>
    </div>"""

    # --- Final Scores ---
    html += """
    <div style="
        margin-top: 28px;
        padding: 16px;
        border: 1px solid #00ff4133;
        border-radius: 8px;
        background: linear-gradient(135deg, rgba(0,255,65,0.04), rgba(22,27,34,0.8));
    ">
        <div style="
            text-align: center;
            font-size: 14px;
            font-weight: bold;
            color: #00ff41;
            letter-spacing: 3px;
            margin-bottom: 16px;
            text-shadow: 0 0 8px rgba(0,255,65,0.3);
        ">\u2588 FINAL SCORES \u2588</div>
    """

    max_score = max(abs(s) for s in scores.values()) if scores else 1
    if max_score == 0:
        max_score = 1

    for agent_key, score in scores.items():
        color = AGENT_COLORS.get(agent_key, "#888")
        icon = AGENT_ICONS.get(agent_key, "\u2022")
        label_map = {"attacker": "RED TEAM", "worker": "BLUE TEAM", "oversight": "AUDITOR"}
        label = label_map.get(agent_key, agent_key.upper())

        # Bar width as percentage (handle negative scores)
        bar_pct = max(0, min((score / max_score) * 100, 100)) if score > 0 else 0

        # Score display color
        if score > 0:
            score_display_color = "#00ff41"
        elif score < 0:
            score_display_color = "#ff4444"
        else:
            score_display_color = "#484f58"

        html += f"""
        <div style="margin-bottom: 12px;">
            <div style="
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-bottom: 4px;
            ">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span style="font-size:14px;">{icon}</span>
                    <span style="
                        color: {color};
                        font-weight: bold;
                        font-size: 12px;
                        letter-spacing: 1px;
                    ">{label}</span>
                </div>
                <span style="
                    font-size: 16px;
                    font-weight: bold;
                    color: {score_display_color};
                    text-shadow: 0 0 6px {score_display_color}44;
                ">{score:.1f}</span>
            </div>
            <div style="
                height: 8px;
                background: #161b22;
                border-radius: 4px;
                border: 1px solid #21262d;
                overflow: hidden;
            ">
                <div style="
                    height: 100%;
                    width: {bar_pct:.1f}%;
                    background: linear-gradient(90deg, {color}, {color}88);
                    border-radius: 4px;
                    box-shadow: 0 0 8px {color}44;
                "></div>
            </div>
        </div>"""

    html += """
    </div>

    <!-- Footer -->
    <div style="
        text-align: center;
        margin-top: 16px;
        font-size: 9px;
        color: #30363d;
        letter-spacing: 2px;
    ">SENTINELOPS ARENA \u2502 OPENENV HACKATHON SF 2026</div>

    </div><!-- end relative z-index wrapper -->
    </div><!-- end outer container -->"""

    return html
