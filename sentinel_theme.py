"""SentinelOps Arena -- Custom Gradio 6 cybersecurity theme."""

from __future__ import annotations

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes


class SentinelTheme(Base):
    """Dark cybersecurity / hacking aesthetic theme for SentinelOps Arena."""

    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.red,
        neutral_hue: colors.Color | str = colors.slate,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font | str | list[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "SFMono-Regular",
            "monospace",
        ),
        font_mono: fonts.Font | str | list[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono"),
            "ui-monospace",
            "Consolas",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )

        # -- Core surface colors -------------------------------------------------
        self.body_background_fill = "#0a0f1a"
        self.body_background_fill_dark = "#0a0f1a"
        self.body_text_color = "#c9d1d9"
        self.body_text_color_dark = "#c9d1d9"
        self.body_text_color_subdued = "#8b949e"
        self.body_text_color_subdued_dark = "#8b949e"

        # -- Block / panel colours ------------------------------------------------
        self.block_background_fill = "#111827"
        self.block_background_fill_dark = "#111827"
        self.block_border_color = "#1e3a2f"
        self.block_border_color_dark = "#1e3a2f"
        self.block_border_width = "1px"
        self.block_label_background_fill = "#0d1520"
        self.block_label_background_fill_dark = "#0d1520"
        self.block_label_text_color = "#00ff41"
        self.block_label_text_color_dark = "#00ff41"
        self.block_shadow = "0 0 8px rgba(0, 255, 65, 0.08)"
        self.block_shadow_dark = "0 0 8px rgba(0, 255, 65, 0.08)"
        self.block_title_text_color = "#e6edf3"
        self.block_title_text_color_dark = "#e6edf3"

        # -- Borders & panels -----------------------------------------------------
        self.border_color_accent = "#00ff41"
        self.border_color_accent_dark = "#00ff41"
        self.border_color_primary = "#1e3a2f"
        self.border_color_primary_dark = "#1e3a2f"
        self.panel_background_fill = "#0d1520"
        self.panel_background_fill_dark = "#0d1520"
        self.panel_border_color = "#1e3a2f"
        self.panel_border_color_dark = "#1e3a2f"

        # -- Primary button (cyber green) -----------------------------------------
        self.button_primary_background_fill = "#00cc33"
        self.button_primary_background_fill_dark = "#00cc33"
        self.button_primary_background_fill_hover = "#00ff41"
        self.button_primary_background_fill_hover_dark = "#00ff41"
        self.button_primary_text_color = "#0a0f1a"
        self.button_primary_text_color_dark = "#0a0f1a"
        self.button_primary_border_color = "#00ff41"
        self.button_primary_border_color_dark = "#00ff41"
        self.button_primary_shadow = "0 0 12px rgba(0, 255, 65, 0.3)"

        # -- Secondary button -----------------------------------------------------
        self.button_secondary_background_fill = "#1a1f2e"
        self.button_secondary_background_fill_dark = "#1a1f2e"
        self.button_secondary_background_fill_hover = "#252b3b"
        self.button_secondary_background_fill_hover_dark = "#252b3b"
        self.button_secondary_text_color = "#c9d1d9"
        self.button_secondary_text_color_dark = "#c9d1d9"
        self.button_secondary_border_color = "#30363d"
        self.button_secondary_border_color_dark = "#30363d"

        # -- Inputs ---------------------------------------------------------------
        self.input_background_fill = "#0d1520"
        self.input_background_fill_dark = "#0d1520"
        self.input_border_color = "#1e3a2f"
        self.input_border_color_dark = "#1e3a2f"
        self.input_border_color_focus = "#00ff41"
        self.input_border_color_focus_dark = "#00ff41"
        self.input_placeholder_color = "#484f58"
        self.input_placeholder_color_dark = "#484f58"
        self.input_text_color = "#c9d1d9"

        # -- Checkbox / toggle ----------------------------------------------------
        self.checkbox_background_color = "#0d1520"
        self.checkbox_background_color_dark = "#0d1520"
        self.checkbox_background_color_selected = "#00cc33"
        self.checkbox_background_color_selected_dark = "#00cc33"
        self.checkbox_border_color = "#30363d"
        self.checkbox_border_color_dark = "#30363d"
        self.checkbox_border_color_selected = "#00ff41"
        self.checkbox_border_color_selected_dark = "#00ff41"
        self.checkbox_label_text_color = "#c9d1d9"

        # -- Table / code ---------------------------------------------------------
        self.table_border_color = "#1e3a2f"
        self.table_border_color_dark = "#1e3a2f"
        self.table_even_background_fill = "#111827"
        self.table_even_background_fill_dark = "#111827"
        self.table_odd_background_fill = "#0d1520"
        self.table_odd_background_fill_dark = "#0d1520"
        self.code_background_fill = "#0d1520"
        self.code_background_fill_dark = "#0d1520"

        # -- Shadows & misc ------------------------------------------------------
        self.shadow_spread = "4px"
        self.shadow_drop = "0 2px 6px rgba(0, 0, 0, 0.4)"
        self.shadow_drop_lg = "0 4px 16px rgba(0, 0, 0, 0.5)"


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
/* ====================== ROOT OVERRIDES ====================== */
:root {
    --sentinel-green: #00ff41;
    --sentinel-green-dim: #00cc33;
    --sentinel-red: #ff4444;
    --sentinel-blue: #4488ff;
    --sentinel-bg: #0a0f1a;
    --sentinel-surface: #111827;
    --sentinel-surface-alt: #0d1520;
    --sentinel-border: #1e3a2f;
    --sentinel-text: #c9d1d9;
}

/* ====================== GLOBAL ====================== */
.gradio-container {
    background: var(--sentinel-bg) !important;
    max-width: 1200px !important;
}

/* ====================== TAB HEADERS ====================== */
.tab-nav button {
    background: var(--sentinel-surface) !important;
    color: var(--sentinel-text) !important;
    border: 1px solid var(--sentinel-border) !important;
    border-bottom: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 10px 20px !important;
    transition: all 0.2s ease;
}

.tab-nav button:hover {
    background: #1a2332 !important;
    color: var(--sentinel-green) !important;
}

.tab-nav button.selected {
    background: var(--sentinel-surface-alt) !important;
    color: var(--sentinel-green) !important;
    border-top: 2px solid var(--sentinel-green) !important;
    box-shadow: 0 -2px 8px rgba(0, 255, 65, 0.15);
}

/* ====================== AGENT COLOR ACCENTS ====================== */
.attacker-red {
    border-left: 3px solid var(--sentinel-red) !important;
    padding-left: 12px !important;
}

.attacker-red .label-wrap span,
.attacker-red label span {
    color: var(--sentinel-red) !important;
}

.worker-blue {
    border-left: 3px solid var(--sentinel-blue) !important;
    padding-left: 12px !important;
}

.worker-blue .label-wrap span,
.worker-blue label span {
    color: var(--sentinel-blue) !important;
}

.oversight-green {
    border-left: 3px solid var(--sentinel-green) !important;
    padding-left: 12px !important;
}

.oversight-green .label-wrap span,
.oversight-green label span {
    color: var(--sentinel-green) !important;
}

/* ====================== GLOWING CARD BORDERS ====================== */
.glow-card {
    border: 1px solid var(--sentinel-border) !important;
    border-radius: 8px !important;
    box-shadow: 0 0 10px rgba(0, 255, 65, 0.06),
                inset 0 0 10px rgba(0, 255, 65, 0.02) !important;
    transition: box-shadow 0.3s ease, border-color 0.3s ease;
}

.glow-card:hover {
    border-color: rgba(0, 255, 65, 0.4) !important;
    box-shadow: 0 0 18px rgba(0, 255, 65, 0.12),
                inset 0 0 12px rgba(0, 255, 65, 0.04) !important;
}

/* ====================== PRIMARY BUTTONS GLOW ====================== */
button.primary {
    box-shadow: 0 0 12px rgba(0, 255, 65, 0.25) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
}

button.primary:hover {
    box-shadow: 0 0 20px rgba(0, 255, 65, 0.45) !important;
    transform: translateY(-1px);
}

/* ====================== HEADER BANNER ====================== */
.sentinel-header {
    text-align: center;
    padding: 32px 20px 24px;
    background: linear-gradient(180deg, #0d1a10 0%, var(--sentinel-bg) 100%);
    border-bottom: 1px solid var(--sentinel-border);
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}

.sentinel-header::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg,
        transparent 0%,
        var(--sentinel-green) 20%,
        var(--sentinel-red) 50%,
        var(--sentinel-blue) 80%,
        transparent 100%
    );
}

.sentinel-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00ff41 0%, #00cc88 40%, #44ffaa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px 0;
    letter-spacing: 0.04em;
    text-shadow: none;
}

.sentinel-header .subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.95rem;
    color: #8b949e;
    letter-spacing: 0.06em;
    margin-bottom: 20px;
}

/* ---- Agent badges ---- */
.agent-badges {
    display: flex;
    justify-content: center;
    gap: 16px;
    flex-wrap: wrap;
    margin-top: 8px;
}

.agent-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border: 1px solid;
}

.badge-red {
    color: var(--sentinel-red);
    border-color: rgba(255, 68, 68, 0.4);
    background: rgba(255, 68, 68, 0.08);
}

.badge-blue {
    color: var(--sentinel-blue);
    border-color: rgba(68, 136, 255, 0.4);
    background: rgba(68, 136, 255, 0.08);
}

.badge-green {
    color: var(--sentinel-green);
    border-color: rgba(0, 255, 65, 0.4);
    background: rgba(0, 255, 65, 0.08);
}

.badge-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}

.badge-red .badge-dot { background: var(--sentinel-red); }
.badge-blue .badge-dot { background: var(--sentinel-blue); }
.badge-green .badge-dot { background: var(--sentinel-green); }

/* Built-on OpenEnv badge */
.openenv-badge {
    display: inline-block;
    margin-top: 16px;
    padding: 4px 14px;
    border-radius: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #8b949e;
    border: 1px solid #30363d;
    background: rgba(255, 255, 255, 0.03);
    letter-spacing: 0.06em;
}

.openenv-badge a {
    color: #58a6ff;
    text-decoration: none;
}

.openenv-badge a:hover {
    text-decoration: underline;
}

/* ====================== ATTACK ALERT PULSE ====================== */
@keyframes alert-pulse {
    0%, 100% { box-shadow: 0 0 4px rgba(255, 68, 68, 0.2); }
    50% { box-shadow: 0 0 16px rgba(255, 68, 68, 0.4); }
}

.attack-alert {
    animation: alert-pulse 2.5s ease-in-out infinite;
    border: 1px solid rgba(255, 68, 68, 0.3) !important;
}

/* ====================== SCAN LINE DECORATION ====================== */
@keyframes scanline {
    0% { transform: translateY(-100%); }
    100% { transform: translateY(100%); }
}

.sentinel-header::after {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 100%;
    background: linear-gradient(
        180deg,
        transparent 0%,
        rgba(0, 255, 65, 0.02) 50%,
        transparent 100%
    );
    animation: scanline 8s linear infinite;
    pointer-events: none;
}

/* ====================== MARKDOWN / TEXT READABILITY ====================== */
.prose h1, .prose h2, .prose h3 {
    color: #e6edf3 !important;
}

.prose p, .prose li {
    color: #c9d1d9 !important;
}

.prose strong {
    color: #e6edf3 !important;
}

.prose a {
    color: #58a6ff !important;
}

.prose code {
    background: var(--sentinel-surface-alt) !important;
    color: var(--sentinel-green) !important;
    padding: 2px 6px;
    border-radius: 4px;
}

.prose hr {
    border-color: var(--sentinel-border) !important;
}

/* ====================== CODE BLOCKS ====================== */
.code-wrap, .cm-editor {
    background: var(--sentinel-surface-alt) !important;
    border: 1px solid var(--sentinel-border) !important;
}

/* ====================== SCROLLBAR ====================== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--sentinel-bg);
}

::-webkit-scrollbar-thumb {
    background: #30363d;
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #484f58;
}
"""

# ---------------------------------------------------------------------------
# Header HTML
# ---------------------------------------------------------------------------

HEADER_HTML = """\
<div class="sentinel-header">
    <h1>SentinelOps Arena</h1>
    <div class="subtitle">Multi-Agent Self-Play RL for Enterprise Security</div>
    <div class="agent-badges">
        <span class="agent-badge badge-red">
            <span class="badge-dot"></span>RED TEAM
        </span>
        <span class="agent-badge badge-blue">
            <span class="badge-dot"></span>BLUE TEAM
        </span>
        <span class="agent-badge badge-green">
            <span class="badge-dot"></span>AUDITOR
        </span>
    </div>
    <div class="openenv-badge">
        Built on <a href="https://github.com/meta-pytorch/OpenEnv" target="_blank">OpenEnv</a>
    </div>
</div>
"""
