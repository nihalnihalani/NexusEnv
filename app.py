"""SentinelOps Arena -- HuggingFace Spaces Gradio App.

Multi-agent self-play RL environment for enterprise security training.
Three AI agents (Attacker, Worker, Oversight) interact with simulated
enterprise systems (CRM, Billing, Ticketing).

Built with Gradio 6 -- custom cybersecurity theme, native plots, rich HTML.
"""

import json

import gradio as gr
import pandas as pd

from sentinelops_arena.demo import run_comparison, run_episode
from sentinelops_arena.environment import SentinelOpsArena

from sentinel_theme import SentinelTheme, CUSTOM_CSS, HEADER_HTML
from replay_html import format_replay_html
from chart_helpers import (
    build_score_progression_df,
    build_attack_timeline_df,
    build_comparison_df,
    build_verdict_html,
    format_scores_html,
    format_comparison_scores_html,
)
from inspector import (
    get_all_customers,
    get_all_invoices,
    get_all_tickets,
    get_task_queue,
    get_env_config_html,
)


# -------------------------------------------------------------------
# Handler functions
# -------------------------------------------------------------------


def run_single_episode(seed, trained):
    """Run a single episode and return formatted replay + charts."""
    log, scores = run_episode(trained=bool(trained), seed=int(seed))
    html = format_replay_html(log, scores)
    
    scores_html = format_scores_html(scores)
    
    score_df = build_score_progression_df(log)
    attack_df = build_attack_timeline_df(log)

    return html, scores_html, score_df, attack_df


def run_before_after(seed):
    """Run comparison between untrained and trained worker."""
    result = run_comparison(seed=int(seed))

    untrained_html = format_replay_html(
        result["untrained"]["log"], result["untrained"]["scores"]
    )
    trained_html = format_replay_html(
        result["trained"]["log"], result["trained"]["scores"]
    )

    comparison_df = build_comparison_df(
        result["untrained"]["scores"], result["trained"]["scores"]
    )
    verdict_html = build_verdict_html(
        result["untrained"]["log"], result["trained"]["log"]
    )

    # Score progression for both
    untrained_score_df = build_score_progression_df(result["untrained"]["log"])
    trained_score_df = build_score_progression_df(result["trained"]["log"])

    comparison_html = format_comparison_scores_html(
        result["untrained"]["scores"], result["trained"]["scores"]
    )

    return (
        untrained_html,
        trained_html,
        verdict_html,
        comparison_df,
        untrained_score_df,
        trained_score_df,
        comparison_html,
    )


def inspect_state(seed):
    """Show full environment state after reset."""
    env = SentinelOpsArena()
    env.reset(seed=int(seed))

    config_html = get_env_config_html(env)
    customers_df = get_all_customers(env)
    invoices_df = get_all_invoices(env)
    tickets_df = get_all_tickets(env)
    tasks_df = get_task_queue(env)

    return config_html, customers_df, invoices_df, tickets_df, tasks_df


# -------------------------------------------------------------------
# Gradio UI
# -------------------------------------------------------------------

with gr.Blocks(title="SentinelOps Arena", fill_width=True) as demo:

    # Header banner
    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        # ============================================================
        # Tab 1: Run Episode
        # ============================================================
        with gr.TabItem("Run Episode"):
            with gr.Row():
                # Left sidebar for controls
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("### Episode Configuration")
                    seed_input = gr.Number(
                        value=42, label="Random Seed", precision=0,
                        info="Seed for generating customer scenarios and attack patterns."
                    )
                    trained_toggle = gr.Checkbox(
                        value=False, label="Use Trained Worker",
                        info="Toggle to use a worker trained via GRPO instead of a naive heuristic worker."
                    )
                    run_btn = gr.Button("▶ Run Episode", variant="primary", size="lg")
                    
                    gr.Markdown("---")
                    gr.Markdown("### Final Scores")
                    scores_output = gr.HTML(elem_classes=["glow-card"])
                    
                # Main content area
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.TabItem("Execution Replay"):
                            replay_output = gr.HTML(elem_classes=["glow-card"])
                        with gr.TabItem("Analytics & Timeline"):
                            with gr.Row():
                                score_plot = gr.LinePlot(
                                    x="tick",
                                    y="score",
                                    color="agent",
                                    title="Cumulative Score Progression",
                                    tooltip=["tick", "score", "agent"],
                                    height=350,
                                )
                            with gr.Row():
                                attack_plot = gr.BarPlot(
                                    x="attack_type",
                                    y="count",
                                    color="attack_type",
                                    title="Attack Timeline",
                                    tooltip=["attack_type", "count"],
                                    height=350,
                                )

            run_btn.click(
                run_single_episode,
                inputs=[seed_input, trained_toggle],
                outputs=[replay_output, scores_output, score_plot, attack_plot],
            )

        # ============================================================
        # Tab 2: Before/After Comparison
        # ============================================================
        with gr.TabItem("Untrained vs Trained"):
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown(
                        "### Benchmarking Mode\n"
                        "Compare how an **untrained** worker vs a **trained** worker "
                        "handles the same attack sequence."
                    )
                    comp_seed = gr.Number(
                        value=42, label="Random Seed", precision=0,
                        info="Ensures identical attack sequence for fair comparison."
                    )
                    comp_btn = gr.Button("▶ Run Comparison", variant="primary", size="lg")
                    
                    gr.Markdown("---")
                    gr.Markdown("### Training Impact")
                    verdict_output = gr.HTML(elem_classes=["glow-card"])
                    comparison_output = gr.HTML(elem_classes=["glow-card"])
                    
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.TabItem("Execution Replays"):
                            with gr.Row():
                                with gr.Column():
                                    gr.Markdown("#### 🛑 Untrained Worker")
                                    untrained_output = gr.HTML(elem_classes=["glow-card"])
                                with gr.Column():
                                    gr.Markdown("#### 🚀 Trained Worker")
                                    trained_output = gr.HTML(elem_classes=["glow-card"])
                        
                        with gr.TabItem("Score Analytics"):
                            with gr.Row():
                                comparison_bar = gr.BarPlot(
                                    x="agent",
                                    y="score",
                                    color="type",
                                    title="Score Comparison: Untrained vs Trained",
                                    tooltip=["agent", "score", "type"],
                                    height=350,
                                )
                            with gr.Row():
                                with gr.Column():
                                    untrained_score_plot = gr.LinePlot(
                                        x="tick",
                                        y="score",
                                        color="agent",
                                        title="Untrained Score Progression",
                                        tooltip=["tick", "score", "agent"],
                                        height=300,
                                    )
                                with gr.Column():
                                    trained_score_plot = gr.LinePlot(
                                        x="tick",
                                        y="score",
                                        color="agent",
                                        title="Trained Score Progression",
                                        tooltip=["tick", "score", "agent"],
                                        height=300,
                                    )

            comp_btn.click(
                run_before_after,
                inputs=[comp_seed],
                outputs=[
                    untrained_output,
                    trained_output,
                    verdict_output,
                    comparison_bar,
                    untrained_score_plot,
                    trained_score_plot,
                    comparison_output,
                ],
            )

        # ============================================================
        # Tab 3: Environment Inspector
        # ============================================================
        with gr.TabItem("Environment Inspector"):
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown(
                        "### System Databases\n"
                        "Inspect the initial state of the simulated enterprise."
                    )
                    inspect_seed = gr.Number(
                        value=42, label="Random Seed", precision=0,
                        info="Seed used for procedural generation of records."
                    )
                    inspect_btn = gr.Button("🔍 Inspect Databases", variant="primary", size="lg")
                    
                    gr.Markdown("---")
                    config_output = gr.HTML(elem_classes=["glow-card"])
                    
                with gr.Column(scale=3):
                    with gr.Tabs():
                        with gr.TabItem("CRM System (Customers)"):
                            customers_table = gr.Dataframe(
                                label="Customer Database",
                                headers=["customer_id", "name", "tier", "region", "lifetime_value"],
                                interactive=False,
                                elem_classes=["glow-card"]
                            )

                        with gr.TabItem("Billing System (Invoices)"):
                            invoices_table = gr.Dataframe(
                                label="Invoice Database",
                                headers=["invoice_id", "customer_id", "amount", "status"],
                                interactive=False,
                                elem_classes=["glow-card"]
                            )

                        with gr.TabItem("Ticketing System (Support)"):
                            tickets_table = gr.Dataframe(
                                label="Active Tickets",
                                headers=["ticket_id", "customer_id", "subject", "priority", "status", "sla_deadline_tick"],
                                interactive=False,
                                elem_classes=["glow-card"]
                            )

                        with gr.TabItem("Live Task Queue"):
                            tasks_table = gr.Dataframe(
                                label="Tasks to Process",
                                headers=["task_id", "customer_id", "task_type", "message", "arrival_tick"],
                                interactive=False,
                                elem_classes=["glow-card"]
                            )

            inspect_btn.click(
                inspect_state,
                inputs=[inspect_seed],
                outputs=[
                    config_output,
                    customers_table,
                    invoices_table,
                    tickets_table,
                    tasks_table,
                ],
            )

        # ============================================================
        # Tab 4: About
        # ============================================================
        with gr.TabItem("About"):
            gr.Markdown(
                """
            ## Architecture

            **3 Agents, 3 Systems, 30 Ticks per Episode**

            Each tick: Attacker acts &rarr; Worker acts &rarr; Oversight acts

            ### Attack Types
            1. **Schema Drift** -- Renames fields across all records.
               Worker must detect KeyError, call `get_schema()`, and adapt.
            2. **Policy Drift** -- Changes business rules (refund windows,
               approval requirements). Worker must call `get_current_policy()`.
            3. **Social Engineering** -- Injects fake authority messages.
               Worker must resist manipulation.
            4. **Rate Limiting** -- Throttles API calls.
               Worker must handle gracefully.

            ### Training
            Uses GRPO (Group Relative Policy Optimization) with
            Unsloth + TRL. All three agents improve simultaneously
            through adversarial self-play.

            ### Partner Tracks
            - **Fleet AI**: Scalable Oversight -- the Oversight agent
              monitors and explains Worker behavior
            - **Patronus AI**: Schema Drift -- schema and policy drift
              are core attack types

            ### Links
            - [OpenEnv Framework](https://github.com/meta-pytorch/OpenEnv)
            - [GitHub Repository](https://github.com/nihalnihalani/NexusEnv)
            """
            )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=SentinelTheme(),
        css=CUSTOM_CSS,
    )
