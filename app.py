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
    scores_text = json.dumps(scores, indent=2)

    score_df = build_score_progression_df(log)
    attack_df = build_attack_timeline_df(log)

    return html, scores_text, score_df, attack_df


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

    comparison_json = {
        "untrained_scores": result["untrained"]["scores"],
        "trained_scores": result["trained"]["scores"],
        "improvement": {
            agent: round(
                result["trained"]["scores"][agent]
                - result["untrained"]["scores"][agent],
                2,
            )
            for agent in result["trained"]["scores"]
        },
    }

    return (
        untrained_html,
        trained_html,
        verdict_html,
        comparison_df,
        untrained_score_df,
        trained_score_df,
        json.dumps(comparison_json, indent=2),
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

with gr.Blocks(title="SentinelOps Arena") as demo:

    # Header banner
    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        # ============================================================
        # Tab 1: Run Episode
        # ============================================================
        with gr.TabItem("Run Episode"):
            with gr.Row():
                seed_input = gr.Number(
                    value=42, label="Random Seed", precision=0
                )
                trained_toggle = gr.Checkbox(
                    value=False, label="Use Trained Worker"
                )
                run_btn = gr.Button("Run Episode", variant="primary")

            with gr.Row():
                with gr.Column(scale=2):
                    replay_output = gr.HTML(label="Episode Replay")
                with gr.Column(scale=1):
                    scores_output = gr.Code(
                        label="Final Scores", language="json"
                    )

            with gr.Accordion("Score Progression & Attack Timeline", open=True):
                with gr.Row():
                    score_plot = gr.LinePlot(
                        x="tick",
                        y="score",
                        color="agent",
                        label="Cumulative Score Progression",
                        height=300,
                    )
                    attack_plot = gr.BarPlot(
                        x="attack_type",
                        y="count",
                        color="attack_type",
                        label="Attack Timeline",
                        height=300,
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
            gr.Markdown(
                "Compare how an **untrained** worker vs a **trained** worker "
                "handles the same attack sequence."
            )
            with gr.Row():
                comp_seed = gr.Number(
                    value=42, label="Random Seed", precision=0
                )
                comp_btn = gr.Button("Run Comparison", variant="primary")

            # Verdict stats
            verdict_output = gr.HTML(label="Training Impact")

            with gr.Row():
                untrained_output = gr.HTML(label="Untrained Worker")
                trained_output = gr.HTML(label="Trained Worker")

            with gr.Accordion("Score Comparison Charts", open=True):
                comparison_bar = gr.BarPlot(
                    x="agent",
                    y="score",
                    color="type",
                    label="Score Comparison: Untrained vs Trained",
                    height=300,
                )
                with gr.Row():
                    untrained_score_plot = gr.LinePlot(
                        x="tick",
                        y="score",
                        color="agent",
                        label="Untrained Score Progression",
                        height=250,
                    )
                    trained_score_plot = gr.LinePlot(
                        x="tick",
                        y="score",
                        color="agent",
                        label="Trained Score Progression",
                        height=250,
                    )

            comparison_output = gr.Code(
                label="Score Details", language="json"
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
                inspect_seed = gr.Number(
                    value=42, label="Random Seed", precision=0
                )
                inspect_btn = gr.Button("Inspect", variant="primary")

            config_output = gr.HTML(label="Environment Configuration")

            with gr.Accordion("Customers (CRM)", open=False):
                customers_table = gr.Dataframe(
                    label="All Customers",
                    headers=["customer_id", "name", "tier", "region", "lifetime_value"],
                )

            with gr.Accordion("Invoices (Billing)", open=False):
                invoices_table = gr.Dataframe(
                    label="All Invoices",
                    headers=["invoice_id", "customer_id", "amount", "status"],
                )

            with gr.Accordion("Tickets (Support)", open=False):
                tickets_table = gr.Dataframe(
                    label="All Tickets",
                    headers=["ticket_id", "customer_id", "subject", "priority", "status", "sla_deadline_tick"],
                )

            with gr.Accordion("Task Queue", open=False):
                tasks_table = gr.Dataframe(
                    label="Task Queue",
                    headers=["task_id", "customer_id", "task_type", "message", "arrival_tick"],
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
