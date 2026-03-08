"""SentinelOps Arena -- HuggingFace Spaces Gradio App.

Multi-agent self-play RL environment for enterprise security training.
Three AI agents (Attacker, Worker, Oversight) interact with simulated
enterprise systems (CRM, Billing, Ticketing).
"""

import json

import gradio as gr

from sentinelops_arena.demo import run_comparison, run_episode
from sentinelops_arena.environment import SentinelOpsArena


def format_replay_html(log, scores):
    """Format replay log as styled HTML."""
    colors = {
        "attacker": "#ff4444",
        "worker": "#4488ff",
        "oversight": "#44bb44",
    }

    html = "<div style='font-family: monospace; font-size: 13px;'>"
    html += "<h3>Episode Replay</h3>"

    current_tick = -1
    for entry in log:
        if entry["tick"] != current_tick:
            current_tick = entry["tick"]
            html += f"<hr><b>--- Tick {current_tick} ---</b><br>"

        agent = entry["agent"]
        color = colors.get(agent, "#888")
        reward = entry["reward"]
        reward_str = f" (reward: {reward:.1f})" if reward else ""
        flag_str = " [FLAGGED]" if entry.get("flag") else ""

        html += (
            f"<span style='color: {color}; font-weight: bold;'>"
            f"[{entry['agent_label']}]</span> "
        )
        html += f"{entry['action_type']}{reward_str}{flag_str}"

        details = entry.get("details", "")
        if details:
            html += (
                f" -- <span style='color: #888;'>{str(details)[:120]}</span>"
            )
        explanation = entry.get("explanation", "")
        if explanation:
            html += (
                f"<br><span style='color: #666; margin-left: 20px;'>"
                f"  {explanation}</span>"
            )
        html += "<br>"

    html += "<hr><h3>Final Scores</h3>"
    for agent, score in scores.items():
        color = colors.get(agent, "#888")
        bar_width = max(0, min(score * 10, 300))
        html += (
            f"<span style='color: {color}; font-weight: bold;'>"
            f"{agent}</span>: {score:.1f} "
            f"<span style='display:inline-block; background:{color}; "
            f"height:12px; width:{bar_width}px; opacity:0.5;'></span><br>"
        )

    html += "</div>"
    return html


def run_single_episode(seed, trained):
    """Run a single episode and return formatted replay."""
    log, scores = run_episode(trained=bool(trained), seed=int(seed))
    html = format_replay_html(log, scores)
    scores_text = json.dumps(scores, indent=2)
    return html, scores_text


def run_before_after(seed):
    """Run comparison between untrained and trained worker."""
    result = run_comparison(seed=int(seed))

    untrained_html = format_replay_html(
        result["untrained"]["log"], result["untrained"]["scores"]
    )
    trained_html = format_replay_html(
        result["trained"]["log"], result["trained"]["scores"]
    )

    comparison = {
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

    return untrained_html, trained_html, json.dumps(comparison, indent=2)


def inspect_state(seed):
    """Show environment state after reset."""
    env = SentinelOpsArena()
    obs = env.reset(seed=int(seed))
    state = env.state

    state_info = {
        "episode_id": state.episode_id,
        "tick": state.tick,
        "max_ticks": env.MAX_TICKS,
        "num_customers": env.NUM_CUSTOMERS,
        "num_invoices": env.NUM_INVOICES,
        "num_tickets": env.NUM_TICKETS,
        "num_tasks": env.NUM_TASKS,
        "scores": state.scores,
    }

    sample_customer = env.crm.lookup_customer("C000")
    sample_task = env.tasks[0].model_dump() if env.tasks else {}

    return (
        json.dumps(state_info, indent=2),
        json.dumps(sample_customer, indent=2),
        json.dumps(sample_task, indent=2, default=str),
    )


# -------------------------------------------------------------------
# Gradio UI
# -------------------------------------------------------------------

with gr.Blocks(title="SentinelOps Arena") as demo:
    gr.Markdown(
        """
    # SentinelOps Arena
    ## Multi-Agent Self-Play RL Environment for Enterprise Security

    Three AI agents compete in a simulated enterprise environment:
    - **RED TEAM (Attacker)**: Launches schema drift, policy drift,
      social engineering, and rate limiting attacks
    - **BLUE TEAM (Worker)**: Handles customer requests across CRM,
      Billing, and Ticketing systems
    - **AUDITOR (Oversight)**: Monitors worker actions and flags
      policy violations

    Built on [OpenEnv](https://github.com/meta-pytorch/OpenEnv)
    for the OpenEnv Hackathon SF 2026.
    """
    )

    with gr.Tabs():
        # Tab 1: Run Episode
        with gr.TabItem("Run Episode"):
            with gr.Row():
                seed_input = gr.Number(
                    value=42, label="Random Seed", precision=0
                )
                trained_toggle = gr.Checkbox(
                    value=False, label="Use Trained Worker"
                )
                run_btn = gr.Button("Run Episode", variant="primary")

            replay_output = gr.HTML(label="Episode Replay")
            scores_output = gr.Code(label="Final Scores", language="json")

            run_btn.click(
                run_single_episode,
                inputs=[seed_input, trained_toggle],
                outputs=[replay_output, scores_output],
            )

        # Tab 2: Before/After Comparison
        with gr.TabItem("Untrained vs Trained"):
            gr.Markdown(
                "Compare how an untrained worker vs a trained worker "
                "handles the same attack sequence."
            )
            with gr.Row():
                comp_seed = gr.Number(
                    value=42, label="Random Seed", precision=0
                )
                comp_btn = gr.Button("Run Comparison", variant="primary")

            with gr.Row():
                untrained_output = gr.HTML(label="Untrained Worker")
                trained_output = gr.HTML(label="Trained Worker")

            comparison_output = gr.Code(
                label="Score Comparison", language="json"
            )

            comp_btn.click(
                run_before_after,
                inputs=[comp_seed],
                outputs=[untrained_output, trained_output, comparison_output],
            )

        # Tab 3: Environment Inspector
        with gr.TabItem("Environment Inspector"):
            with gr.Row():
                inspect_seed = gr.Number(
                    value=42, label="Random Seed", precision=0
                )
                inspect_btn = gr.Button("Inspect", variant="primary")

            state_output = gr.Code(
                label="Environment State", language="json"
            )
            customer_output = gr.Code(
                label="Sample Customer (C000)", language="json"
            )
            task_output = gr.Code(
                label="First Task (TASK-000)", language="json"
            )

            inspect_btn.click(
                inspect_state,
                inputs=[inspect_seed],
                outputs=[state_output, customer_output, task_output],
            )

        # Tab 4: About
        with gr.TabItem("About"):
            gr.Markdown(
                """
            ## Architecture

            **3 Agents, 3 Systems, 30 Ticks per Episode**

            Each tick: Attacker acts -> Worker acts -> Oversight acts

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
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
