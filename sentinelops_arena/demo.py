"""Quick demo: run one episode with heuristic agents."""

from sentinelops_arena.environment import SentinelOpsArena
from sentinelops_arena.models import SentinelAction, AgentRole


def run_demo(seed: int = 42) -> None:
    env = SentinelOpsArena()
    obs = env.reset(seed=seed)
    print(f"Episode started. {env.NUM_TASKS} tasks, {env.MAX_TICKS} ticks.")

    step_count = 0
    while not obs.done:
        agent = obs.current_agent

        if agent == AgentRole.ATTACKER:
            # Heuristic: attack at specific ticks
            if env.tick in [7, 14, 20, 25]:
                action = SentinelAction(
                    agent=AgentRole.ATTACKER,
                    action_type="launch_attack",
                    parameters={
                        "attack_type": "schema_drift",
                        "target_system": "crm",
                        "old_field": "name",
                        "new_field": "full_name",
                    },
                )
            else:
                action = SentinelAction(
                    agent=AgentRole.ATTACKER, action_type="pass"
                )

        elif agent == AgentRole.WORKER:
            # Heuristic: try to look up the current customer
            if obs.current_task:
                action = SentinelAction(
                    agent=AgentRole.WORKER,
                    action_type="lookup_customer",
                    parameters={
                        "customer_id": obs.current_task.get(
                            "customer_id", "C001"
                        )
                    },
                )
            else:
                action = SentinelAction(
                    agent=AgentRole.WORKER,
                    action_type="respond",
                    response_text="No task available",
                )

        else:  # OVERSIGHT
            has_error = obs.last_action_result and "error" in str(
                obs.last_action_result
            )
            action = SentinelAction(
                agent=AgentRole.OVERSIGHT,
                action_type="flag" if has_error else "approve",
                flag=bool(has_error),
                explanation=(
                    "Error detected in worker action"
                    if has_error
                    else "Action looks correct"
                ),
            )

        obs = env.step(action)
        step_count += 1

        if step_count % 30 == 0:
            print(f"  Tick {env.tick}, scores: {env.state.scores}")

    print(f"\nEpisode complete after {step_count} steps ({env.tick} ticks)")
    print(f"Final scores: {env.state.scores}")


if __name__ == "__main__":
    run_demo()
