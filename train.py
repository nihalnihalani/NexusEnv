"""
Minimal Training Script for OpenEnv Hackathon
==============================================
Uses HuggingFace TRL's GRPOTrainer with OpenEnv environment integration.

Run in Google Colab with GPU runtime:
    !pip install "openenv-core[core]>=0.2.1" trl transformers torch accelerate
    # Or with Unsloth for 2x faster training:
    !pip install unsloth "openenv-core[core]>=0.2.1" trl

Usage:
    python train.py --env_url https://<your-hf-space>.hf.space
"""

import argparse

from hackathon_env.client import HackathonEnv
from hackathon_env.models import HackathonAction


def collect_rollouts(env_url: str, prompts: list[str]) -> list[dict]:
    """
    Collect rollouts by interacting with the OpenEnv environment.

    Args:
        env_url: URL of the deployed OpenEnv environment
        prompts: List of prompts to send to the environment

    Returns:
        List of rollout dicts with prompt, completion, and reward
    """
    rollouts = []

    with HackathonEnv(base_url=env_url) as env:
        for prompt in prompts:
            env.reset()
            result = env.step(HackathonAction(message=prompt))

            rollouts.append({
                "prompt": prompt,
                "completion": result.observation.echoed_message,
                "reward": result.reward,
            })

    return rollouts


def reward_function(completions: list[str], **kwargs) -> list[float]:
    """
    Reward function for GRPO training.
    Extracts rewards from environment rollout results.
    """
    env_rewards = kwargs.get("env_reward", [])
    if env_rewards:
        return env_rewards
    # Fallback: simple length-based reward
    return [len(c) * 0.1 for c in completions]


def main():
    parser = argparse.ArgumentParser(description="Train with OpenEnv + TRL GRPO")
    parser.add_argument(
        "--env_url",
        type=str,
        default="http://localhost:8000",
        help="URL of the OpenEnv environment server",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model to train",
    )
    parser.add_argument(
        "--use_unsloth",
        action="store_true",
        help="Use Unsloth for faster training",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    args = parser.parse_args()

    print(f"Environment URL: {args.env_url}")
    print(f"Model: {args.model_name}")
    print(f"Using Unsloth: {args.use_unsloth}")

    # --- Step 1: Verify environment connectivity ---
    print("\n[1/3] Verifying environment connection...")
    with HackathonEnv(base_url=args.env_url) as env:
        result = env.reset()
        print(f"  Environment ready: {result.observation.echoed_message}")

        test_result = env.step(HackathonAction(message="test"))
        print(f"  Test step reward: {test_result.reward}")

    # --- Step 2: Load model ---
    print("\n[2/3] Loading model...")
    if args.use_unsloth:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=2048,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # --- Step 3: Train with GRPO ---
    print("\n[3/3] Starting GRPO training...")
    from trl import GRPOTrainer, GRPOConfig

    training_args = GRPOConfig(
        output_dir="./output",
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        max_completion_length=256,
        logging_steps=1,
        save_steps=100,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_function],
        args=training_args,
    )

    trainer.train()
    print("\nTraining complete! Model saved to ./output")


if __name__ == "__main__":
    main()
