"""
Evaluation Script for Official WebShop Agent
Tests trained agent on official WebShop environment
"""
import sys
import os
import torch
import json
from typing import Dict, List

# Add project to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.official_webshop_wrapper import OfficialWebShopWrapper
from ragen.official_agent import OfficialWebShopAgent, SimpleTokenizer


def evaluate_agent(
    agent: OfficialWebShopAgent,
    tokenizer: SimpleTokenizer,
    num_products: int = 100,
    num_episodes: int = 10,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict:
    """
    Evaluate agent on official WebShop.

    Args:
        agent: Trained agent
        tokenizer: Text tokenizer
        num_products: Number of products in environment
        num_episodes: Number of episodes to evaluate
        device: 'cpu' or 'cuda'
        verbose: Print episode details

    Returns:
        results: Dictionary with evaluation metrics
    """
    device = torch.device(device)
    agent.eval()

    env = OfficialWebShopWrapper(num_products=num_products)

    episode_rewards = []
    episode_steps = []
    successes = 0

    print("=" * 70)
    print(f"Evaluating Agent on {num_episodes} Episodes")
    print("=" * 70)

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 15

        if verbose:
            print(f"\nEpisode {ep + 1}/{num_episodes}")
            instruction = info.get('instruction', 'N/A')
            print(f"Instruction: {instruction}")
            print("-" * 70)

        while not done and step_count < max_steps:
            # Tokenize observation
            obs_tokens = tokenizer.tokenize(obs, max_len=200).to(device)

            # Get available actions
            available_actions = env.get_available_actions(obs)

            # Select action
            with torch.no_grad():
                action_str, log_prob, value = agent.select_action(
                    obs_tokens, available_actions, tokenizer.vocab
                )

            # Take step
            next_obs, reward, done, step_info = env.step(action_str)

            total_reward += reward
            step_count += 1

            if verbose:
                print(f"  Step {step_count}: {action_str}")
                print(f"    Reward: {reward:.3f}, Done: {done}")

            obs = next_obs

        episode_rewards.append(total_reward)
        episode_steps.append(step_count)

        # Count success (reward > 0.5 typically means correct purchase)
        if total_reward > 0.5:
            successes += 1

        if verbose:
            print(f"  Episode Reward: {total_reward:.3f}")
            print(f"  Steps: {step_count}")
            print()

    # Compute metrics
    success_rate = successes / num_episodes
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_steps = sum(episode_steps) / len(episode_steps)

    results = {
        'num_episodes': num_episodes,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps
    }

    print("=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Avg Reward: {avg_reward:.3f}")
    print(f"Avg Steps: {avg_steps:.1f}")
    print("=" * 70)

    return results


def load_agent(model_path: str, device: str = 'cpu'):
    """
    Load trained agent from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: 'cpu' or 'cuda'

    Returns:
        agent: Loaded agent
        tokenizer: Tokenizer with vocabulary
    """
    checkpoint = torch.load(model_path, map_location=device)

    # Rebuild tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.vocab = checkpoint['tokenizer_vocab']
    tokenizer.idx_to_word = {v: k for k, v in tokenizer.vocab.items()}

    # Create and load agent
    agent = OfficialWebShopAgent(
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=64,
        hidden_dim=128
    )
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.to(device)
    agent.eval()

    print(f"✓ Loaded agent from {model_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Reward: {checkpoint.get('reward', 'N/A'):.3f}")

    return agent, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/official_agent_minimal.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--num_products', type=int, default=100,
                        help='Number of products in environment')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu/cuda)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed episode info')
    parser.add_argument('--output', type=str, default='results/eval_official.json',
                        help='Output file for results')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"✗ Model not found: {args.model}")
        print("\nPlease train a model first:")
        print("  python train_official_minimal.py --epochs 1")
        sys.exit(1)

    # Load agent
    agent, tokenizer = load_agent(args.model, args.device)

    # Evaluate
    results = evaluate_agent(
        agent,
        tokenizer,
        num_products=args.num_products,
        num_episodes=args.num_episodes,
        device=args.device,
        verbose=args.verbose
    )

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {args.output}")
