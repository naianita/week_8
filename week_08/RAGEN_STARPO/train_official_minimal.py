"""
Training Script for Official WebShop - MINIMAL CONFIG FOR LOCAL TESTING
Configured for: 1-5 epochs, 100 products, batch size 4
Runs in ~30 minutes on Mac for testing/debugging
"""
import sys
import os
import torch
import torch.optim as optim
from torch.distributions import Categorical
import json
from typing import List, Dict

# Add project to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.official_webshop_wrapper import OfficialWebShopWrapper
from ragen.official_agent import OfficialWebShopAgent, SimpleTokenizer
from ragen.stage1_vstar import compute_gae_advantages
from ragen.stage2_policy_opt import compute_ppo_loss


def collect_trajectory_official(env_wrapper, agent, tokenizer, device, max_steps=15):
    """
    Collect a single trajectory using the official WebShop.

    Returns:
        trajectory: List of steps with observations, actions, rewards
        total_reward: Cumulative reward
    """
    trajectory = []
    obs, info = env_wrapper.reset()
    done = False
    total_reward = 0
    step = 0

    while not done and step < max_steps:
        # Tokenize observation
        obs_tokens = tokenizer.tokenize(obs, max_len=200).to(device)

        # Get available actions
        available_actions = env_wrapper.get_available_actions(obs)

        # Select action
        with torch.no_grad():
            action_str, log_prob, value = agent.select_action(
                obs_tokens, available_actions, tokenizer.vocab
            )

        # Take step in environment
        next_obs, reward, done, step_info = env_wrapper.step(action_str)

        # Store trajectory
        trajectory.append({
            'obs': obs,
            'obs_tokens': obs_tokens,
            'action': action_str,
            'log_prob': log_prob,
            'value': value,
            'reward': reward,
            'done': done
        })

        obs = next_obs
        total_reward += reward
        step += 1

    return trajectory, total_reward


def collect_batch_trajectories(env_fn, agent, tokenizer, device, num_episodes=4):
    """
    Collect multiple episodes for batch training.

    Args:
        env_fn: Function that creates environment
        agent: Agent model
        tokenizer: Text tokenizer
        device: torch device
        num_episodes: Number of episodes to collect

    Returns:
        batch_data: Dictionary with batched tensors
    """
    all_trajectories = []
    episode_rewards = []

    for ep in range(num_episodes):
        env = env_fn()
        trajectory, total_reward = collect_trajectory_official(
            env, agent, tokenizer, device
        )
        all_trajectories.extend(trajectory)
        episode_rewards.append(total_reward)

        if (ep + 1) % 2 == 0:
            avg_reward = sum(episode_rewards[-2:]) / 2
            print(f"  Episode {ep+1}/{num_episodes}, Avg Reward: {avg_reward:.3f}")

    # Convert to batched tensors
    obs_tokens = torch.cat([t['obs_tokens'] for t in all_trajectories], dim=0)
    log_probs = torch.stack([t['log_prob'] for t in all_trajectories])
    values = torch.stack([t['value'] for t in all_trajectories])
    rewards = torch.tensor([t['reward'] for t in all_trajectories], dtype=torch.float32, device=device)
    dones = torch.tensor([1.0 if t['done'] else 0.0 for t in all_trajectories], dtype=torch.float32, device=device)

    # For PPO, we need action indices. Use dummy action encoding for now
    # This is simplified - proper implementation would use proper action encoding
    action_indices = torch.zeros(len(all_trajectories), dtype=torch.long, device=device)

    return {
        'obs_tokens': obs_tokens,
        'action_indices': action_indices,
        'old_log_probs': log_probs,
        'values': values,
        'rewards': rewards,
        'dones': dones,
        'episode_rewards': episode_rewards
    }


def train_official_minimal(
    num_epochs: int = 1,
    num_products: int = 100,
    batch_size: int = 4,
    lr: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coeff: float = 0.5,
    entropy_coeff: float = 0.1,
    device: str = 'cpu'
):
    """
    Main training function - MINIMAL CONFIG for local testing.

    Args:
        num_epochs: Number of training epochs (1-5 for testing)
        num_products: Number of products to use (100 for testing)
        batch_size: Number of episodes per update (4 for testing)
        lr: Learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_epsilon: PPO clip epsilon
        value_coeff: Value loss coefficient
        entropy_coeff: Entropy bonus coefficient
        device: 'cpu' or 'cuda'
    """
    print("=" * 70)
    print("MINIMAL CONFIG TRAINING - Official WebShop")
    print("=" * 70)
    print(f"Config: {num_epochs} epochs, {num_products} products, batch size {batch_size}")
    print(f"Device: {device}")
    print("=" * 70)

    device = torch.device(device)

    # Initialize tokenizer
    print("\nInitializing tokenizer...")
    tokenizer = SimpleTokenizer(max_vocab_size=3000)  # Smaller vocab for speed

    # Build vocab from sample texts
    sample_texts = [
        "Instruction: Find red running shoes for men",
        "WebShop search results showing products",
        "Click to view product details and buy",
        "Price: $29.99 to $199.99",
        "Click buy now to purchase item"
    ]
    tokenizer.build_vocab(sample_texts)
    print(f"✓ Vocab size: {tokenizer.vocab_size()}")

    # Initialize agent
    print("\nInitializing agent...")
    agent = OfficialWebShopAgent(
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=64,  # Smaller for speed
        hidden_dim=128
    ).to(device)

    num_params = sum(p.numel() for p in agent.parameters())
    print(f"✓ Agent created with {num_params:,} parameters")

    # Optimizer
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    # Environment factory
    def make_env():
        return OfficialWebShopWrapper(num_products=num_products)

    # Training loop
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70 + "\n")

    training_history = []
    best_reward = -float('inf')

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 70)

        # Collect batch of trajectories
        batch_data = collect_batch_trajectories(
            make_env, agent, tokenizer, device, num_episodes=batch_size
        )

        avg_episode_reward = sum(batch_data['episode_rewards']) / len(batch_data['episode_rewards'])
        print(f"  Avg Episode Reward: {avg_episode_reward:.3f}")

        # Compute advantages using GAE
        with torch.no_grad():
            # Bootstrap value (use zero for terminal state)
            last_value = torch.zeros(1, device=device)
            values_with_bootstrap = torch.cat([batch_data['values'], last_value])

        advantages, value_targets = compute_gae_advantages(
            batch_data['rewards'],
            values_with_bootstrap,
            batch_data['dones'],
            gamma,
            gae_lambda
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        print("  Updating policy...")
        update_losses = []

        for update_step in range(2):  # Fewer updates for speed
            # Forward pass
            obs_tokens = batch_data['obs_tokens']
            action_logits, new_values = agent(obs_tokens)

            # Compute PPO loss
            # Note: This is simplified - proper implementation would compute action log probs
            # For now, use a simplified loss
            value_loss = ((new_values.squeeze() - value_targets) ** 2).mean()
            policy_loss = -advantages.mean()  # Simplified
            entropy = 0.01  # Placeholder

            total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            update_losses.append(total_loss.item())

        avg_loss = sum(update_losses) / len(update_losses)
        print(f"  Loss: {avg_loss:.4f}")

        # Track history
        training_history.append({
            'epoch': epoch + 1,
            'avg_reward': avg_episode_reward,
            'loss': avg_loss
        })

        # Save best model
        if avg_episode_reward > best_reward:
            best_reward = avg_episode_reward
            os.makedirs('models', exist_ok=True)
            torch.save({
                'agent_state_dict': agent.state_dict(),
                'tokenizer_vocab': tokenizer.vocab,
                'epoch': epoch + 1,
                'reward': best_reward
            }, 'models/official_agent_minimal.pth')
            print(f"  ✓ Saved best model (reward: {best_reward:.3f})")

        print()

    # Training complete
    print("=" * 70)
    print(f"Training Complete!")
    print(f"Best Reward: {best_reward:.3f}")
    print("=" * 70)

    # Save training history
    os.makedirs('results', exist_ok=True)
    with open('results/official_minimal_training.json', 'w') as f:
        json.dump({
            'config': {
                'num_epochs': num_epochs,
                'num_products': num_products,
                'batch_size': batch_size,
                'lr': lr
            },
            'history': training_history,
            'best_reward': best_reward
        }, f, indent=2)

    return agent, tokenizer


if __name__ == "__main__":
    # Minimal config for local testing
    print("\n" + "=" * 70)
    print("OFFICIAL WEBSHOP - MINIMAL LOCAL TESTING")
    print("=" * 70)
    print("\nThis config is designed for:")
    print("  - Quick testing/debugging (30 min)")
    print("  - Verifying code works")
    print("  - Running on Mac without GPU")
    print("\nFor real training, use:")
    print("  - 200 epochs")
    print("  - 50,000 products")
    print("  - Batch size 16")
    print("  - GPU (Colab/Modal)")
    print("=" * 70 + "\n")

    # Parse command line args if provided
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs (default: 1)')
    parser.add_argument('--products', type=int, default=100, help='Number of products (default: 100)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    args = parser.parse_args()

    # Train
    agent, tokenizer = train_official_minimal(
        num_epochs=args.epochs,
        num_products=args.products,
        batch_size=args.batch_size,
        device=args.device
    )

    print("\n✓ Training script completed successfully!")
    print("To run with different config:")
    print("  python train_official_minimal.py --epochs 5 --products 100 --batch_size 4")
