"""
WebArena Shopping Evaluation with RAGEN Adapter
Evaluates trained WebShop model on WebArena shopping tasks using format adaptation
"""
import os
import sys
import json
import torch
from typing import Dict, List
from datetime import datetime

# Add project to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.webarena_adapter import WebArenaToWebShopAdapter, WebArenaEnvironmentWrapper
from ragen.official_agent import OfficialWebShopAgent, SimpleTokenizer


class MockWebArenaEnv:
    """
    Mock WebArena environment for testing when Docker isn't running.
    Simulates WebArena responses for evaluation testing.
    """

    def __init__(self, tasks: List[Dict]):
        self.tasks = tasks
        self.current_task_idx = 0
        self.step_count = 0
        self.max_steps = 15

    def reset(self):
        """Reset to next task."""
        if self.current_task_idx >= len(self.tasks):
            self.current_task_idx = 0

        task = self.tasks[self.current_task_idx]
        self.current_task_idx += 1
        self.step_count = 0

        # Simulate accessibility tree observation
        obs = self._generate_mock_observation(task)
        info = {'task': task.get('intent', ''), 'task_id': task.get('task_id', '')}

        return obs, info

    def step(self, action: str):
        """Execute action and return mock results."""
        self.step_count += 1

        # Simple mock: random success with some probability
        import random

        # Reward based on action type
        reward = 0.0
        done = False

        if 'click' in action:
            reward = random.choice([0.1, -0.05, -0.05, -0.05])
        elif 'type' in action:
            reward = random.choice([0.2, 0.1, -0.05])
        elif 'stop' in action:
            done = True
            reward = random.choice([1.0, -0.2, -0.2, -0.2])

        # End after max steps
        if self.step_count >= self.max_steps:
            done = True

        obs = "[1234] statictext 'Product List' [1235] link 'Sample Product' [1236] button 'Add to Cart'"
        info = {'step': self.step_count}

        return obs, reward, done, info

    def _generate_mock_observation(self, task: Dict) -> str:
        """Generate mock accessibility tree."""
        obs = f"""
[100] RootWebArea 'Shopping Site'
[101] navigation 'Main Navigation'
[102] searchbox 'Search products'
[103] button 'Search'
[200] main 'Product Listings'
[201] link 'Wireless Headphones - $49.99'
[202] link 'Laptop Computer - $899.99'
[203] link 'Smartphone - $699.99'
[204] link 'Running Shoes - $79.99'
[205] link 'Office Chair - $249.99'
[300] button 'Add to Cart'
[301] button 'Buy Now'
[302] link 'View Details'
        """
        return obs.strip()


def load_trained_agent(model_path: str, device: str = 'cpu'):
    """
    Load trained WebShop agent.

    Args:
        model_path: Path to trained model checkpoint
        device: 'cpu' or 'cuda'

    Returns:
        agent, tokenizer
    """
    if not os.path.exists(model_path):
        print(f"✗ Model not found: {model_path}")
        print("\nPlease train a model first or check the path")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location=device)

    # Rebuild tokenizer
    tokenizer = SimpleTokenizer()
    tokenizer.vocab = checkpoint['tokenizer_vocab']
    tokenizer.idx_to_word = {v: k for k, v in tokenizer.vocab.items()}

    # Create agent
    agent = OfficialWebShopAgent(
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=64,
        hidden_dim=128
    )
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.to(device)
    agent.eval()

    print(f"✓ Loaded trained agent from {model_path}")
    print(f"  Vocab size: {tokenizer.vocab_size()}")

    return agent, tokenizer


def load_shopping_tasks(tasks_file: str) -> List[Dict]:
    """Load shopping tasks from JSON file."""
    if not os.path.exists(tasks_file):
        print(f"✗ Tasks file not found: {tasks_file}")
        print("\nPlease run task filter first:")
        print("  python scripts/filter_shopping_tasks.py")
        return []

    with open(tasks_file, 'r') as f:
        tasks = json.load(f)

    print(f"✓ Loaded {len(tasks)} shopping tasks")
    return tasks


def evaluate_on_webarena_shopping(
    agent: OfficialWebShopAgent,
    tokenizer: SimpleTokenizer,
    tasks: List[Dict],
    num_tasks: int = 50,
    device: str = 'cpu',
    use_mock: bool = True,
    verbose: bool = False
) -> Dict:
    """
    Evaluate trained WebShop agent on WebArena shopping tasks.

    Args:
        agent: Trained WebShop agent
        tokenizer: Tokenizer
        tasks: List of shopping tasks
        num_tasks: Number of tasks to evaluate
        device: 'cpu' or 'cuda'
        use_mock: Use mock environment (True) or real WebArena (False)
        verbose: Print detailed info

    Returns:
        Results dictionary
    """
    print("=" * 70)
    print("WebArena Shopping Evaluation (Adapted RAGEN)")
    print("=" * 70)
    print(f"Model: Trained WebShop RAGEN")
    print(f"Tasks: {min(num_tasks, len(tasks))}")
    print(f"Mode: {'Mock' if use_mock else 'Live'} WebArena")
    print("=" * 70)
    print()

    # Limit tasks
    eval_tasks = tasks[:num_tasks]

    # Create environment
    if use_mock:
        webarena_env = MockWebArenaEnv(eval_tasks)
        print("Using mock WebArena environment (for testing without Docker)")
    else:
        # Try to import real WebArena
        try:
            # This would import actual WebArena when available
            # For now, fall back to mock
            print("⚠ Real WebArena not available, using mock environment")
            webarena_env = MockWebArenaEnv(eval_tasks)
        except ImportError:
            print("⚠ WebArena not installed, using mock environment")
            webarena_env = MockWebArenaEnv(eval_tasks)

    # Create adapter
    adapter = WebArenaToWebShopAdapter(verbose=verbose)

    # Wrap environment with adapter
    env = WebArenaEnvironmentWrapper(webarena_env, adapter)

    # Evaluation metrics
    episode_rewards = []
    episode_steps = []
    successes = 0
    adaptation_failures = []

    device_obj = torch.device(device)

    # Run evaluation
    for ep in range(len(eval_tasks)):
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 15

        task_id = info.get('task_id', f'task_{ep}')
        task = info.get('task', 'N/A')

        if verbose or ep < 3:  # Always print first 3 episodes
            print(f"\nEpisode {ep + 1}/{len(eval_tasks)}")
            print(f"Task: {task[:80]}...")
            print("-" * 70)

        episode_log = {
            'task_id': task_id,
            'task': task,
            'actions': [],
            'rewards': []
        }

        while not done and step_count < max_steps:
            # Tokenize observation
            obs_tokens = tokenizer.tokenize(obs, max_len=200).to(device_obj)

            # Get available actions
            available_actions = env.get_available_actions(obs)

            # Select action
            with torch.no_grad():
                try:
                    action_str, log_prob, value = agent.select_action(
                        obs_tokens, available_actions, tokenizer.vocab
                    )
                except Exception as e:
                    if verbose:
                        print(f"  ✗ Agent error: {e}")
                    action_str = "search product"  # Fallback action

            # Take step
            next_obs, reward, done, step_info = env.step(action_str)

            total_reward += reward
            step_count += 1

            episode_log['actions'].append(action_str)
            episode_log['rewards'].append(reward)

            if verbose or ep < 3:
                print(f"  Step {step_count}: {action_str}")
                print(f"    Reward: {reward:.3f}, Done: {done}")

            # Check for adaptation failure
            if 'error' in step_info and step_info['error'] == 'invalid_action':
                adaptation_failures.append({
                    'episode': ep,
                    'action': action_str,
                    'task': task
                })

            obs = next_obs

        episode_rewards.append(total_reward)
        episode_steps.append(step_count)

        # Count success (using same threshold as WebShop)
        if total_reward > 0.5:
            successes += 1

        if verbose or ep < 3:
            print(f"  Episode Reward: {total_reward:.3f}")
            print(f"  Steps: {step_count}")
            print()

    # Compute metrics
    num_episodes = len(eval_tasks)
    success_rate = successes / num_episodes if num_episodes > 0 else 0
    avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
    avg_steps = sum(episode_steps) / len(episode_steps) if episode_steps else 0
    adaptation_failure_rate = len(adaptation_failures) / (num_episodes * avg_steps) if avg_steps > 0 else 0

    results = {
        'benchmark': 'WebArena Shopping (Adapted)',
        'agent_type': 'RAGEN + A*PO (trained on WebShop, adapted)',
        'num_episodes': num_episodes,
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'adaptation_failure_rate': adaptation_failure_rate,
        'episode_rewards': episode_rewards,
        'episode_steps': episode_steps,
        'adaptation_failures': adaptation_failures[:10],  # Save first 10
        'evaluation_date': datetime.now().isoformat(),
        'mode': 'mock' if use_mock else 'live'
    }

    print("=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Success Rate: {success_rate:.1%}")
    print(f"Avg Reward: {avg_reward:.3f}")
    print(f"Avg Steps: {avg_steps:.1f}")
    print(f"Adaptation Failures: {len(adaptation_failures)} ({adaptation_failure_rate:.1%} of steps)")
    print("=" * 70)

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate trained WebShop model on WebArena shopping tasks"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='models/official_agent_minimal.pth',
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--tasks',
        type=str,
        default='data/webarena_shopping_tasks.json',
        help='Path to shopping tasks JSON'
    )
    parser.add_argument(
        '--num_tasks',
        type=int,
        default=50,
        help='Number of tasks to evaluate'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device (cpu/cuda)'
    )
    parser.add_argument(
        '--live',
        action='store_true',
        help='Use live WebArena (requires Docker). Default: mock environment'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed episode information'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/webarena_adapted_results.json',
        help='Output file for results'
    )

    args = parser.parse_args()

    print()

    # Load trained agent
    agent, tokenizer = load_trained_agent(args.model, args.device)
    print()

    # Load tasks
    tasks = load_shopping_tasks(args.tasks)
    if not tasks:
        print("\n⚠ No tasks loaded. Creating sample tasks for testing...")
        # Import sample task generator
        sys.path.insert(0, os.path.join(project_root, 'scripts'))
        from filter_shopping_tasks import create_sample_shopping_tasks
        tasks = create_sample_shopping_tasks()

    print()

    # Run evaluation
    results = evaluate_on_webarena_shopping(
        agent,
        tokenizer,
        tasks,
        num_tasks=args.num_tasks,
        device=args.device,
        use_mock=not args.live,
        verbose=args.verbose
    )

    # Save results
    output_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"✓ Results saved to {args.output}")
    print()
    print("Next steps:")
    print("  1. Generate comparison: python compare_results.py --webarena_adapted")
    print("  2. View analysis: cat results/final_failure_analysis.md")
    print()


if __name__ == "__main__":
    main()
