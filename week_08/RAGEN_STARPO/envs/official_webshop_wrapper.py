"""
Wrapper for Official WebShop Environment with Action Space Handler
Configured for minimal testing (100 products)
"""
import sys
import os
import re
from typing import List, Tuple, Dict

# Add WebShop to path
webshop_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'WebShop')
if webshop_path not in sys.path:
    sys.path.insert(0, webshop_path)

# Delayed import - will happen in __init__ to allow graceful failure
WebAgentTextEnv = None


class OfficialWebShopActionSpace:
    """
    Dynamic action space handler for official WebShop.
    Actions are TEXT-BASED, not fixed indices.
    """

    def __init__(self):
        # Common search keywords for initialization
        self.common_searches = [
            "shoes", "red", "blue", "black", "white",
            "laptop", "headphones", "watch", "phone",
            "shirt", "pants", "jacket", "dress",
            "cheap", "best", "premium", "new"
        ]

    def parse_available_actions(self, observation: str) -> List[str]:
        """
        Parse observation to extract available actions.
        Returns list of action strings like ['search', 'click[b000...']
        """
        actions = ['search']  # Always can search

        # Extract clickable buttons using [SEP] delimiter
        if '[SEP]' in observation:
            parts = observation.split('[SEP]')
            for part in parts:
                part = part.strip()
                # Look for button text or product IDs
                if part.startswith('B0') and len(part) == 10:  # ASIN format
                    actions.append(f'click[{part.lower()}]')
                elif any(keyword in part.lower() for keyword in ['buy now', 'next', 'prev', 'back']):
                    actions.append(f'click[{part.lower().strip()}]')

        return actions

    def sample_search_query(self, instruction: str) -> str:
        """
        Generate search query from instruction text.
        Extract key nouns/adjectives from instruction.
        """
        # Simple keyword extraction - take important words
        words = instruction.lower().split()

        # Filter out common words
        stop_words = {'i', 'am', 'looking', 'for', 'a', 'an', 'the', 'to', 'find', 'buy', 'want'}
        keywords = [w for w in words if w not in stop_words and len(w) > 2][:3]

        if keywords:
            return ' '.join(keywords)
        return 'shoes'  # Fallback


class OfficialWebShopWrapper:
    """
    Wrapper around official WebAgentTextEnv for minimal testing.
    Configured for 100 products, text observation mode.
    """

    def __init__(self, num_products: int = 100):
        """
        Initialize wrapper to use YOUR local data files.
        """
        self.num_products = num_products
        self.action_space = OfficialWebShopActionSpace()
        self.step_count = 0
        self.max_steps = 15

        # Try to import and initialize environment
        self.env = None
        try:
            import os
            
            # Try multiple data locations
            possible_paths = [
                '/root/WebShop/data',
                '/root/code/data',
                '/root/code/WebShop/data',
                os.path.join(os.getcwd(), 'data'),
            ]
            
            webshop_data = None
            for path in possible_paths:
                if os.path.exists(path) and os.listdir(path):
                    webshop_data = path
                    print(f"✓ Found data at: {path}")
                    break
            
            if not webshop_data:
                raise RuntimeError("Could not find data directory in any expected location")
            
            os.environ['DATA_DIR'] = webshop_data
            
            # Print what data files we found
            files = os.listdir(webshop_data)
            print(f"✓ Data files available: {files[:5]}...")  # Show first 5
            
            # Import WebAgentTextEnv
            from web_agent_site.envs import WebAgentTextEnv as WebEnv

            # Initialize environment with YOUR data
            self.env = WebEnv(
                observation_mode='text',
                num_products=num_products,
                human_goals=True  # Use the instructions from your JSON files
            )
            print(f"✓ Initialized Official WebShop with {num_products} products")
            
        except Exception as e:
            print(f"✗ Error initializing WebShop: {e}")
            import traceback
            traceback.print_exc()
            print("\n" + "=" * 60)
            print("FAILED TO LOAD YOUR DATA FILES")
            print("=" * 60)
            print("Check that your data/ folder contains:")
            print("  - items_shuffle_1000.json OR")
            print("  - items_ins_v2_1000.json")
            print("=" * 60)

  
    def reset(self) -> Tuple[str, Dict]:
        """Reset environment and return initial observation."""
        self.step_count = 0

        if self.env is None:
            # Return mock observation if env failed to initialize
            return self._get_mock_observation(), {}

        try:
            obs, info = self.env.reset()

            # Extract instruction text
            instruction = self.env.get_instruction_text() if hasattr(self.env, 'get_instruction_text') else ""

            # Format observation with instruction
            formatted_obs = f"Instruction: {instruction}\n{obs}"

            return formatted_obs, {'instruction': instruction}
        except Exception as e:
            print(f"Reset error: {e}")
            return self._get_mock_observation(), {}

    def step(self, action: str) -> Tuple[str, float, bool, Dict]:
        """
        Take action in environment.

        Args:
            action: Text action like 'search red shoes' or 'click[b00abc123]'

        Returns:
            observation, reward, done, info
        """
        self.step_count += 1

        if self.env is None:
            # Mock step for testing without data
            return self._get_mock_observation(), 0.0, True, {}

        try:
            obs, reward, done, info = self.env.step(action)

            # Enforce max steps
            if self.step_count >= self.max_steps:
                done = True
                reward -= 0.1  # Small penalty for timeout

            return obs, reward, done, info or {}
        except Exception as e:
            print(f"Step error with action '{action}': {e}")
            return f"Error: {e}", 0.0, True, {}

    def get_available_actions(self, observation: str) -> List[str]:
        """Get list of available actions from current observation."""
        return self.action_space.parse_available_actions(observation)

    def _get_mock_observation(self) -> str:
        """Return mock observation for testing without data."""
        return (
            "Instruction: Find a red running shoes\n"
            "[SEP] WebShop [SEP] "
            "[SEP] search [SEP]"
        )


def test_official_wrapper():
    """Test function to verify wrapper works."""
    print("=" * 60)
    print("Testing Official WebShop Wrapper (Minimal Config)")
    print("=" * 60)

    # Create wrapper with 100 products
    wrapper = OfficialWebShopWrapper(num_products=100)

    # Test reset
    print("\n1. Testing reset...")
    obs, info = wrapper.reset()
    print(f"Observation: {obs[:200]}...")
    print(f"Info: {info}")

    # Test action parsing
    print("\n2. Testing action parsing...")
    actions = wrapper.get_available_actions(obs)
    print(f"Available actions: {actions[:5]}")

    # Test search action
    print("\n3. Testing search action...")
    instruction = info.get('instruction', 'red shoes')
    search_query = wrapper.action_space.sample_search_query(instruction)
    print(f"Search query: {search_query}")

    obs, reward, done, info = wrapper.step(f'search[{search_query}]')
    print(f"Reward: {reward}, Done: {done}")
    print(f"New observation: {obs[:200]}...")

    print("\n" + "=" * 60)
    print("Wrapper test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_official_wrapper()
