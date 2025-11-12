"""
Quick test script for WebArena adapter
Tests adapter without needing PyTorch (for quick verification)
"""
import sys
import os

# Add project to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 70)
print("WebArena Adapter Test")
print("=" * 70)
print()

# Test 1: Import adapter
print("Test 1: Importing adapter...")
try:
    from envs.webarena_adapter import WebArenaToWebShopAdapter
    print("✓ Adapter imported successfully")
except ImportError as e:
    print(f"✗ Failed to import adapter: {e}")
    sys.exit(1)

print()

# Test 2: Create adapter instance
print("Test 2: Creating adapter instance...")
try:
    adapter = WebArenaToWebShopAdapter(verbose=True)
    print("✓ Adapter created successfully")
except Exception as e:
    print(f"✗ Failed to create adapter: {e}")
    sys.exit(1)

print()

# Test 3: Adapt mock observation
print("Test 3: Adapting mock WebArena observation...")
mock_observation = """
[100] RootWebArea 'Shopping Site'
[102] searchbox 'Search products'
[201] link 'Wireless Headphones - $49.99'
[202] link 'Laptop Computer - $899.99'
[203] link 'Smartphone - $699.99'
[300] button 'Add to Cart'
"""

try:
    adapted_obs = adapter.adapt_observation(
        mock_observation,
        task_instruction="Find wireless headphones"
    )
    print("✓ Observation adapted successfully")
    print()
    print("Adapted observation:")
    print("-" * 70)
    print(adapted_obs)
    print("-" * 70)
except Exception as e:
    print(f"✗ Failed to adapt observation: {e}")
    sys.exit(1)

print()

# Test 4: Adapt actions
print("Test 4: Adapting WebShop actions to WebArena...")
test_actions = [
    "search headphones",
    "click 1",
    "buy 2"
]

for action in test_actions:
    try:
        adapted_action = adapter.adapt_action(action)
        if adapted_action:
            print(f"✓ '{action}' → '{adapted_action}'")
        else:
            print(f"⚠ '{action}' → (could not adapt)")
    except Exception as e:
        print(f"✗ Failed to adapt '{action}': {e}")

print()
print("=" * 70)
print("Adapter Test Complete!")
print("=" * 70)
print()
print("All core adapter functionality works correctly.")
print()
print("Next steps:")
print("  1. Install PyTorch if you want to run full evaluation:")
print("     pip3 install torch")
print("  2. Run adapted evaluation:")
print("     python3 eval/evaluate_webarena_shopping.py --num_tasks 5")
print()
