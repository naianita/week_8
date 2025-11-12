"""
Filter Shopping Tasks from WebArena
Extracts e-commerce tasks for RAGEN evaluation
"""
import os
import sys
import json
from typing import List, Dict

# Add project to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Shopping-related keywords to identify e-commerce tasks
SHOPPING_KEYWORDS = [
    'buy', 'purchase', 'shop', 'product', 'item', 'price', 'cart', 'checkout',
    'order', 'add to cart', 'search for', 'find', 'compare', 'available',
    'stock', 'shipping', 'delivery', 'payment', 'discount', 'sale',
    'category', 'filter', 'sort', 'review', 'rating', 'customer'
]

# E-commerce website identifiers
ECOMMERCE_SITES = [
    'shopping', 'onestopshop', 'shop.', 'store', 'amazon', 'ebay',
    'localhost:7770'  # WebArena's OneStopShop default port
]


def is_shopping_task(task: Dict) -> bool:
    """
    Determine if a task is related to e-commerce/shopping.

    Args:
        task: Task dictionary with 'intent', 'sites', 'require_login', etc.

    Returns:
        True if task is shopping-related
    """
    # Check task intent/instruction
    intent = task.get('intent', '').lower()
    if any(keyword in intent for keyword in SHOPPING_KEYWORDS):
        return True

    # Check sites used
    sites = task.get('sites', [])
    if isinstance(sites, list):
        for site in sites:
            if any(ecomm in site.lower() for ecomm in ECOMMERCE_SITES):
                return True

    # Check start_url
    start_url = task.get('start_url', '').lower()
    if any(ecomm in start_url for ecomm in ECOMMERCE_SITES):
        return True

    return False


def load_webarena_tasks(config_dir: str) -> List[Dict]:
    """
    Load tasks from WebArena config files.

    Args:
        config_dir: Path to WebArena config_files directory

    Returns:
        List of task dictionaries
    """
    tasks = []

    if not os.path.exists(config_dir):
        print(f"✗ Config directory not found: {config_dir}")
        print("\nPlease ensure WebArena is set up:")
        print("  bash setup_webarena.sh")
        return tasks

    # WebArena tasks are typically in JSON files
    for filename in os.listdir(config_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(config_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)

                    # Handle different formats
                    if isinstance(data, list):
                        tasks.extend(data)
                    elif isinstance(data, dict):
                        if 'tasks' in data:
                            tasks.extend(data['tasks'])
                        else:
                            tasks.append(data)

            except json.JSONDecodeError:
                print(f"⚠ Skipping invalid JSON: {filename}")
                continue
            except Exception as e:
                print(f"⚠ Error loading {filename}: {e}")
                continue

    return tasks


def filter_and_save_shopping_tasks(
    webarena_dir: str,
    output_file: str = 'data/webarena_shopping_tasks.json',
    max_tasks: int = 100
):
    """
    Filter shopping tasks from WebArena and save to file.

    Args:
        webarena_dir: Path to WebArena directory
        output_file: Output JSON file path
        max_tasks: Maximum number of tasks to include
    """
    print("=" * 70)
    print("WebArena Shopping Task Filter")
    print("=" * 70)
    print()

    # Look for config files
    config_dir = os.path.join(webarena_dir, 'config_files')
    if not os.path.exists(config_dir):
        # Try alternative locations
        config_dir = os.path.join(webarena_dir, 'configs')

    if not os.path.exists(config_dir):
        print("✗ Cannot find config_files directory in WebArena")
        print("\nTrying to create sample shopping tasks from common patterns...")

        # Create sample tasks if WebArena tasks aren't available
        shopping_tasks = create_sample_shopping_tasks()
    else:
        # Load all tasks
        print(f"Loading tasks from: {config_dir}")
        all_tasks = load_webarena_tasks(config_dir)
        print(f"✓ Loaded {len(all_tasks)} total tasks")

        # Filter shopping tasks
        shopping_tasks = [task for task in all_tasks if is_shopping_task(task)]
        print(f"✓ Found {len(shopping_tasks)} shopping tasks")

    # Limit number of tasks
    if len(shopping_tasks) > max_tasks:
        shopping_tasks = shopping_tasks[:max_tasks]
        print(f"✓ Limited to {max_tasks} tasks")

    # Save to file
    output_path = os.path.join(project_root, output_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(shopping_tasks, f, indent=2)

    print()
    print(f"✓ Saved shopping tasks to: {output_file}")
    print()

    # Print sample tasks
    if shopping_tasks:
        print("Sample tasks:")
        for i, task in enumerate(shopping_tasks[:3], 1):
            intent = task.get('intent', task.get('instruction', 'N/A'))
            print(f"  {i}. {intent[:80]}...")

    print()
    print("=" * 70)


def create_sample_shopping_tasks() -> List[Dict]:
    """
    Create sample shopping tasks for testing when WebArena tasks aren't available.

    Returns:
        List of sample task dictionaries
    """
    print("Creating sample shopping tasks for testing...")

    sample_tasks = [
        {
            "task_id": "shopping_001",
            "intent": "Search for wireless headphones and find the cheapest option",
            "sites": ["shopping"],
            "start_url": "http://localhost:7770",
            "require_login": False,
            "eval_types": ["string_match"]
        },
        {
            "task_id": "shopping_002",
            "intent": "Find the price of HP Inkjet Fax Machine",
            "sites": ["shopping"],
            "start_url": "http://localhost:7770",
            "require_login": False,
            "eval_types": ["string_match"]
        },
        {
            "task_id": "shopping_003",
            "intent": "Add the highest rated laptop to your cart",
            "sites": ["shopping"],
            "start_url": "http://localhost:7770",
            "require_login": False,
            "eval_types": ["url_match"]
        },
        {
            "task_id": "shopping_004",
            "intent": "Compare prices of two different smartphones",
            "sites": ["shopping"],
            "start_url": "http://localhost:7770",
            "require_login": False,
            "eval_types": ["string_match"]
        },
        {
            "task_id": "shopping_005",
            "intent": "Search for running shoes under $100",
            "sites": ["shopping"],
            "start_url": "http://localhost:7770",
            "require_login": False,
            "eval_types": ["string_match"]
        },
        {
            "task_id": "shopping_006",
            "intent": "Find products in the Electronics category",
            "sites": ["shopping"],
            "start_url": "http://localhost:7770/categories",
            "require_login": False,
            "eval_types": ["url_match"]
        },
        {
            "task_id": "shopping_007",
            "intent": "Check if a specific product is in stock",
            "sites": ["shopping"],
            "start_url": "http://localhost:7770",
            "require_login": False,
            "eval_types": ["string_match"]
        },
        {
            "task_id": "shopping_008",
            "intent": "Sort products by price from low to high",
            "sites": ["shopping"],
            "start_url": "http://localhost:7770/search",
            "require_login": False,
            "eval_types": ["url_match"]
        },
        {
            "task_id": "shopping_009",
            "intent": "Read customer reviews for a product",
            "sites": ["shopping"],
            "start_url": "http://localhost:7770",
            "require_login": False,
            "eval_types": ["string_match"]
        },
        {
            "task_id": "shopping_010",
            "intent": "Find all products with 5-star ratings",
            "sites": ["shopping"],
            "start_url": "http://localhost:7770",
            "require_login": False,
            "eval_types": ["string_match"]
        }
    ]

    print(f"✓ Created {len(sample_tasks)} sample tasks")
    return sample_tasks


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter shopping tasks from WebArena"
    )
    parser.add_argument(
        '--webarena_dir',
        type=str,
        default='WebArena',
        help='Path to WebArena directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/webarena_shopping_tasks.json',
        help='Output file for filtered tasks'
    )
    parser.add_argument(
        '--max_tasks',
        type=int,
        default=100,
        help='Maximum number of tasks to include'
    )

    args = parser.parse_args()

    webarena_path = os.path.join(project_root, args.webarena_dir)

    filter_and_save_shopping_tasks(
        webarena_path,
        args.output,
        args.max_tasks
    )


if __name__ == "__main__":
    main()
