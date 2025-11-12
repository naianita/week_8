"""
WebArena to WebShop Adapter
Translates between WebArena's accessibility tree format and WebShop's text format
"""
import re
from typing import Dict, List, Tuple, Optional


class WebArenaToWebShopAdapter:
    """
    Bidirectional adapter between WebArena and WebShop formats.

    WebArena uses:
    - Observations: Accessibility tree with element IDs
    - Actions: Browser commands like click[id], type[id][text][enter]

    WebShop uses:
    - Observations: Simple text descriptions
    - Actions: Text commands like "search X", "click Y", "buy Z"
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.element_map = {}  # Maps simple indices to WebArena element IDs
        self.reverse_map = {}  # Maps WebArena IDs to simple indices
        self.search_box_id = None
        self.last_observation = ""

    def adapt_observation(self, accessibility_tree: str, task_instruction: str = "") -> str:
        """
        Convert WebArena accessibility tree to WebShop-like text.

        Args:
            accessibility_tree: Raw accessibility tree from WebArena
            task_instruction: Original task description

        Returns:
            WebShop-style text observation
        """
        # Reset mappings for new observation
        self.element_map = {}
        self.reverse_map = {}
        self.search_box_id = None

        # Parse accessibility tree for interactive elements
        elements = self._parse_accessibility_tree(accessibility_tree)

        # Build WebShop-style observation
        observation = ""

        # Add instruction if provided
        if task_instruction:
            observation += f"Instruction: {task_instruction}\n\n"

        # Extract and format products
        products = self._extract_products(elements)
        if products:
            observation += "Products available:\n"
            for idx, (elem_id, product_name, price) in enumerate(products, start=1):
                self.element_map[idx] = elem_id
                self.reverse_map[elem_id] = idx

                price_str = f" - ${price}" if price else ""
                observation += f"  [{idx}] {product_name}{price_str}\n"

        # Extract available actions
        actions = self._extract_actions(elements)
        if actions:
            observation += "\nAvailable actions:\n"
            for action_type, elem_id, label in actions:
                if action_type == "button" and "cart" in label.lower():
                    idx = len(self.element_map) + 1
                    self.element_map[idx] = elem_id
                    self.reverse_map[elem_id] = idx
                    observation += f"  [{idx}] {label}\n"
                elif action_type in ["textbox", "searchbox"] or "search" in label.lower():
                    self.search_box_id = elem_id
                    observation += f"  [search] Search box available\n"

        # Fallback: if no structured data, use raw text
        if not observation.strip() or observation == (f"Instruction: {task_instruction}\n\n" if task_instruction else ""):
            observation += "\n" + self._extract_text_content(accessibility_tree)

        self.last_observation = observation

        if self.verbose:
            print(f"\n--- Adapted Observation ---")
            print(observation[:500])  # Print first 500 chars
            print(f"Element map: {self.element_map}")

        return observation.strip()

    def adapt_action(self, webshop_action: str) -> Optional[str]:
        """
        Convert WebShop action to WebArena browser command.

        Args:
            webshop_action: Action from WebShop agent (e.g., "search shoes", "click 2", "buy 1")

        Returns:
            WebArena browser command (e.g., "type[142][shoes][1]", "click[1582]")
            Returns None if action cannot be adapted
        """
        action = webshop_action.strip().lower()

        # Handle search actions
        if action.startswith("search"):
            search_query = action.replace("search", "").strip()
            if not search_query:
                return None

            if self.search_box_id:
                # type[id][text][enter_flag] - enter_flag=1 means press Enter
                webarena_action = f"type[{self.search_box_id}][{search_query}][1]"
                if self.verbose:
                    print(f"Adapted: '{webshop_action}' -> '{webarena_action}'")
                return webarena_action
            else:
                # No search box found, try goto with search URL
                # This is a fallback - may not work for all sites
                return None

        # Handle click actions
        elif action.startswith("click"):
            try:
                # Extract index number
                match = re.search(r'click\s+(\d+)', action)
                if match:
                    idx = int(match.group(1))
                    if idx in self.element_map:
                        elem_id = self.element_map[idx]
                        webarena_action = f"click[{elem_id}]"
                        if self.verbose:
                            print(f"Adapted: '{webshop_action}' -> '{webarena_action}'")
                        return webarena_action
            except (ValueError, KeyError):
                pass
            return None

        # Handle buy actions (map to "Add to Cart" button click)
        elif action.startswith("buy"):
            try:
                # Extract index number
                match = re.search(r'buy\s+(\d+)', action)
                if match:
                    idx = int(match.group(1))
                    # First click on the product
                    if idx in self.element_map:
                        elem_id = self.element_map[idx]
                        webarena_action = f"click[{elem_id}]"
                        if self.verbose:
                            print(f"Adapted: '{webshop_action}' -> '{webarena_action}' (buy as click)")
                        return webarena_action
            except (ValueError, KeyError):
                pass
            return None

        # Unknown action
        if self.verbose:
            print(f"Cannot adapt action: '{webshop_action}'")
        return None

    def _parse_accessibility_tree(self, tree: str) -> List[Tuple[int, str, str]]:
        """
        Parse accessibility tree to extract elements.

        Returns:
            List of (element_id, element_type, element_text) tuples
        """
        elements = []

        # Pattern: [ID] type 'text' or [ID] type "text"
        # Example: [1582] button 'Add to Cart'
        pattern = r'\[(\d+)\]\s+(\w+)\s+["\']([^"\']+)["\']'

        for match in re.finditer(pattern, tree):
            elem_id = int(match.group(1))
            elem_type = match.group(2).lower()
            elem_text = match.group(3)
            elements.append((elem_id, elem_type, elem_text))

        return elements

    def _extract_products(self, elements: List[Tuple[int, str, str]]) -> List[Tuple[int, str, Optional[str]]]:
        """
        Extract product listings from elements.

        Returns:
            List of (element_id, product_name, price) tuples
        """
        products = []

        # Look for links that appear to be products
        product_keywords = ['product', 'item', 'buy', 'shop']
        price_pattern = r'\$[\d,]+\.?\d*'

        for elem_id, elem_type, elem_text in elements:
            if elem_type == 'link':
                # Check if this looks like a product link
                if any(kw in elem_text.lower() for kw in product_keywords) or len(elem_text) > 10:
                    # Try to find associated price
                    price = None
                    # Look for price in subsequent elements (simple heuristic)
                    for _, next_type, next_text in elements:
                        if next_type == 'statictext':
                            price_match = re.search(price_pattern, next_text)
                            if price_match:
                                price = price_match.group(0).replace('$', '')
                                break

                    products.append((elem_id, elem_text, price))

        return products[:20]  # Limit to 20 products like WebShop

    def _extract_actions(self, elements: List[Tuple[int, str, str]]) -> List[Tuple[str, int, str]]:
        """
        Extract available actions (buttons, textboxes, etc.)

        Returns:
            List of (action_type, element_id, label) tuples
        """
        actions = []

        for elem_id, elem_type, elem_text in elements:
            if elem_type in ['button', 'textbox', 'searchbox']:
                actions.append((elem_type, elem_id, elem_text))

        return actions

    def _extract_text_content(self, tree: str) -> str:
        """
        Extract plain text content from accessibility tree as fallback.
        """
        # Remove element IDs and tags
        text = re.sub(r'\[\d+\]\s+\w+\s+', '', tree)
        # Remove quotes
        text = text.replace('"', '').replace("'", '')
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()[:500]  # Limit length


class WebArenaEnvironmentWrapper:
    """
    Wrapper for WebArena environment that uses the adapter.
    Makes WebArena look like WebShop to the agent.
    """

    def __init__(self, webarena_env, adapter: WebArenaToWebShopAdapter):
        self.env = webarena_env
        self.adapter = adapter
        self.current_task = None

    def reset(self):
        """Reset environment and return adapted observation."""
        obs, info = self.env.reset()
        self.current_task = info.get('task', '')

        # Adapt observation
        adapted_obs = self.adapter.adapt_observation(obs, self.current_task)

        return adapted_obs, info

    def step(self, webshop_action: str):
        """
        Execute WebShop action in WebArena environment.

        Args:
            webshop_action: Action from WebShop agent

        Returns:
            (adapted_observation, reward, done, info)
        """
        # Adapt action
        webarena_action = self.adapter.adapt_action(webshop_action)

        if webarena_action is None:
            # Action couldn't be adapted - return penalty
            return self.adapter.last_observation, -0.1, False, {'error': 'invalid_action'}

        # Execute in WebArena
        obs, reward, done, info = self.env.step(webarena_action)

        # Adapt observation
        adapted_obs = self.adapter.adapt_observation(obs, self.current_task)

        return adapted_obs, reward, done, info

    def get_available_actions(self, obs: str) -> List[str]:
        """
        Generate list of available actions in WebShop format.
        This is needed by the WebShop agent.
        """
        actions = []

        # Always allow search
        if self.adapter.search_box_id:
            actions.extend([
                "search shoes", "search laptop", "search phone",
                "search shirt", "search watch", "search bag"
            ])

        # Add click actions for mapped elements
        for idx in sorted(self.adapter.element_map.keys()):
            actions.append(f"click {idx}")
            actions.append(f"buy {idx}")

        # Limit to reasonable number
        return actions[:50]
