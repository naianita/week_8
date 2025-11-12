"""
Agent Architecture for Official WebShop
Handles text-based observations and dynamic action generation
"""
import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import List, Tuple, Dict
import re


class WebShopTextEncoder(nn.Module):
    """
    Encodes text observations into fixed-size embeddings.
    Uses simple embedding + LSTM architecture.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: [batch_size, seq_len]
        Returns:
            encoded: [batch_size, hidden_dim]
        """
        embedded = self.embedding(token_ids)  # [B, S, E]
        lstm_out, (hidden, _) = self.lstm(embedded)  # hidden: [2, B, H]
        # Use last layer's hidden state
        encoded = self.layer_norm(hidden[-1])  # [B, H]
        return encoded


class ActionGenerator(nn.Module):
    """
    Generates actions from encoded observations.
    Supports two action types: search query generation and click selection.
    """

    def __init__(self, hidden_dim: int = 256, vocab_size: int = 5000):
        super().__init__()

        # Action type classifier (search vs click)
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # [search, click]
        )

        # Search query generator - simple keyword selection
        self.search_query_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, vocab_size)  # Score for each word in vocab
        )

        # Click target selector - scores clickable elements
        self.click_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, encoded: torch.Tensor, mode: str = 'type'):
        """
        Args:
            encoded: [batch_size, hidden_dim]
            mode: 'type' | 'search' | 'click'
        """
        if mode == 'type':
            return self.action_type_head(encoded)
        elif mode == 'search':
            return self.search_query_head(encoded)
        elif mode == 'click':
            return self.click_scorer(encoded)
        else:
            raise ValueError(f"Unknown mode: {mode}")


class OfficialWebShopAgent(nn.Module):
    """
    Complete agent for Official WebShop.
    Architecture:
    1. Text Encoder: Obs -> Embedding
    2. Action Generator: Embedding -> Action
    3. Value Head: Embedding -> Value estimate
    """

    def __init__(self, vocab_size: int = 5000, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()

        self.encoder = WebShopTextEncoder(vocab_size, embedding_dim, hidden_dim)
        self.action_generator = ActionGenerator(hidden_dim, vocab_size)

        # Value head for RL
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.hidden_dim = hidden_dim

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            token_ids: [batch_size, seq_len]

        Returns:
            action_type_logits: [batch_size, 2]
            value: [batch_size, 1]
        """
        encoded = self.encoder(token_ids)
        action_type_logits = self.action_generator(encoded, mode='type')
        value = self.value_head(encoded)
        return action_type_logits, value

    def select_action(self, token_ids: torch.Tensor, available_actions: List[str],
                      vocab: Dict[str, int]) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Select action given observation and available actions.

        Args:
            token_ids: [1, seq_len]
            available_actions: List of action strings
            vocab: Word to index mapping

        Returns:
            action_str: Selected action
            log_prob: Log probability of action
            value: Value estimate
        """
        with torch.no_grad():
            encoded = self.encoder(token_ids)
            action_type_logits = self.action_generator(encoded, mode='type')
            value = self.value_head(encoded)

            # Sample action type
            action_type_dist = Categorical(logits=action_type_logits)
            action_type = action_type_dist.sample()  # 0: search, 1: click

            if action_type == 0 or not any('click' in a for a in available_actions):
                # Generate search query
                search_logits = self.action_generator(encoded, mode='search')
                # Sample top 3 words
                top_k = torch.topk(search_logits[0], k=3)
                word_indices = top_k.indices.tolist()

                # Convert indices to words
                idx_to_word = {v: k for k, v in vocab.items()}
                keywords = [idx_to_word.get(idx, 'shoes') for idx in word_indices]
                action_str = f"search[{' '.join(keywords[:2])}]"
                log_prob = action_type_dist.log_prob(action_type)

            else:
                # Select from available click actions
                click_actions = [a for a in available_actions if 'click' in a]
                if not click_actions:
                    action_str = 'search[shoes]'
                else:
                    # Simple: pick first click action (can be improved)
                    action_str = click_actions[0]
                log_prob = action_type_dist.log_prob(action_type)

        return action_str, log_prob, value.squeeze()


class SimpleTokenizer:
    """
    Simple tokenizer for WebShop text observations.
    Builds vocabulary from observations and tokenizes text.
    """

    def __init__(self, max_vocab_size: int = 5000):
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '[SEP]': 2}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>', 2: '[SEP]'}
        self.max_vocab_size = max_vocab_size
        self.next_idx = 3

    def build_vocab(self, texts: List[str]):
        """Build vocabulary from list of texts."""
        word_counts = {}

        for text in texts:
            # Simple word extraction
            words = re.findall(r'\b\w+\b', text.lower())
            for word in words:
                if word not in self.vocab:
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Add most common words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_words[:self.max_vocab_size - 3]:
            if word not in self.vocab:
                self.vocab[word] = self.next_idx
                self.idx_to_word[self.next_idx] = word
                self.next_idx += 1

    def tokenize(self, text: str, max_len: int = 200) -> torch.Tensor:
        """
        Tokenize text to tensor.

        Args:
            text: Input text
            max_len: Maximum sequence length

        Returns:
            token_ids: [1, max_len]
        """
        words = re.findall(r'\b\w+\b', text.lower())
        token_ids = [self.vocab.get(word, self.vocab['<UNK>']) for word in words[:max_len]]

        # Pad or truncate
        if len(token_ids) < max_len:
            token_ids.extend([self.vocab['<PAD>']] * (max_len - len(token_ids)))
        else:
            token_ids = token_ids[:max_len]

        return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)

    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)


def test_agent():
    """Test agent architecture."""
    print("Testing Official WebShop Agent...")

    # Create tokenizer and build vocab
    tokenizer = SimpleTokenizer(max_vocab_size=5000)
    sample_texts = [
        "Instruction: Find red running shoes",
        "WebShop search results",
        "Click to buy now",
        "Price: $29.99"
    ]
    tokenizer.build_vocab(sample_texts)

    print(f"Vocab size: {tokenizer.vocab_size()}")

    # Create agent
    agent = OfficialWebShopAgent(
        vocab_size=tokenizer.vocab_size(),
        embedding_dim=128,
        hidden_dim=256
    )

    print(f"Agent parameters: {sum(p.numel() for p in agent.parameters()):,}")

    # Test forward pass
    obs = "Instruction: Find red shoes [SEP] WebShop [SEP] search"
    token_ids = tokenizer.tokenize(obs)

    action_logits, value = agent(token_ids)
    print(f"Action logits shape: {action_logits.shape}")
    print(f"Value shape: {value.shape}")

    # Test action selection
    available_actions = ['search', 'click[b0012345ab]', 'click[buy now]']
    action, log_prob, value = agent.select_action(token_ids, available_actions, tokenizer.vocab)
    print(f"Selected action: {action}")
    print(f"Log prob: {log_prob}")
    print(f"Value: {value}")

    print("\n✓ Agent test passed!")


if __name__ == "__main__":
    test_agent()
