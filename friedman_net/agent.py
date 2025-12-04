"""
EconomicAgent: Neural network agent that competes economically in the market.
"""

from typing import Tuple
import torch
import torch.nn as nn


class EconomicAgent(nn.Module):
    """
    A neural network agent that competes economically.

    Architecture:
        Input(784) -> Linear(hidden_dim) -> ReLU -> Linear(10) [classification]
                                                 -> Linear(1)  [bid score]

    Economic State:
        - wallet: Current balance (revenue - rent)
        - age: Number of generations survived
        - hidden_dim: Model complexity (evolutionary parameter)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        starting_wallet: float,
        min_hidden_dim: int = 4,
        max_hidden_dim: int = 2048
    ):
        super().__init__()

        # Ensure hidden_dim is within bounds
        hidden_dim = max(min_hidden_dim, min(hidden_dim, max_hidden_dim))

        # Architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Classification head
        self.bid_head = nn.Linear(hidden_dim, 1)      # Bid score head

        # Economic state
        self.hidden_dim = hidden_dim
        self.wallet = starting_wallet
        self.initial_wallet = starting_wallet
        self.age = 0
        self.revenue_earned = 0.0  # Track total revenue for this generation

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dual outputs.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            output_logits: Classification logits [batch_size, output_dim]
            bid_scores: Confidence/willingness to pay [batch_size, 1]
        """
        hidden = self.relu(self.fc1(x))
        output_logits = self.fc2(hidden)
        bid_scores = self.bid_head(hidden)
        return output_logits, bid_scores

    def reset_generation_stats(self):
        """Reset per-generation statistics."""
        self.initial_wallet = self.wallet
        self.revenue_earned = 0.0
