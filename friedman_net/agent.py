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
        uid: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        initial_cash_reserves: float,
        min_hidden_dim: int = 4,
        max_hidden_dim: int = 2048,
        bid_floor: float = 0.01
    ):
        super().__init__()

        # Identity
        self.uid = uid

        # Ensure hidden_dim is within bounds
        hidden_dim = max(min_hidden_dim, min(hidden_dim, max_hidden_dim))

        # Architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # Classification head
        self.bid_head = nn.Linear(hidden_dim, 1)      # Bid score head

        # Economic state
        self.hidden_dim = hidden_dim
        self.wallet = initial_cash_reserves
        self.initial_wallet = initial_cash_reserves
        self.age = 0
        self.revenue_earned = 0.0  # Track total revenue for this generation
        self.bid_floor = bid_floor  # Minimum bid to prevent bidding collapse

        # Bidding statistics (per generation)
        self.num_bids = 0  # Total bids made
        self.num_wins = 0  # Successful bids (in top-k)
        self.num_losses = 0  # Failed bids (not in top-k)

    def forward(self, x: torch.Tensor, expected_revenue: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dual outputs.

        Args:
            x: Input tensor [batch_size, input_dim]
            expected_revenue: Expected payout per prediction (for strategic bidding)

        Returns:
            output_logits: Classification logits [batch_size, output_dim]
            bid_scores: Confidence/willingness to pay [batch_size, 1]
        """
        hidden = self.relu(self.fc1(x))
        output_logits = self.fc2(hidden)

        # Base bid from network + floor
        base_bid = self.relu(self.bid_head(hidden)) + self.bid_floor

        # Scale bids by expected revenue (agents bid more when payouts are higher)
        # This allows strategic bidding and future problem-specific rewards
        if expected_revenue > 0:
            bid_scores = base_bid * expected_revenue
        else:
            bid_scores = base_bid

        return output_logits, bid_scores

    def reset_generation_stats(self):
        """Reset per-generation statistics."""
        self.initial_wallet = self.wallet
        self.revenue_earned = 0.0
        self.num_bids = 0
        self.num_wins = 0
        self.num_losses = 0
