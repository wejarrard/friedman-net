"""
MarketLayer: Container for competing agents with Top-K gating mechanism.
"""

from typing import List, Tuple
import torch
import torch.nn as nn

from .agent import EconomicAgent


class MarketLayer(nn.Module):
    """
    Container for competing agents with Top-K gating mechanism.

    Forward pass:
        1. Collect bids from all agents
        2. Select Top-K highest bidders per sample
        3. Average outputs from winners
        4. Return (final_output, winner_indices) for revenue tracking

    Economic State:
        - market_wallet: Treasury that receives rent and pays out rewards
    """

    def __init__(self, agents: List[EconomicAgent], top_k: int, initial_market_wallet: float = 0.0):
        super().__init__()
        self.agents = nn.ModuleList(agents)
        self.top_k = top_k
        self.market_wallet = initial_market_wallet

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[List[int]]]:
        """
        Forward pass with Top-K gating.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            final_outputs: Ensemble predictions [batch_size, output_dim]
            winner_indices: List of winner agent indices per sample
        """
        batch_size = x.size(0)
        num_agents = len(self.agents)

        if num_agents == 0:
            raise ValueError("MarketLayer has no agents!")

        # Collect outputs and bids from all agents
        all_outputs = []
        all_bids = []

        for agent in self.agents:
            output, bid = agent(x)
            all_outputs.append(output)  # [batch_size, output_dim]
            all_bids.append(bid.squeeze(-1))  # [batch_size]

        # Stack: [num_agents, batch_size, output_dim] and [num_agents, batch_size]
        all_outputs = torch.stack(all_outputs)
        all_bids = torch.stack(all_bids)

        # Per-sample Top-K selection
        actual_k = min(self.top_k, num_agents)
        final_outputs = []
        winner_indices = []

        for i in range(batch_size):
            sample_bids = all_bids[:, i]  # [num_agents]

            # Get Top-K agent indices
            top_k_values, top_k_indices = torch.topk(sample_bids, actual_k)
            winner_indices.append(top_k_indices.tolist())

            # Average outputs from Top-K agents
            sample_outputs = all_outputs[top_k_indices, i, :]  # [actual_k, output_dim]
            sample_final = sample_outputs.mean(dim=0)  # [output_dim]
            final_outputs.append(sample_final)

        final_outputs = torch.stack(final_outputs)  # [batch_size, output_dim]

        return final_outputs, winner_indices
