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
        - treasury: Treasury that receives rent and pays out rewards
    """

    def __init__(self, agents: List[EconomicAgent], top_k: int, initial_treasury: float = 0.0):
        super().__init__()
        self.agents = nn.ModuleList(agents)
        self.top_k = top_k
        self.treasury = initial_treasury

    def forward(self, x: torch.Tensor, expected_revenue: float = 0.0) -> Tuple[torch.Tensor, List[List[int]], torch.Tensor]:
        """
        Forward pass with Top-K gating.

        Args:
            x: Input tensor [batch_size, input_dim]
            expected_revenue: Expected payout per prediction (passed to agents for strategic bidding)

        Returns:
            final_outputs: Ensemble predictions [batch_size, output_dim]
            winner_indices: List of winner agent indices per sample
            all_bids: Tensor of all bids [batch_size, num_agents]
        """
        batch_size = x.size(0)
        num_agents = len(self.agents)

        if num_agents == 0:
            raise ValueError("MarketLayer has no agents!")

        # Collect outputs and bids from all agents
        all_outputs = []
        all_bids = []

        for agent in self.agents:
            output, bid = agent(x, expected_revenue)
            all_outputs.append(output)  # [batch_size, output_dim]
            all_bids.append(bid.squeeze(-1))  # [batch_size]

        # Stack: [batch_size, num_agents, output_dim] and [batch_size, num_agents]
        # We permute to put batch dimension first for easier vectorized indexing
        all_outputs = torch.stack(all_outputs).permute(1, 0, 2)
        all_bids = torch.stack(all_bids).permute(1, 0)

        # Vectorized Top-K selection
        actual_k = min(self.top_k, num_agents)
        
        # [batch_size, actual_k]
        top_k_values, top_k_indices = torch.topk(all_bids, actual_k, dim=1)
        
        # Expand indices to gather outputs: [batch_size, actual_k, output_dim]
        gather_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, all_outputs.size(-1))
        
        # Gather outputs from winners: [batch_size, actual_k, output_dim]
        selected_outputs = torch.gather(all_outputs, 1, gather_indices)
        
        # Average outputs: [batch_size, output_dim]
        final_outputs = selected_outputs.mean(dim=1)
        
        # Convert indices to list for revenue tracking (CPU operation)
        winner_indices = top_k_indices.tolist()

        return final_outputs, winner_indices, all_bids
