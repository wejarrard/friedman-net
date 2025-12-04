"""
FriedmanManager: Stateless evolution engine managing economic rules and mutations.
"""

from typing import Tuple, List
import torch
import torch.nn as nn

from .agent import EconomicAgent
from .market import MarketLayer


class FriedmanManager:
    """
    Stateless evolution engine managing economic rules and mutations.

    Responsibilities:
        - Assess rent (deduct based on complexity)
        - Handle bankruptcy (remove unprofitable agents)
        - Execute IPO (spawn mutated children from profitable parents)
    """

    @staticmethod
    def assess_rent(market_layer: MarketLayer, rent_factor: float):
        """
        Deduct rent from all agents and deposit into market wallet.

        Rent = hidden_dim * rent_factor
        """
        with torch.no_grad():
            total_rent_collected = 0.0
            for agent in market_layer.agents:
                rent = agent.hidden_dim * rent_factor
                agent.wallet -= rent
                total_rent_collected += rent

            # Deposit collected rent into market wallet
            market_layer.market_wallet += total_rent_collected

        return total_rent_collected

    @staticmethod
    def handle_bankruptcy(
        market_layer: MarketLayer,
        starting_wallet: float
    ) -> Tuple[int, List[int]]:
        """
        Remove agents with negative wallets.

        Safety: Always keep at least one agent alive (bailout).

        Returns:
            num_bankruptcies: Number of agents removed
            bankrupt_indices: Indices of removed agents
        """
        agents = market_layer.agents

        # Identify solvent agents
        solvent = []
        bankrupt_indices = []

        for i, agent in enumerate(agents):
            if agent.wallet >= 0:
                solvent.append(agent)
            else:
                bankrupt_indices.append(i)

        # Safety: Keep at least one agent (bailout the best one)
        if len(solvent) == 0:
            best_agent = max(agents, key=lambda a: a.wallet)
            best_agent.wallet = starting_wallet * 0.1  # Emergency bailout
            solvent = [best_agent]
            bankrupt_indices = [i for i, a in enumerate(agents) if a is not best_agent]

        # Rebuild ModuleList with survivors
        market_layer.agents = nn.ModuleList(solvent)

        return len(bankrupt_indices), bankrupt_indices

    @staticmethod
    def execute_ipo(
        parent: EconomicAgent,
        mutation_type: str,
        input_dim: int,
        output_dim: int,
        expand_factor: float,
        shrink_factor: float,
        noise_std: float,
        min_hidden_dim: int,
        max_hidden_dim: int
    ) -> EconomicAgent:
        """
        Create a child agent with mutated architecture and inherited weights.

        Args:
            parent: Parent agent to clone and mutate
            mutation_type: 'expand', 'shrink', or 'noise'
            input_dim: Input dimension for new agent
            output_dim: Output dimension for new agent
            expand_factor: Factor to expand hidden dimension
            shrink_factor: Factor to shrink hidden dimension
            noise_std: Standard deviation for noise mutation
            min_hidden_dim: Minimum allowed hidden dimension
            max_hidden_dim: Maximum allowed hidden dimension

        Returns:
            child: New agent with mutated architecture
        """
        parent_dim = parent.hidden_dim

        # Determine child hidden_dim based on mutation type
        if mutation_type == 'expand':
            child_dim = int(parent_dim * expand_factor)
        elif mutation_type == 'shrink':
            child_dim = int(parent_dim * shrink_factor)
        else:  # noise
            child_dim = parent_dim

        # Enforce bounds
        child_dim = max(min_hidden_dim, min(child_dim, max_hidden_dim))

        # Create child agent
        child = EconomicAgent(
            input_dim,
            child_dim,
            output_dim,
            starting_wallet=0.0,
            min_hidden_dim=min_hidden_dim,
            max_hidden_dim=max_hidden_dim
        )

        # Inherit weights based on mutation type
        with torch.no_grad():
            if mutation_type == 'expand':
                FriedmanManager._copy_weights_expand(parent, child, parent_dim, child_dim)
            elif mutation_type == 'shrink':
                FriedmanManager._copy_weights_shrink(parent, child, parent_dim, child_dim)
            else:  # noise
                FriedmanManager._copy_weights_noise(parent, child, noise_std)

        # Wallet inheritance: Split 50/50
        wallet_split = parent.wallet * 0.5
        child.wallet = wallet_split
        parent.wallet = wallet_split

        # Child is newborn
        child.age = 0
        child.initial_wallet = child.wallet

        return child

    @staticmethod
    def _copy_weights_expand(
        parent: EconomicAgent,
        child: EconomicAgent,
        parent_dim: int,
        child_dim: int
    ):
        """
        Copy parent weights to top-left corner, pad rest with zeros.

        Preserves parent function while adding new capacity.
        """
        # fc1: [parent_dim, input_dim] -> [child_dim, input_dim]
        child.fc1.weight[:parent_dim, :] = parent.fc1.weight.clone()
        child.fc1.bias[:parent_dim] = parent.fc1.bias.clone()

        # fc2: [output_dim, parent_dim] -> [output_dim, child_dim]
        child.fc2.weight[:, :parent_dim] = parent.fc2.weight.clone()
        child.fc2.bias[:] = parent.fc2.bias.clone()

        # bid_head: [1, parent_dim] -> [1, child_dim]
        child.bid_head.weight[:, :parent_dim] = parent.bid_head.weight.clone()
        child.bid_head.bias[:] = parent.bid_head.bias.clone()

    @staticmethod
    def _copy_weights_shrink(
        parent: EconomicAgent,
        child: EconomicAgent,
        parent_dim: int,
        child_dim: int
    ):
        """
        Slice first child_dim neurons from parent.

        Aggressive pruning for efficiency.
        """
        # fc1: [parent_dim, input_dim] -> [child_dim, input_dim]
        child.fc1.weight[:, :] = parent.fc1.weight[:child_dim, :].clone()
        child.fc1.bias[:] = parent.fc1.bias[:child_dim].clone()

        # fc2: [output_dim, parent_dim] -> [output_dim, child_dim]
        child.fc2.weight[:, :] = parent.fc2.weight[:, :child_dim].clone()
        child.fc2.bias[:] = parent.fc2.bias.clone()

        # bid_head: [1, parent_dim] -> [1, child_dim]
        child.bid_head.weight[:, :] = parent.bid_head.weight[:, :child_dim].clone()
        child.bid_head.bias[:] = parent.bid_head.bias.clone()

    @staticmethod
    def _copy_weights_noise(parent: EconomicAgent, child: EconomicAgent, noise_std: float):
        """
        Copy all weights and add Gaussian noise.

        Exploration without changing complexity.
        """
        # fc1
        child.fc1.weight[:, :] = parent.fc1.weight.clone() + \
                                 torch.randn_like(parent.fc1.weight) * noise_std
        child.fc1.bias[:] = parent.fc1.bias.clone() + \
                           torch.randn_like(parent.fc1.bias) * noise_std

        # fc2
        child.fc2.weight[:, :] = parent.fc2.weight.clone() + \
                                torch.randn_like(parent.fc2.weight) * noise_std
        child.fc2.bias[:] = parent.fc2.bias.clone() + \
                           torch.randn_like(parent.fc2.bias) * noise_std

        # bid_head
        child.bid_head.weight[:, :] = parent.bid_head.weight.clone() + \
                                      torch.randn_like(parent.bid_head.weight) * noise_std
        child.bid_head.bias[:] = parent.bid_head.bias.clone() + \
                                torch.randn_like(parent.bid_head.bias) * noise_std
