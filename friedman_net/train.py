"""
Training loop for Friedman-Descent evolutionary training system.
"""

import random
from typing import List, Dict
import torch
import torch.nn as nn

from .agent import EconomicAgent
from .market import MarketLayer
from .manager import FriedmanManager
from .utils import load_mnist_data, evaluate, print_market_report, create_optimizer


def train(
    # Economic Parameters
    rent_factor: float,
    revenue_per_win: float,
    starting_wallet: float,
    ipo_probability: float,
    profit_threshold: float,
    initial_market_wallet: float,
    market_injection: float,

    # Evolution Parameters
    expand_factor: float,
    shrink_factor: float,
    noise_std: float,
    mutation_weights: Dict[str, float],
    min_hidden_dim: int,
    max_hidden_dim: int,

    # Training Parameters
    initial_population: int,
    initial_hidden_dims: List[int],
    epochs_per_generation: int,
    num_generations: int,
    top_k: int,
    learning_rate: float,
    batch_size: int,

    # Dataset Parameters
    input_dim: int,
    output_dim: int,

    # Reproducibility
    random_seed: int,

    # Device
    device: torch.device = torch.device('cpu')
):
    """
    Main training loop implementing generational evolutionary training.

    Each generation consists of 4 phases:
        1. BOOM: Training with gradient descent
        2. AUDIT: Evaluation and rent assessment
        3. BUST: Bankruptcy handling
        4. IPO: Evolutionary mutations

    Args:
        Economic parameters for rent, revenue, and IPO mechanics
        Evolution parameters for mutations
        Training parameters for population and learning
        Dataset parameters
        Random seed for reproducibility
    """
    # Set random seeds
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    print(f"\n{'='*70}")
    print("Friedman-Descent: Generational Evolutionary Training System")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  Device: {device}")
    print(f"  Generations: {num_generations}")
    print(f"  Epochs per Generation: {epochs_per_generation}")
    print(f"  Initial Population: {initial_population}")
    print(f"  Top-K: {top_k}")
    print(f"  Rent Factor: {rent_factor}")
    print(f"  Revenue per Win: ${revenue_per_win}")
    print(f"{'='*70}\n")

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size)

    # Initialize population with diverse hidden dimensions
    print("Initializing population...")
    initial_agents = [
        EconomicAgent(
            input_dim,
            hidden_dim,
            output_dim,
            starting_wallet,
            min_hidden_dim,
            max_hidden_dim
        )
        for hidden_dim in initial_hidden_dims
    ]

    # Create market layer and manager
    market_layer = MarketLayer(initial_agents, top_k=top_k, initial_market_wallet=initial_market_wallet)
    market_layer = market_layer.to(device)
    friedman_manager = FriedmanManager()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    print("Starting evolutionary training...\n")

    # ========================================================================
    # INITIAL BASELINE EVALUATION (Before any training)
    # ========================================================================

    print("Evaluating initial random population...")
    initial_test_loss, initial_accuracy = evaluate(market_layer, test_loader, criterion, device)
    print(f"Initial Test Accuracy (untrained): {initial_accuracy:.2f}%")
    print(f"Initial Test Loss (untrained): {initial_test_loss:.4f}\n")

    # ========================================================================
    # GENERATIONAL LOOP
    # ========================================================================

    for generation in range(num_generations):

        # Reset generation statistics
        for agent in market_layer.agents:
            agent.reset_generation_stats()

        # ====================================================================
        # PHASE 1: BOOM (Training)
        # ====================================================================

        optimizer = create_optimizer(market_layer, learning_rate)
        market_layer.train()

        total_loss = 0.0
        num_batches = 0

        for epoch in range(epochs_per_generation):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                # Forward pass
                outputs, winner_indices = market_layer(data)
                loss = criterion(outputs, target)

                # Track loss
                total_loss += loss.item()
                num_batches += 1

                # Backward pass
                loss.backward()
                optimizer.step()

                # Revenue distribution (outside autograd) - Pay from market wallet
                with torch.no_grad():
                    for sample_winners in winner_indices:
                        for agent_idx in sample_winners:
                            # Only pay if market has enough money
                            if market_layer.market_wallet >= revenue_per_win:
                                market_layer.market_wallet -= revenue_per_win
                                market_layer.agents[agent_idx].wallet += revenue_per_win
                                market_layer.agents[agent_idx].revenue_earned += revenue_per_win

        avg_train_loss = total_loss / num_batches

        # ====================================================================
        # PHASE 2: AUDIT (Evaluation)
        # ====================================================================

        with torch.no_grad():
            test_loss, test_accuracy = evaluate(market_layer, test_loader, criterion, device)

            # Inject money into market (economic stimulus)
            market_layer.market_wallet += market_injection

            # Assess rent (collect from agents and deposit into market)
            total_rent_collected = friedman_manager.assess_rent(market_layer, rent_factor)

            # Age all agents
            for agent in market_layer.agents:
                agent.age += 1

        # ====================================================================
        # PHASE 3: BUST (Bankruptcy)
        # ====================================================================

        num_bankruptcies, bankrupt_indices = friedman_manager.handle_bankruptcy(
            market_layer,
            starting_wallet
        )

        # ====================================================================
        # PHASE 4: IPO (Evolution)
        # ====================================================================

        ipo_details = []
        new_agents = []

        with torch.no_grad():
            # Identify profitable agents
            profitable_agents = [
                agent for agent in market_layer.agents
                if (agent.wallet - agent.initial_wallet) > profit_threshold
            ]

            # Each profitable agent may spawn a child
            for parent in profitable_agents:
                if random.random() < ipo_probability:
                    # Select mutation type
                    mutation_type = random.choices(
                        list(mutation_weights.keys()),
                        weights=list(mutation_weights.values())
                    )[0]

                    # Create child
                    child = friedman_manager.execute_ipo(
                        parent,
                        mutation_type,
                        input_dim,
                        output_dim,
                        expand_factor,
                        shrink_factor,
                        noise_std,
                        min_hidden_dim,
                        max_hidden_dim
                    )
                    new_agents.append(child)
                    ipo_details.append((mutation_type, child.hidden_dim))

        # Add new agents to population
        for child in new_agents:
            child = child.to(device)
            market_layer.agents.append(child)

        num_ipos = len(new_agents)

        # ====================================================================
        # PHASE 5: REPORTING
        # ====================================================================

        print_market_report(
            generation, market_layer, avg_train_loss, test_loss, test_accuracy,
            num_bankruptcies, num_ipos, ipo_details
        )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Final Population: {len(market_layer.agents)} agents")

    if len(market_layer.agents) > 0:
        final_test_loss, final_accuracy = evaluate(market_layer, test_loader, criterion, device)
        print(f"Final Test Loss: {final_test_loss:.4f}")
        print(f"Final Test Accuracy: {final_accuracy:.2f}%")

        agents = list(market_layer.agents)
        avg_hidden = sum(a.hidden_dim for a in agents) / len(agents)
        print(f"Final Avg Hidden Dim: {avg_hidden:.1f}")

        oldest_agent = max(agents, key=lambda a: a.age)
        print(f"Oldest Agent: Age {oldest_agent.age}, Hidden Dim {oldest_agent.hidden_dim}")

    print(f"{'='*70}\n")

    return market_layer
