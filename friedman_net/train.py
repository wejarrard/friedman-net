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
    bid_floor: float,
    initial_cash_reserves: float,
    num_ipos_per_generation: int,
    profit_threshold: float,
    initial_treasury: float,
    treasury_injection: float,

    # Evolution Parameters
    expand_factor: float,
    shrink_factor: float,
    noise_std: float,
    ipo_random_init_prob: float,
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
    print(f"  Inflation Rate: {treasury_injection*100:.1f}%")
    print(f"{'='*70}\n")

    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size)

    # Initialize population with diverse hidden dimensions
    print("Initializing population...")
    
    agent_id_counter = 0
    initial_agents = []
    for hidden_dim in initial_hidden_dims:
        initial_agents.append(EconomicAgent(
            uid=agent_id_counter,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            initial_cash_reserves=initial_cash_reserves,
            min_hidden_dim=min_hidden_dim,
            max_hidden_dim=max_hidden_dim,
            bid_floor=bid_floor
        ))
        agent_id_counter += 1

    # Create market layer and manager
    market_layer = MarketLayer(initial_agents, top_k=top_k, initial_treasury=initial_treasury)
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

        # Dynamic revenue per win calculation
        # Market allocates its entire wallet across all predictions for the upcoming generation
        # Total wins = epochs * batches * batch_size * top_k
        # We estimate total wins to determine per-win payout
        
        num_batches_est = len(train_loader)
        total_predictions_est = epochs_per_generation * num_batches_est * batch_size
        
        if total_predictions_est > 0:
            allocation_per_prediction = market_layer.treasury / total_predictions_est
        else:
            allocation_per_prediction = 0.0
            
        # We don't deduct from treasury yet, we stream it out as wins happen
        # Ideally we'd decrement, but floating point drift might occur.
        # Let's just trust the division and maybe zero it out at end of phase if we want perfect accounting.
        # For simplicity: we will deduct as we go, and if we run out, we run out (though math says we shouldn't).

        optimizer = create_optimizer(market_layer, learning_rate)
        market_layer.train()

        images_seen = 0

        for epoch in range(epochs_per_generation):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                batch_size_actual = data.size(0)
                images_seen += batch_size_actual

                optimizer.zero_grad()

                # Forward pass with revenue information
                # Agents can now bid strategically based on expected payout
                outputs, winner_indices, all_bids = market_layer(data, allocation_per_prediction)
                loss = criterion(outputs, target)

                # Logging for the first batch of the first epoch of the generation
                if epoch == 0 and batch_idx == 0:
                    with torch.no_grad():
                        avg_bid = all_bids.mean().item()
                        max_bid = all_bids.max().item()
                        min_bid = all_bids.min().item()
                        print(f"  [Gen {generation} Bidding Stats] Avg Bid: {avg_bid:.4f}, Max: {max_bid:.4f}, Min: {min_bid:.4f}, Payout/Img: ${allocation_per_prediction:.6f}")
                        # Optional: Show detailed bids for first sample
                        # print(f"  [Sample 0 Bids] {all_bids[0].tolist()}")

                # Backward pass
                loss.backward()
                optimizer.step()

                # Revenue distribution (outside autograd) - Pay from market treasury
                with torch.no_grad():
                    # Track bidding stats for each agent
                    for agent in market_layer.agents:
                        agent.num_bids += batch_size_actual  # Each agent bids on every sample

                    for batch_sample_idx, sample_winners in enumerate(winner_indices):
                        num_winners = len(sample_winners)
                        if num_winners > 0:
                            payout_per_winner = allocation_per_prediction / num_winners

                            for agent_idx in sample_winners:
                                # Track win
                                market_layer.agents[agent_idx].num_wins += 1

                                # Get the bid amount this agent made for this sample
                                bid_amount = all_bids[batch_sample_idx, agent_idx].item()

                                # Treasury collects the bid
                                market_layer.treasury += bid_amount
                                market_layer.agents[agent_idx].wallet -= bid_amount

                                # Agent receives the payout
                                # Only pay if market has enough money (safety check)
                                payout = min(payout_per_winner, market_layer.treasury)
                                if payout > 0:
                                    market_layer.treasury -= payout
                                    market_layer.agents[agent_idx].wallet += payout
                                    market_layer.agents[agent_idx].revenue_earned += payout

                    # Calculate losses (bids that didn't win)
                    for agent in market_layer.agents:
                        agent.num_losses = agent.num_bids - agent.num_wins

        # Clean up any floating point dust
        # market_layer.treasury = max(0.0, market_layer.treasury)

        # ====================================================================
        # PHASE 2: AUDIT (Rent Assessment)
        # ====================================================================

        with torch.no_grad():
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
            initial_cash_reserves
        )

        # ====================================================================
        # PHASE 4: IPO (Evolution)
        # ====================================================================

        ipo_details = []
        new_agents = []
        
        # Calculate total money supply (Treasury + All Agent Wallets)
        total_agent_wealth = sum(a.wallet for a in market_layer.agents)
        total_money_supply = market_layer.treasury + total_agent_wealth
        
        # Inflation: Print new money and give to Treasury
        inflation_amount = total_money_supply * treasury_injection
        market_layer.treasury += inflation_amount
        
        # Calculate IPO funding based on largest company reserve
        # Policy: New IPOs get 1/10th of the richest company's wallet.
        # This is funded directly by the Treasury.
        
        if len(market_layer.agents) > 0:
            max_agent_wallet = max(a.wallet for a in market_layer.agents)
        else:
            max_agent_wallet = initial_cash_reserves # Fallback if extinction
            
        target_funding_per_ipo = max_agent_wallet / 10.0
        total_required_funding = target_funding_per_ipo * num_ipos_per_generation
        
        # Check if Treasury can afford it
        if market_layer.treasury >= total_required_funding:
            funding_per_ipo = target_funding_per_ipo
        else:
            # Partial funding if treasury is low
            if num_ipos_per_generation > 0:
                funding_per_ipo = market_layer.treasury / num_ipos_per_generation
            else:
                funding_per_ipo = 0.0
        
        # Deduct funding from Treasury
        total_actual_funding = funding_per_ipo * num_ipos_per_generation
        market_layer.treasury -= total_actual_funding
        
        with torch.no_grad():
            # Find the single best parent to spawn a child
            # Priority: 1. Profitable > Threshold (highest profit)
            #           2. Highest Wallet (if none meet threshold)
            
            best_parent = None
            
            # Sort by profit first
            agents_by_profit = sorted(
                market_layer.agents, 
                key=lambda a: (a.wallet - a.initial_wallet), 
                reverse=True
            )
            
            if agents_by_profit:
                best_candidate = agents_by_profit[0]
                # We prefer a profitable agent, but if none exist, we take the best available
                best_parent = best_candidate

            # Execute IPOs
            for _ in range(num_ipos_per_generation):
                # Decide: Mutation or Random New Entrant?
                # If no parent exists (rare, but possible if all dead?), force random init
                is_random_init = random.random() < ipo_random_init_prob or best_parent is None
                
                if is_random_init:
                    mutation_type = 'random_init'
                    # Pick a random power of 2 within bounds
                    # Generate powers of 2 list dynamically based on min/max
                    powers_of_2 = []
                    curr = min_hidden_dim
                    while curr <= max_hidden_dim:
                        powers_of_2.append(curr)
                        curr *= 2
                    
                    # execute_ipo handles random_init logic, but expects a parent arg.
                    # We can pass best_parent (even if None? No, types say EconomicAgent).
                    # Actually, if best_parent is None, we have a problem if execute_ipo touches it.
                    # But execute_ipo only touches parent if mutation_type != 'random_init'.
                    # Let's verify manager.py. It accesses `parent.hidden_dim` at the top.
                    # So we must pass a dummy parent or handle None in manager.
                    # Let's handle it here: if best_parent is None, we must ensure we don't crash.
                    # But wait, `agents_by_profit` comes from `market_layer.agents`.
                    # If population > 0, best_parent is NOT None.
                    # If population == 0, we have an extinction event, and the loop might crash or not run.
                    # If population == 0, we SHOULD spawn random agents to restart!
                    # So we need a dummy parent if best_parent is None.
                    pass 
                else:
                    # Select mutation type from weights
                    mutation_type = random.choices(
                        list(mutation_weights.keys()),
                        weights=list(mutation_weights.values())
                    )[0]

                # Handle Extinction Case for Random Init
                parent_arg = best_parent
                if parent_arg is None:
                     # Create a temporary dummy agent just to satisfy the method signature
                     # It won't be used for random_init anyway
                     parent_arg = EconomicAgent(0, input_dim, min_hidden_dim, output_dim, 0, bid_floor=bid_floor)

                # Create child
                child = friedman_manager.execute_ipo(
                    parent_arg,
                    agent_id_counter,
                    mutation_type,
                    input_dim,
                    output_dim,
                    expand_factor,
                    shrink_factor,
                    noise_std,
                    min_hidden_dim,
                    max_hidden_dim,
                    bid_floor,
                    initial_funding=funding_per_ipo
                )
                new_agents.append(child)
                agent_id_counter += 1
                ipo_details.append((mutation_type, child.hidden_dim))
                
            # If no IPOs happened, the inflation 'evaporates' or stays in the void.
            # Alternatively, we could add it to the market wallet, but the user said:
            # "inflation should instead just be the money given to the new companies"

        # Warmup training for new IPO agents (1 epoch)
        if len(new_agents) > 0:
            print(f"  [Warmup] Training {len(new_agents)} new IPO agents for 1 epoch...")

            for child in new_agents:
                child = child.to(device)
                child.train()

                # Create optimizer for just this child
                child_optimizer = torch.optim.Adam(child.parameters(), lr=learning_rate)

                # Train for 1 epoch
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)

                    child_optimizer.zero_grad()
                    output, bid = child(data)
                    loss = criterion(output, target)
                    loss.backward()
                    child_optimizer.step()

        # Add new agents to population
        for child in new_agents:
            child = child.to(device)
            market_layer.agents.append(child)

        num_ipos = len(new_agents)

        # ====================================================================
        # PHASE 5: EVALUATION (After population changes)
        # ====================================================================

        with torch.no_grad():
            # Evaluate on both train and test sets after all population changes
            # This shows the actual state of the population entering the next generation
            train_loss, train_accuracy = evaluate(market_layer, train_loader, criterion, device)
            test_loss, test_accuracy = evaluate(market_layer, test_loader, criterion, device)

        # ====================================================================
        # PHASE 6: REPORTING
        # ====================================================================

        print_market_report(
            generation, market_layer, train_loss, test_loss, test_accuracy,
            num_bankruptcies, num_ipos, ipo_details, images_seen, train_accuracy
        )

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================

    print(f"\n{'='*70}")
    print("Training Complete!")
    print(f"{'='*70}")
    print(f"Final Population: {len(market_layer.agents)} agents")

    if len(market_layer.agents) > 0:
        final_train_loss, final_train_accuracy = evaluate(market_layer, train_loader, criterion, device)
        final_test_loss, final_test_accuracy = evaluate(market_layer, test_loader, criterion, device)
        print(f"Final Train Loss: {final_train_loss:.4f}, Accuracy: {final_train_accuracy:.2f}%")
        print(f"Final Test Loss: {final_test_loss:.4f}, Accuracy: {final_test_accuracy:.2f}%")

        agents = list(market_layer.agents)
        avg_hidden = sum(a.hidden_dim for a in agents) / len(agents)
        print(f"Final Avg Hidden Dim: {avg_hidden:.1f}")

        oldest_agent = max(agents, key=lambda a: a.age)
        print(f"Oldest Agent: Age {oldest_agent.age}, Hidden Dim {oldest_agent.hidden_dim}")

    print(f"{'='*70}\n")

    return market_layer
