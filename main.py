"""
Friedman-Descent: Generational Evolutionary Training System
Main entry point with hyperparameters and training invocation.
"""

import torch
import random
from friedman_net.train import train


# ============================================================================
# HYPERPARAMETERS & CONFIGURATION
# ============================================================================

# Economic Parameters
RENT_FACTOR = 0.1       # Cost per hidden unit per generation
BID_FLOOR = 0.01        # Minimum bid to prevent bidding collapse
TOTAL_MONEY_SUPPLY = 10000.0 # Total money in the system
NUM_IPOS_PER_GENERATION = 1 # Number of new IPOs to launch per generation
IPO_PROBABILITY = 0.3       # Chance profitable agent spawns child
PROFIT_THRESHOLD = 100.0    # Minimum profit to be eligible for IPO
TREASURY_INJECTION = 0.05   # Inflation rate (percentage of total money supply) added to Treasury per generation

# Evolution Parameters
EXPAND_FACTOR = 2.0
SHRINK_FACTOR = 0.5
NOISE_STD = 0.01
IPO_RANDOM_INIT_PROB = 0.3 # Probability that an IPO is a completely new random agent
MUTATION_WEIGHTS = {'expand': 0.4, 'shrink': 0.2, 'noise': 0.4}
MIN_HIDDEN_DIM = 4
MAX_HIDDEN_DIM = 2048

# Training Parameters
INITIAL_POPULATION = 25
INITIAL_HIDDEN_DIMS = [random.choice([4, 8, 16, 32, 64, 128, 256, 512]) for _ in range(INITIAL_POPULATION)]
EPOCHS_PER_GENERATION = 3
NUM_GENERATIONS = 20
TOP_K = 5
LEARNING_RATE = 0.001
BATCH_SIZE = 64

# Derived Parameters
INITIAL_TREASURY = TOTAL_MONEY_SUPPLY * 0.5 # Treasury starts with half the money
INITIAL_CASH_RESERVES = (TOTAL_MONEY_SUPPLY * 0.5) / INITIAL_POPULATION # Agents split the other half

# Dataset Parameters
INPUT_DIM = 784  # MNIST flattened (28x28)
OUTPUT_DIM = 10  # MNIST classes

# Reproducibility
RANDOM_SEED = 42

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# TRAINING
# ============================================================================

def main():
    market_layer = train(
        # Economic Parameters
        rent_factor=RENT_FACTOR,
        bid_floor=BID_FLOOR,
        initial_cash_reserves=INITIAL_CASH_RESERVES,
        num_ipos_per_generation=NUM_IPOS_PER_GENERATION,
        profit_threshold=PROFIT_THRESHOLD,
        initial_treasury=INITIAL_TREASURY,
        treasury_injection=TREASURY_INJECTION,

        # Evolution Parameters
        expand_factor=EXPAND_FACTOR,
        shrink_factor=SHRINK_FACTOR,
        noise_std=NOISE_STD,
        ipo_random_init_prob=IPO_RANDOM_INIT_PROB,
        mutation_weights=MUTATION_WEIGHTS,
        min_hidden_dim=MIN_HIDDEN_DIM,
        max_hidden_dim=MAX_HIDDEN_DIM,

        # Training Parameters
        initial_population=INITIAL_POPULATION,
        initial_hidden_dims=INITIAL_HIDDEN_DIMS,
        epochs_per_generation=EPOCHS_PER_GENERATION,
        num_generations=NUM_GENERATIONS,
        top_k=TOP_K,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,

        # Dataset Parameters
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,

        # Reproducibility
        random_seed=RANDOM_SEED,

        # Device
        device=DEVICE
    )

    # Save the trained model
    torch.save(market_layer.state_dict(), "market_layer.pth")
    print("Model saved to market_layer.pth")

    return market_layer

if __name__ == "__main__":
    main()
