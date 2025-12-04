"""
Friedman-Descent: Generational Evolutionary Training System
Main entry point with hyperparameters and training invocation.
"""

import torch
from friedman_net import train


# ============================================================================
# HYPERPARAMETERS & CONFIGURATION
# ============================================================================

# Economic Parameters
RENT_FACTOR = 20.0          # Cost per hidden unit per generation
REVENUE_PER_WIN = 0.3       # Reward for being in Top-K (paid from market treasury)
STARTING_WALLET = 2000.0    # Initial capital for new agents
IPO_PROBABILITY = 0.3       # Chance profitable agent spawns child
PROFIT_THRESHOLD = 100.0    # Minimum profit to be eligible for IPO
INITIAL_MARKET_WALLET = 10000.0  # Starting treasury for the market
MARKET_INJECTION = 500.0    # Money injected into market each generation

# Evolution Parameters
EXPAND_FACTOR = 1.5
SHRINK_FACTOR = 0.8
NOISE_STD = 0.01
MUTATION_WEIGHTS = {'expand': 0.4, 'shrink': 0.2, 'noise': 0.4}
MIN_HIDDEN_DIM = 4
MAX_HIDDEN_DIM = 2048

# Training Parameters
INITIAL_POPULATION = 5
INITIAL_HIDDEN_DIMS = [32, 64, 128, 256, 512]
EPOCHS_PER_GENERATION = 3
NUM_GENERATIONS = 20
TOP_K = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 64

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

if __name__ == "__main__":
    market_layer = train(
        # Economic Parameters
        rent_factor=RENT_FACTOR,
        revenue_per_win=REVENUE_PER_WIN,
        starting_wallet=STARTING_WALLET,
        ipo_probability=IPO_PROBABILITY,
        profit_threshold=PROFIT_THRESHOLD,
        initial_market_wallet=INITIAL_MARKET_WALLET,
        market_injection=MARKET_INJECTION,

        # Evolution Parameters
        expand_factor=EXPAND_FACTOR,
        shrink_factor=SHRINK_FACTOR,
        noise_std=NOISE_STD,
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
