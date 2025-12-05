"""
Utility functions for data loading, evaluation, and reporting.
"""

from typing import Tuple, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .market import MarketLayer


def load_mnist_data(batch_size: int, data_dir: str = './data') -> Tuple[DataLoader, DataLoader]:
    """
    Load MNIST dataset with flattened images.

    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/load MNIST data

    Returns:
        train_loader: Training data loader
        test_loader: Test data loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to [784]
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def evaluate(
    market_layer: MarketLayer,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device = torch.device('cpu')
) -> Tuple[float, float]:
    """
    Evaluate market layer on test set.

    Args:
        market_layer: The market layer to evaluate
        test_loader: Test data loader
        criterion: Loss function
        device: Device to run evaluation on

    Returns:
        test_loss: Average test loss
        accuracy: Test accuracy as percentage
    """
    market_layer.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs, _, _ = market_layer(data)
            loss = criterion(outputs, target)
            total_loss += loss.item()
            num_batches += 1

            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    market_layer.train()
    avg_test_loss = total_loss / num_batches
    accuracy = 100.0 * correct / total
    return avg_test_loss, accuracy


def print_market_report(
    generation: int,
    market_layer: MarketLayer,
    train_loss: float,
    test_loss: float,
    test_accuracy: float,
    num_bankruptcies: int,
    num_ipos: int,
    ipo_details: List[Tuple[str, int]],
    images_seen: int,
    train_accuracy: float = None
):
    """
    Print detailed market report for the generation.

    Args:
        generation: Current generation number
        market_layer: The market layer with agents
        train_loss: Average training loss for the generation
        test_loss: Average test loss for the generation
        test_accuracy: Test accuracy percentage
        num_bankruptcies: Number of agents that went bankrupt
        num_ipos: Number of new agents created
        ipo_details: List of (mutation_type, child_hidden_dim) for each IPO
        images_seen: Total number of training images processed in this generation
        train_accuracy: Training accuracy percentage (optional)
    """
    agents = list(market_layer.agents)
    population = len(agents)

    if population == 0:
        print(f"\n{'='*70}")
        print(f"Generation {generation} - EXTINCTION EVENT!")
        print(f"{'='*70}\n")
        return

    avg_hidden = sum(a.hidden_dim for a in agents) / population
    std_hidden = (sum((a.hidden_dim - avg_hidden)**2 for a in agents) / population) ** 0.5
    avg_wallet = sum(a.wallet for a in agents) / population
    avg_age = sum(a.age for a in agents) / population

    # Calculate money supply: treasury + sum of all agent wallets
    total_agent_wealth = sum(a.wallet for a in agents)
    money_supply = market_layer.treasury + total_agent_wealth

    print(f"\n{'='*70}")
    print(f"Generation {generation} Report")
    print(f"{'='*70}")
    print(f"Population: {population} agents")
    print(f"Images Seen: {images_seen}")
    print(f"Market Treasury: ${market_layer.treasury:.2f}")
    print(f"Money Supply: ${money_supply:.2f}")

    # Display losses and accuracies
    if train_accuracy is not None:
        print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    else:
        print(f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Avg Hidden Dim: {avg_hidden:.1f} (std: {std_hidden:.1f})")
    print(f"Avg Agent Wallet: ${avg_wallet:.2f}")
    print(f"Avg Age: {avg_age:.1f} generations")
    print(f"Bankruptcies: {num_bankruptcies}")
    print(f"New IPOs: {num_ipos}")

    if ipo_details:
        print(f"\nIPO Details:")
        for mutation_type, child_dim in ipo_details:
            print(f"  - {mutation_type.upper()}: hidden_dim={child_dim}")

    print(f"\nAgent Details:")
    for i, agent in enumerate(agents):
        profit = agent.wallet - agent.initial_wallet
        profit_str = f"+${profit:.1f}" if profit >= 0 else f"-${abs(profit):.1f}"

        # Calculate win rate
        win_rate = (agent.num_wins / agent.num_bids * 100) if agent.num_bids > 0 else 0.0

        print(f"  Agent {agent.uid:3d}: hidden={agent.hidden_dim:4d}, "
              f"wallet=${agent.wallet:7.2f}, age={agent.age:2d}, profit={profit_str}, "
              f"bids={agent.num_bids}, wins={agent.num_wins} ({win_rate:.1f}%)")
    print(f"{'='*70}\n")


def create_optimizer(market_layer: MarketLayer, learning_rate: float) -> torch.optim.Optimizer:
    """
    Create fresh optimizer for current population.

    Args:
        market_layer: The market layer containing agents
        learning_rate: Learning rate for optimizer

    Returns:
        optimizer: Adam optimizer
    """
    return torch.optim.Adam(market_layer.parameters(), lr=learning_rate)
