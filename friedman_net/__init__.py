"""
Friedman-Descent: Generational Evolutionary Training System

A market-based neuroevolution approach where neural network agents compete economically,
pay rent proportional to complexity, and evolve through mutations.
"""

from .agent import EconomicAgent
from .market import MarketLayer
from .manager import FriedmanManager
from .utils import load_mnist_data, evaluate, print_market_report, create_optimizer
from .train import train

__all__ = [
    'EconomicAgent',
    'MarketLayer',
    'FriedmanManager',
    'load_mnist_data',
    'evaluate',
    'print_market_report',
    'create_optimizer',
    'train',
]
