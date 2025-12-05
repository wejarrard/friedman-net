
import torch
from friedman_net.agent import EconomicAgent
from friedman_net.manager import FriedmanManager
from friedman_net.market import MarketLayer
from friedman_net.utils import create_optimizer

def check_ipo_trainability():
    print("Checking if IPO children are trainable...")
    
    # 1. Create Parent
    parent = EconomicAgent(0, 784, 32, 10, 1000.0)
    
    # 2. Create Child via IPO (Noise mutation)
    manager = FriedmanManager()
    print("Executing IPO (Noise)...")
    child = manager.execute_ipo(
        parent, 1, 'noise', 784, 10, 1.5, 0.8, 0.1, 4, 2048, initial_funding=500.0
    )
    
    # 3. Verify Child Gradients
    print(f"Child fc1 requires_grad: {child.fc1.weight.requires_grad}")
    
    if not child.fc1.weight.requires_grad:
        print("FAILURE: Child weights do not require grad!")
        return

    # 4. Train Child
    market = MarketLayer([child], top_k=1, initial_treasury=1000.0)
    optimizer = create_optimizer(market, 0.01)
    
    initial_sum = child.fc1.weight.sum().item()
    print(f"Initial Child Weight Sum: {initial_sum:.4f}")
    
    data = torch.randn(64, 784)
    target = torch.randint(0, 10, (64,))
    
    optimizer.zero_grad()
    outputs, _, _ = market(data)
    loss = torch.nn.CrossEntropyLoss()(outputs, target)
    loss.backward()
    
    if child.fc1.weight.grad is None:
        print("FAILURE: No gradient after backward pass!")
        return
        
    grad_norm = child.fc1.weight.grad.norm().item()
    print(f"Gradient Norm: {grad_norm:.4f}")
    
    optimizer.step()
    
    final_sum = child.fc1.weight.sum().item()
    print(f"Final Child Weight Sum:   {final_sum:.4f}")
    
    if abs(final_sum - initial_sum) > 0:
        print("SUCCESS: Child agent updated successfully.")
    else:
        print("FAILURE: Child agent weights did not change.")

if __name__ == "__main__":
    check_ipo_trainability()
