
import torch
# from friedman_net.train import train # Not used
# import friedman_net.main as main_config # Not used

def check_weights_updating():
    print("Running weight update check...")
    
    # We don't need to modify main_config for this standalone test
    
    # Capture initial state
    # We need to intercept the market_layer right after creation in the train function
    # This is tricky without modifying source code. 
    # Instead, let's run a very short training (1 batch) by hacking the loader? 
    # No, simpler: Let's modify train.py slightly to print weight sums before and after.
    
    # Actually, I will create a standalone script that replicates the training step 
    # on a single batch to prove gradients are applied.
    
    from friedman_net.agent import EconomicAgent
    from friedman_net.market import MarketLayer
    from friedman_net.utils import create_optimizer
    
    # Setup
    agent = EconomicAgent(0, 784, 32, 10, 1000.0)
    market = MarketLayer([agent], top_k=1, initial_treasury=1000.0)
    optimizer = create_optimizer(market, 0.01) # High LR to ensure visible change
    
    # Get initial weights
    initial_weight_sum = agent.fc1.weight.sum().item()
    print(f"Initial fc1 weight sum: {initial_weight_sum:.6f}")
    
    # Create dummy data
    data = torch.randn(64, 784)
    target = torch.randint(0, 10, (64,))
    
    # Forward
    optimizer.zero_grad()
    outputs, _, _ = market(data)
    loss = torch.nn.CrossEntropyLoss()(outputs, target)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    
    # Check gradients
    if agent.fc1.weight.grad is None:
        print("ERROR: No gradients computed!")
    else:
        grad_norm = agent.fc1.weight.grad.norm().item()
        print(f"Gradient norm: {grad_norm:.6f}")
        
    # Step
    optimizer.step()
    
    # Check final weights
    final_weight_sum = agent.fc1.weight.sum().item()
    print(f"Final fc1 weight sum:   {final_weight_sum:.6f}")
    
    diff = abs(final_weight_sum - initial_weight_sum)
    if diff > 0:
        print(f"SUCCESS: Weights changed by {diff:.6f}")
    else:
        print("FAILURE: Weights did not change!")

if __name__ == "__main__":
    check_weights_updating()
