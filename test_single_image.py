"""
Test the trained Friedman-Descent model on a single image.
Displays the image, true label, and prediction.
"""

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from friedman_net.market import MarketLayer
from friedman_net.agent import EconomicAgent


def load_model(model_path: str, device: torch.device):
    """
    Load the trained MarketLayer model.

    Note: We need to reconstruct the model architecture first,
    then load the saved weights.
    """
    # Load the state dict to inspect the architecture
    state_dict = torch.load(model_path, map_location=device)

    # Infer the number of agents and their hidden dimensions from the state dict
    # The state dict has keys like: 'agents.0.fc1.weight', 'agents.0.fc2.weight', etc.
    agent_indices = set()
    agent_configs = {}

    for key in state_dict.keys():
        if key.startswith('agents.'):
            parts = key.split('.')
            agent_idx = int(parts[1])
            agent_indices.add(agent_idx)

            # Get hidden dim from fc1.weight shape [hidden_dim, input_dim]
            if parts[2] == 'fc1' and parts[3] == 'weight':
                hidden_dim = state_dict[key].shape[0]
                agent_configs[agent_idx] = hidden_dim

    # Get input and output dimensions from first agent
    if 0 in agent_configs:
        input_dim = state_dict['agents.0.fc1.weight'].shape[1]
        output_dim = state_dict['agents.0.fc2.weight'].shape[0]
    else:
        raise ValueError("No agents found in saved model!")

    print(f"Model Architecture:")
    print(f"  Input Dim: {input_dim}")
    print(f"  Output Dim: {output_dim}")
    print(f"  Number of Agents: {len(agent_configs)}")
    print(f"  Agent Hidden Dims: {sorted(agent_configs.values())}")

    # Reconstruct agents
    agents = []
    for idx in sorted(agent_indices):
        agent = EconomicAgent(
            uid=idx,
            input_dim=input_dim,
            hidden_dim=agent_configs[idx],
            output_dim=output_dim,
            initial_cash_reserves=0.0,  # Not needed for inference
            min_hidden_dim=4,
            max_hidden_dim=2048
        )
        agents.append(agent)

    # Recreate market layer
    # We don't know the exact top_k, but it doesn't matter for inference
    market_layer = MarketLayer(agents, top_k=2)

    # Load the weights
    market_layer.load_state_dict(state_dict)
    market_layer = market_layer.to(device)
    market_layer.eval()

    return market_layer


def load_test_image(image_idx: int = 0):
    """Load a single test image from MNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))  # Flatten to [784]
    ])

    # Load MNIST test set
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # Get the image and label
    image, label = test_dataset[image_idx]

    return image, label, test_dataset


def show_image_with_prediction(image_tensor, true_label, predicted_label):
    """Display the image with its true and predicted labels."""
    # Convert normalized tensor back to image
    # Undo normalization: x_original = x_normalized * std + mean
    image_normalized = image_tensor.view(28, 28)
    image = image_normalized * 0.3081 + 0.1307

    # Clip to [0, 1] range
    image = torch.clamp(image, 0, 1)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.imshow(image.cpu().numpy(), cmap='gray')
    plt.axis('off')

    # Add title with labels
    is_correct = true_label == predicted_label
    color = 'green' if is_correct else 'red'
    status = '✓' if is_correct else '✗'

    plt.title(
        f"{status} True Label: {true_label} | Predicted: {predicted_label}",
        fontsize=16,
        color=color,
        pad=20
    )

    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    print("\nImage saved to 'prediction_result.png'")
    plt.show()


def main():
    # Configuration
    model_path = 'market_layer.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_idx = 0  # Change this to test different images

    print(f"Device: {device}")
    print(f"Loading model from: {model_path}\n")

    # Load model
    market_layer = load_model(model_path, device)

    # Load test image
    print(f"\nLoading test image #{image_idx}...")
    image, true_label = load_test_image(image_idx)[:2]

    # Make prediction
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)  # Add batch dimension
        outputs, winner_indices, all_bids = market_layer(image_batch)

        # Get predicted class
        _, predicted = outputs.max(1)
        predicted_label = predicted.item()

        # Get prediction confidence (softmax probabilities)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0, predicted_label].item() * 100

    # Display results
    print(f"\n{'='*50}")
    print(f"Image #{image_idx} Results:")
    print(f"{'='*50}")
    print(f"True Label: {true_label}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Result: {'✓ CORRECT' if true_label == predicted_label else '✗ INCORRECT'}")

    # Show which agents won
    print(f"\nWinning Agents: {winner_indices[0]}")
    print(f"Top Bids: {all_bids[0, winner_indices[0]].cpu().numpy()}")

    # Show all class probabilities
    print(f"\nAll Class Probabilities:")
    for digit in range(10):
        prob = probabilities[0, digit].item() * 100
        bar = '█' * int(prob / 2)  # Simple bar chart
        marker = ' <--' if digit == predicted_label else ''
        print(f"  {digit}: {prob:5.2f}% {bar}{marker}")

    print(f"{'='*50}\n")

    # Visualize
    show_image_with_prediction(image, true_label, predicted_label)


if __name__ == "__main__":
    main()
