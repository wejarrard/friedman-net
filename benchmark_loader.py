import torch
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def benchmark():
    print("Starting benchmark...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    # Load actual data
    dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: 64")
    
    start_time = time.time()
    
    # Simulate one epoch of "training" (just loading and moving to device)
    batches = 0
    images = 0
    for data, target in loader:
        # Simulate the minimal work done in your loop
        # data = data.to('cpu') 
        batches += 1
        images += data.size(0)
        
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Processed {images} images in {duration:.4f} seconds")
    print(f"Rate: {images/duration:.2f} images/sec")

if __name__ == "__main__":
    benchmark()