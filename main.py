import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Using device:", device)

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

class StandardMLP(nn.Module):
    """Standard fully connected MLP"""
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(StandardMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def clamp_weights(self, bounds_dict):
        with torch.no_grad():
            for name, layer in self.named_children():
                if isinstance(layer, nn.Linear):
                    r = bounds_dict.get(name, None)
                    if r is not None:
                        layer.weight.data.clamp_(-r, r)

def count_parameters(model):
    """Count total number of parameters"""
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, train_loader, criterion, optimizer, device, layer_bounds, dense=True):
    """Train for one epoch"""
    model.train()
    correct = 0
    total = 0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        
        # If dense, clamp weights after optimizer step
        if dense:
            model.clamp_weights(layer_bounds)
        
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    return 100 * correct / total


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return 100 * correct / total


def run_experiment(model_type, width, seed, density=1.0, epochs=20,dense=True,B=10.0):
    print(f"Training {model_type} | width={width} | seed={seed}")
    
    set_seed(seed)

    model = StandardMLP(hidden_size=width).to(device)

    input_dim = 784
    hidden_dim = width
    layer_bounds = {
        'fc1': B / input_dim,
        'fc2': B / hidden_dim 
    }
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        train_epoch(model, train_loader, criterion, optimizer,
                    device, layer_bounds, dense)

    train_acc = evaluate(model, train_loader, device)
    test_acc = evaluate(model, test_loader, device)
    params = count_parameters(model)

    return train_acc, test_acc, params

import numpy as np
import pandas as pd

def main():
    """Run all experiments and save results table per seed"""
    
    # Experimental configurations
    dense = True
    architecture = "Dense MLP" if dense else "Standard MLP"

    widths = [1,2,4,8,16,32,64,128,256,512,1024,2048]
    seeds = [5,6,7,8,9]#[0, 1, 2, 3, 4]
    epochs = 20
    B=10
    print("=" * 60)
    print(f"Architecture: {architecture}")
    if dense:
        print(f"B = {B}")
    print("=" * 60)
    
    for seed in seeds:
        results = []  # store results for this seed
        
        for width in widths:
            train_acc, test_acc, params = run_experiment(
                architecture,
                width,
                seed=seed,
                density=1.0,
                epochs=epochs,
                B=B,
                dense=dense
            )
            
            results.append({
                'Architecture': architecture,
                'Width': width,
                'Train Acc (%)': f"{train_acc:.2f}",
                'Test Acc (%)': f"{test_acc:.2f}",
                'Parameters': f"{params/1000:.1f}K" if params < 1e6 else f"{params/1e6:.2f}M"
            })
        
        # Create DataFrame and display
        df = pd.DataFrame(results)
        print("\n" + "=" * 60)
        print(f"RESULTS TABLE FOR SEED {seed}")
        print("=" * 60)
        print(df.to_string(index=False))
        
        # Save CSV for this seed
        file_name = (
            f"mnist_dense_mlp_results_B_is_{B}_seed{seed}.csv" if dense 
            else f"mnist_standard_mlp_results_seed{seed}.csv"
        )
        df.to_csv(file_name, index=False)
        print(f"\nResults for seed {seed} saved to '{file_name}'")

if __name__ == "__main__":
    results_df = main()