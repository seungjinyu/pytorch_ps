import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
import time
import os
import random
import numpy as np

# root dir setting
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

# platform device check
def setting_platform():
    return torch.device("cuda" if torch.cuda.is_available() else "mps")

# verbose print of parameter and gradient shape/size/sample values
def print_param_and_grad_stats_verbose(model: nn.Module, sample_count: int = 5):
    print("\n[Gradient Format Detail]")
    for name, param in model.named_parameters():
        param_numel = param.numel()
        grad_numel = param.grad.numel() if param.grad is not None else 0

        param_size = param_numel * 4 / (1024 * 1024)  # float32: 4 bytes
        grad_size = grad_numel * 4 / (1024 * 1024)

        shape = param.shape

        grad_sample = []
        if param.grad is not None:
            flat_grad = param.grad.view(-1).detach().cpu().numpy()
            grad_sample = flat_grad[:sample_count].tolist()

        print(f"{name:<35} | shape: {str(shape):<30} | param: {param_numel:>16,d} | grad num: {grad_numel:>8,d} ")

def main():
    # training parameters
    batch_size = 128
    learning_rate = 0.01
    num_epochs = 1

    device = setting_platform()
    print(f"Running on {device}")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # train/test set
    train_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=True)
    num_samples = len(train_dataset)
    random.seed(42)
    indices = random.sample(range(num_samples), num_samples // 10)
    train_dataset = Subset(train_dataset, indices)

    test_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Set up the model to resnet18")
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    print("Set up the criterion is CrossEntropyLoss")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
        print_param_and_grad_stats_verbose(model)

        end_time = time.time()
        print(f"Elapsed time {end_time - start_time:.4f} seconds")

        # evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == '__main__':
    main()
