import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import random
import time

# root dir
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
os.makedirs(root_dir, exist_ok=True)

# 저장 디렉토리
save_dir = os.path.join(root_dir, "epoch_data")
os.makedirs(save_dir, exist_ok=True)

def main():
    batch_size = 128
    learning_rate = 0.01
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=True)
    indices = random.sample(range(len(train_dataset)), len(train_dataset) // 10)
    train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # 저장: gradient 및 parameter
        grad_dict = {}
        param_dict = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_dict[name] = param.grad.detach().cpu().numpy()
            param_dict[name] = param.data.detach().cpu().numpy()

        grad_file = os.path.join(save_dir, f"epo_grad_{epoch + 1}.npz")
        param_file = os.path.join(save_dir, f"epo_param_{epoch + 1}.npz")

        np.savez_compressed(grad_file, **grad_dict)
        np.savez_compressed(param_file, **param_dict)

        print(f"Epoch {epoch + 1}: Gradients and Parameters saved.")

if __name__ == '__main__':
    main()
