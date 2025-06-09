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
import resnet18_utils as ru
import zlib, bz2, lzma, lz4.frame, zstandard as zstd, snappy, blosc2

# root dir setting
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

def compress_and_measure(data: bytes, algorithm: str):
    start = time.time()
    if algorithm == "zlib":
        comp = zlib.compress(data)
    elif algorithm == "bz2":
        comp = bz2.compress(data)
    elif algorithm == "lzma":
        comp = lzma.compress(data)
    elif algorithm == "lz4":
        comp = lz4.frame.compress(data)
    elif algorithm == "zstd":
        comp = zstd.ZstdCompressor().compress(data)
    elif algorithm == "snappy":
        comp = snappy.compress(data)
    elif algorithm == "blosc":
        comp = blosc2.compress(data)
    else:
        raise ValueError("Unknown compression algorithm")

    comp_time = time.time() - start
    return comp, comp_time

def main():
    batch_size = 128
    learning_rate = 0.01
    num_epochs = 1
    device = ru.setting_platform()

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=root_dir, train=False, transform=transform, download=True)

    num_samples = len(train_dataset)
    random.seed(42)
    indices = random.sample(range(num_samples), num_samples // 10)
    train_dataset = Subset(train_dataset, indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

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

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")
        end_time = time.time()
        print(f"Elapsed time {end_time - start_time:.4f} seconds")

        ru.print_param_and_grad_stats(model)

        # Accuracy check (optional, commented out)
        # model.eval()
        # correct, total = 0, 0
        # with torch.no_grad():
        #     for inputs, targets in test_loader:
        #         inputs, targets = inputs.to(device), targets.to(device)
        #         outputs = model(inputs)
        #         _, predicted = outputs.max(1)
        #         total += targets.size(0)
        #         correct += predicted.eq(targets).sum().item()
        # print(f"Test Accuracy: {100 * correct / total:.2f}%")

        epoch_results = {algo: {"size": 0, "time": 0.0} for algo in ["zlib", "bz2", "lzma", "lz4", "zstd", "snappy", "blosc"]}
        total_original_size = 0

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad_np = param.grad.detach().cpu().numpy().astype(np.float32)
            grad_bytes = grad_np.tobytes()
            original_size = len(grad_bytes)
            total_original_size += original_size

            for algo in epoch_results:
                comp_data, comp_time = compress_and_measure(grad_bytes, algo)
                epoch_results[algo]["size"] += len(comp_data)
                epoch_results[algo]["time"] += comp_time

        print(f"\n[Epoch {epoch+1} Compression Summary] Total Original Size: {total_original_size} bytes")
        for algo, stats in epoch_results.items():
            ratio = stats["size"] / total_original_size
            print(f"[{algo.upper()}] Total Compressed: {stats['size']} bytes | Ratio: {ratio:.2%} | Time: {stats['time']:.4f} sec")

if __name__ == '__main__':
    main()