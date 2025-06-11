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
import zlib, bz2, lzma, lz4.frame, zstandard as zstd, snappy
import csv
from datetime import datetime
from collections import defaultdict

# root dir setting
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = os.path.abspath(os.path.join(os.path.dirname(__file__), f"compression_results_{timestamp}.csv"))

# Write CSV header
with open(result_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Epoch", "Type", "Algorithm", "Original Size", "Compressed Size",
        "Compression Ratio", "Compress Time (s)", "Decompress Time (s)", "Delta Calc Time (s)"
    ])

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
    else:
        raise ValueError("Unknown compression algorithm")
    comp_time = time.time() - start
    return comp, comp_time

def decompress_and_measure(data: bytes, algorithm: str):
    start = time.time()
    if algorithm == "zlib":
        decomp = zlib.decompress(data)
    elif algorithm == "bz2":
        decomp = bz2.decompress(data)
    elif algorithm == "lzma":
        decomp = lzma.decompress(data)
    elif algorithm == "lz4":
        decomp = lz4.frame.decompress(data)
    elif algorithm == "zstd":
        decomp = zstd.ZstdDecompressor().decompress(data)
    elif algorithm == "snappy":
        decomp = snappy.decompress(data)
    else:
        raise ValueError("Unknown compression algorithm")
    decomp_time = time.time() - start
    return decomp, decomp_time

def main():
    batch_size = 128
    learning_rate = 0.01
    num_epochs = 3
    device = ru.setting_platform()
    algorithms = ["zlib", "bz2", "lzma", "lz4", "zstd", "snappy"]

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

    prev_params = {}

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

        compression_tasks = []
        delta_calc_times = {}
        total_original_size = 0

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad_np = param.grad.detach().cpu().numpy().astype(np.float32)
            grad_bytes = grad_np.tobytes()
            original_size = len(grad_bytes)
            total_original_size += original_size
            compression_tasks.append(("grad", grad_bytes, original_size))

            current = param.data.detach().cpu().numpy().astype(np.float32)

            start_delta = time.time()
            if name in prev_params:
                delta = current - prev_params[name]
            else:
                delta = current
            delta_time = time.time() - start_delta
            delta_calc_times[name] = delta_time

            prev_params[name] = current.copy()
            delta_bytes = delta.tobytes()
            compression_tasks.append(("delta", delta_bytes, len(delta_bytes)))

        epoch_results = {"grad": defaultdict(lambda: {"size": 0, "time": 0.0}),
                         "delta": defaultdict(lambda: {"size": 0, "time": 0.0})}

        for task, algo in [(task, algo) for task in compression_tasks for algo in algorithms]:
            kind, data_bytes, original_size = task
            try:
                comp_data, comp_time = compress_and_measure(data_bytes, algo)
                comp_size = len(comp_data)
                ratio = comp_size / original_size

                decomp_data, decomp_time = decompress_and_measure(comp_data, algo)
                assert decomp_data == data_bytes, f"[ERROR] Decompressed data mismatch with {algo} on {kind}"

                epoch_results[kind][algo]["size"] += comp_size
                epoch_results[kind][algo]["time"] += comp_time

                delta_time = ""
                if kind == "delta":
                    delta_time = f"{np.mean(list(delta_calc_times.values())):.6f}"

                with open(result_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1, kind, algo,
                        original_size, comp_size,
                        f"{ratio:.4f}", f"{comp_time:.4f}", f"{decomp_time:.4f}",
                        delta_time
                    ])
            except Exception as e:
                print(f"[WARNING] Compression/Decompression failed for {algo} on {kind}: {e}")
                continue

        print(f"\n[Epoch {epoch+1} Compression Summary] Total Original Size: {total_original_size} bytes")
        for kind in ["grad", "delta"]:
            print(f"\n>>> {kind.upper()} Compression:")
            for algo in algorithms:
                if algo in epoch_results[kind]:
                    stats = epoch_results[kind][algo]
                    ratio = stats["size"] / total_original_size
                    print(f"[{algo.upper()}] Total Compressed: {stats['size']} bytes | Ratio: {ratio:.2%} | Time: {stats['time']:.4f} sec")

if __name__ == '__main__':
    main()
