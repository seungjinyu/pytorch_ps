# resnet18_zip_v3.py
import torch, os, csv, time
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
import numpy as np
import zlib, bz2, zstandard as zstd
from collections import defaultdict
from datetime import datetime
import resnet18_utils as ru
# Setup
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
epoch_dir = os.path.join(root_dir, "epoch_data")
os.makedirs(epoch_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_file = f"data/compression_results_{timestamp}.csv"
algorithms = ["zlib", "bz2", "zstd"]

# CSV Header
with open(result_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Algorithm", "Original Size", "Compressed Size", "Compression Ratio",
                     "Delta Time", "Compress Time", "Decompress Time", "Reconstruct Time", "Total Time"])

# Compress
def compress(data: bytes, algo: str):
    start = time.time()
    if algo == "zlib": c = zlib.compress(data)
    elif algo == "bz2": c = bz2.compress(data)
    elif algo == "zstd": c = zstd.ZstdCompressor().compress(data)
    else: raise ValueError()
    return c, time.time() - start

# Decompress
def decompress(data: bytes, algo: str):
    start = time.time()
    if algo == "zlib": d = zlib.decompress(data)
    elif algo == "bz2": d = bz2.decompress(data)
    elif algo == "zstd": d = zstd.ZstdDecompressor().decompress(data)
    else: raise ValueError()
    return d, time.time() - start

# Evaluate Accuracy
def evaluate(model, loader, device):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def compute_delta_bytes(current: np.ndarray , previous: np.ndarray) -> tuple[bytes, float]:
    """Compute delta and return its byte representation"""
    start = time.time()
    delta = np.subtract(current, previous,dtype = np.float32)
    delta_bytes = delta.tobytes()
    return delta_bytes, time.time() - start

# Main loop
def main():

    device  = ru.setting_platform()

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])
    train_ds = torchvision.datasets.CIFAR10(root=root_dir, train=True, download=True, transform=transform)
    test_ds  = torchvision.datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)

    indices = torch.randperm(len(train_ds))[:len(train_ds)//10]
    train_ds = Subset(train_ds, indices)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=128, shuffle=False)

    epochs = 31

    prev_params = {}
    best_acc = 0.0
    best_epoch = 0
    best_state_dict = None

    print("Starting training...")
    for epoch in range(1, epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, test_loader, device)

        if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
                best_state_dict = model.state_dict()
        total_size = 0
        epoch_metrics = defaultdict(lambda: {"total": 0, "comp": 0, "ratio": 0,
                                             "delta_t": 0, "comp_t": 0, "decomp_t": 0, "recon_t": 0, "count": 0})

        for name, param in model.named_parameters():
            if param.grad is None: continue
            grad = param.grad.detach().cpu().numpy().astype(np.float32)
            grad_bytes = grad.tobytes()
            original_size = len(grad_bytes)
            total_size += original_size

            for algo in algorithms:
                t0 = time.time()
                comp_bytes, comp_time = compress(grad_bytes, algo)
                decomp_bytes, decomp_time = decompress(comp_bytes, algo)
                t1 = time.time()

                assert decomp_bytes == grad_bytes
                ratio = len(comp_bytes) / original_size

                epoch_metrics[algo]["total"] += original_size
                epoch_metrics[algo]["comp"] += len(comp_bytes)
                epoch_metrics[algo]["comp_t"] += comp_time
                epoch_metrics[algo]["decomp_t"] += decomp_time
                epoch_metrics[algo]["count"] += 1

                if epoch > 1:
                    current = param.data.detach().cpu().numpy().astype(np.float32)
                    # start_delta = time.time()
                    # delta = current - prev_params[name]
                    # delta_bytes = delta.tobytes()
                    # delta_time = time.time() - start_delta
                    delta_bytes , delta_time = compute_delta_bytes(current, prev_params[name])
                    d_comp_bytes, d_comp_time = compress(delta_bytes, algo)
                    d_decomp_bytes, d_decomp_time = decompress(d_comp_bytes, algo)

                    start_recon = time.time()
                    recon_grad = np.frombuffer(d_decomp_bytes, dtype=np.float32).reshape(current.shape)
                    recon_time = time.time() - start_recon

                    assert recon_grad.shape == grad.shape
                    param.grad = torch.tensor(recon_grad, dtype=torch.float32).to(param.device)  # ✅ 핵심 반영


                    epoch_metrics[algo]["delta_t"] += delta_time
                    epoch_metrics[algo]["recon_t"] += recon_time

            prev_params[name] = param.data.detach().cpu().numpy().astype(np.float32)

        # Write result
        with open(result_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            for algo in algorithms:
                m = epoch_metrics[algo]
                if m["count"] == 0: continue
                ratio = m["comp"] / m["total"]
                writer.writerow([
                    epoch, algo, m["total"], m["comp"], f"{ratio:.4f}",
                    f"{m['delta_t']:.4f}", f"{m['comp_t']:.4f}",
                    f"{m['decomp_t']:.4f}", f"{m['recon_t']:.4f}",
                    f"{(m['delta_t']+m['comp_t']+m['decomp_t']+m['recon_t']):.4f}"
                ])

        print(f"[Epoch {epoch}] Accuracy: {acc:.2%} | Total: {total_size//1024} KB")

    export_path = os.path.join("ex_models", f"resnet18_best_epoch{best_epoch}_{timestamp}.pt")
    # torch.save(model.state_dict(), export_path)
    torch.save(best_state_dict,export_path)
    print(f"\n🏆 Best model saved from Epoch {best_epoch} ({best_acc:.2%}) → {export_path}")

    # model = resnet18(weights=None)
    # model.fc = nn.Linear(model.fc.in_features, 10)
    # model.load_state_dict(torch.load("data/resnet18_final_20250612_104023.pt"))
    # model.eval()


if __name__ == '__main__':
    main()
