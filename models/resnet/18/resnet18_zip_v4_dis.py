import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import zlib, bz2
import zstandard as zstd
import blosc2
import os, time, csv
from collections import defaultdict

# ───── 설정 ─────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = "epoch_weights"
result_dir = "compression_results"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# ───── 데이터셋 (CIFAR-10) ─────
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# ───── 모델 ─────
model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)  # CIFAR-10 용
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ───── 압축 유틸 ─────
def delta_encode(arr: np.ndarray) -> np.ndarray:
    return np.diff(arr, prepend=arr[0])

def compress_zstd(data: bytes) -> bytes:
    return zstd.ZstdCompressor(level=3).compress(data)

def compress_bz2(data: bytes) -> bytes:
    return bz2.compress(data)

def compress_zlib(data: bytes) -> bytes:
    return zlib.compress(data)

def compress_blosc2(data: np.ndarray) -> bytes:
    return blosc2.pack_array(data, cname="zstd", clevel=5, shuffle=blosc2.SHUFFLE)

def classify_and_compress(name, tensor):
    np_data = tensor.cpu().numpy().astype(np.float32)
    original_bytes = np_data.tobytes()
    original_size = len(original_bytes)

    if "conv" in name and tensor.dim() == 4:
        flat = np_data.flatten()
        delta = delta_encode(flat).astype(np.float32)
        compressed = compress_zstd(delta.tobytes())
        method = "delta+zstd"

    elif "bn" in name or ("running_" in name):
        quantized = (np_data * 256).astype(np.int16)
        compressed = compress_bz2(quantized.tobytes())
        method = "quantize+bz2"

    elif "bias" in name:
        compressed = compress_zlib(original_bytes)
        method = "zlib"

    elif "fc" in name:
        compressed = compress_zstd(original_bytes)
        method = "zstd"

    else:
        compressed = compress_blosc2(np_data)
        method = "blosc2"

    compressed_size = len(compressed)
    compression_ratio = compressed_size / original_size
    return {
        "name": name,
        "method": method,
        "original_size": original_size,
        "compressed_size": compressed_size,
        "compression_ratio": compression_ratio,
    }

# ───── 학습 + 압축 루프 ─────
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 평가
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    acc = 100. * correct / total
    print(f"🎯 Epoch {epoch} Test Accuracy: {acc:.2f}%")

    # 저장
    ckpt_path = os.path.join(save_dir, f"epoch_{epoch}.pt")
    torch.save(model.state_dict(), ckpt_path)

    # 압축
    state_dict = model.state_dict()
    results = []
    for name, tensor in state_dict.items():
        try:
            start = time.time()
            result = classify_and_compress(name, tensor)
            result["time"] = time.time() - start
            result["epoch"] = epoch
            results.append(result)
            print(f"[✓] Epoch {epoch} - {name}: {result['method']} | Ratio: {result['compression_ratio']:.3f}")
        except Exception as e:
            print(f"[✗] {name}: Error - {e}")

    # 저장
    csv_path = os.path.join(result_dir, f"compression_epoch{epoch}.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "name", "method", "original_size", "compressed_size", "compression_ratio", "time"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"📦 Epoch {epoch} compression results saved to: {csv_path}\n")
