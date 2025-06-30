import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets

from torch.profiler import profile, ProfilerActivity
from datetime import datetime
import os

num_threads = 4

# === 설정 ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
trace_file = f"profile_trace_{timestamp}_t{num_threads}_cpu.json"

# 디바이스 설정: CPU만 사용
device = torch.device("cpu")
torch.manual_seed(0)
torch.set_num_threads(num_threads)

# 모델 준비
model = models.resnet50(weights=None).to(device)
model.train()

# 데이터셋 준비
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()
])
dataset = datasets.CIFAR10(root="../../data", train=True, transform=transform, download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# 손실 함수, 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# === Profiler 실행 (CPU만) ===
with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    with_stack=True,
    with_flops=True,
    profile_memory=True
) as prof:
    for epoch in range(5):
        print(f"🔁 Epoch {epoch+1}")
        for step, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.autograd.profiler.record_function("model_forward"):
                outputs = model(inputs)

            with torch.autograd.profiler.record_function("model_loss"):
                loss = criterion(outputs, labels)

            with torch.autograd.profiler.record_function("model_backward"):
                loss.backward()

            optimizer.step()
            prof.step()

            # 시간 절약: epoch당 10개 배치만 처리
            if step >= 99:
                break

# === JSON 저장 ===
prof.export_chrome_trace(trace_file)
print(f"✅ JSON trace saved to {trace_file}")
