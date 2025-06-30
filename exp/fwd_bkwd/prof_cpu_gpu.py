import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets

from torch.profiler import profile, ProfilerActivity
from datetime import datetime
import os

thread_num = 1

# === 설정 ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
trace_file = f"profile_trace_{timestamp}_t{thread_num}_gpu.json"

# 디바이스 설정 (GPU 있으면 GPU, 없으면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.set_num_threads(thread_num)

# 모델 준비
# model = models.mobilenet_v2(weights=None).to(device)
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

# 배치 하나 가져오기
inputs, labels = next(iter(loader))
inputs, labels = inputs.to(device), labels.to(device)



# === Profiler 실행 ===
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
    with_flops=True,
    profile_memory=True
) as prof:
    for epoch in range(5):
        print(f"Epoch {epoch+1}")
        for step, (inputs, labels) in enumerate(loader):
            inputs , labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autograd.profiler.record_function("model_forward"):
                outputs = model(inputs)
            with torch.autograd.profiler.record_function("model_loss"):
                loss = criterion(outputs, labels)
            with torch.autograd.profiler.record_function("model_backward"):
                loss.backward()
            optimizer.step()

            if step >= 99:
                break


# === JSON 저장 ===
prof.export_chrome_trace(trace_file)
print(f"✅ JSON trace saved to {trace_file}")
