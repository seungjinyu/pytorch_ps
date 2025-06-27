import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets

from torch.profiler import tensorboard_trace_handler
from torch.profiler import profile, record_function, ProfilerActivity
from datetime import datetime
import os

# ========== 환경 설정 ==========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"logdir_{timestamp}"
os.makedirs(log_dir, exist_ok=True)

torch.manual_seed(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

device = torch.device("cpu")

print(f"Running on {device}")

# ========== 모델 ==========
model = models.mobilenet_v2(weights=None).to(device)
model.train()

# ========== 데이터 ==========
transform = transforms.Compose([
    transforms.Resize(64),  # 원래는 224지만 실험 간소화
    transforms.ToTensor()
])
dataset = datasets.CIFAR10(root="../../data", train=True, transform=transform, download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

# ========== 손실 및 옵티마이저 ==========
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ========== 입력 1 배치 ==========
inputs, labels = next(iter(loader))
inputs, labels = inputs.to(device), labels.to(device)

# ========== 프로파일링 시작 ==========
with profile(
    activities=[ProfilerActivity.CPU],
    schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
    on_trace_ready=tensorboard_trace_handler(log_dir),
    record_shapes=True,
    with_stack=True
) as prof:

    for step, (inputs, labels) in enumerate(loader):
        if step >= 4:  # wait=0, warmup=1, active=2 → 최소 3 step 이상 필요
            break

        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with record_function("model_forward"):
            outputs = model(inputs)
        with record_function("model_loss"):
            loss = criterion(outputs, labels)
        with record_function("model_backward"):
            loss.backward()
        optimizer.step()

        prof.step()

print(f"✅ Profiling trace saved to: {log_dir}")
