import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets

from torch.profiler import profile, ProfilerActivity
from datetime import datetime

import os, json



# === 설정 ===
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
trace_file = f"profile_trace_{timestamp}.json"

torch.manual_seed(0)



torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

# device = torch.device("cpu")
device = torch.device("cuda")


model = models.mobilenet_v2(weights=None).to(device)
model.train()

# ========== 데이터 ==========
transform = transforms.Compose([
    transforms.Resize(64),  # 원래는 224지만 실험 간소화
    transforms.ToTensor()
])
dataset = datasets.CIFAR10(root="../data", train=True, transform=transform, download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

inputs, labels = next(iter(loader))
inputs, labels = inputs.to(device), labels.to(device)

# === Profiler 실행 ===
with profile(
    activities=[ProfilerActivity.CPU],
    record_shapes=True,
    with_stack=True
) as prof:
    optimizer.zero_grad()
    with torch.autograd.profiler.record_function("model_forward"):
        outputs = model(inputs)
    with torch.autograd.profiler.record_function("model_loss"):
        loss = criterion(outputs, labels)
    with torch.autograd.profiler.record_function("model_backward"):
        loss.backward()
    optimizer.step()

# === JSON 저장 ===
prof.export_chrome_trace(trace_file)
print(f"✅ JSON trace saved to {trace_file}")
