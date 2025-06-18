import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import numpy as np
import os
from datetime import datetime

# ───── 설정 ─────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
base_epoch_dir = os.path.join(os.path.dirname(__file__), "epoch_data")

# 날짜/시간 기반 실험 디렉토리 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = os.path.join(base_epoch_dir, timestamp)
os.makedirs(experiment_dir, exist_ok=True)

# ───── 데이터셋 (10% CIFAR-10) ─────
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

full_trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
trainset = torch.utils.data.Subset(full_trainset, range(len(full_trainset) // 10))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# ───── 모델 ─────
model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 10)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ───── 이전 파라미터 저장용 ─────
prev_params = {}

# ───── 학습 + 저장 루프 ─────
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # ───── 저장 경로 설정 ─────
    epoch_path = os.path.join(experiment_dir, f"epoch_{epoch}")
    os.makedirs(epoch_path, exist_ok=True)

    # ───── 파라미터별 저장 ─────
    for name, param in model.named_parameters():
        param_np = param.data.cpu().numpy()
        grad_np = param.grad.cpu().numpy() if param.grad is not None else None

        np.save(os.path.join(epoch_path, f"param_{name}.npy"), param_np)
        if grad_np is not None:
            np.save(os.path.join(epoch_path, f"grad_{name}.npy"), grad_np)

        if epoch > 0 and name in prev_params:
            delta = param_np - prev_params[name]
            np.save(os.path.join(epoch_path, f"delta_{name}.npy"), delta)

        prev_params[name] = param_np.copy()

    print(f"[✓] Epoch {epoch} saved at: {epoch_path}")
