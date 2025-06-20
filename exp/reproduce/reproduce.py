import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import random
import os
import datetime

# === 재현성 설정 ===
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === 모델 정의 ===
model = models.resnet18()  # torchvision 기본 resnet18
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# === 더미 데이터 생성 ===
x = torch.randn(5, 3, 224, 224, device=device)  # resnet18에 맞는 input
target = torch.randn(5, 1000, device=device)    # output 크기는 1000 클래스

# === Loss 및 학습 ===
criterion = nn.MSELoss()
output = model(x)
loss = criterion(output, target)
loss.backward()

# === Gradient 수집 ===
deterministic_grad = {}
for name, param in model.named_parameters():
    deterministic_grad[name] = param.grad.clone().cpu().numpy()

# === Timestamp 파일로 저장 ===
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "./deterministic_output"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, f"deterministic_grad_resnet_{timestamp}.npz")
np.savez_compressed(save_path, **deterministic_grad)

print(f"[Saved] Gradient saved to {save_path}")
