# Exp on docker 

import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import random
import os
import datetime

# ==== 재현성 세팅 ====
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==== 데이터셋 경로 (volume mount된 data 폴더 사용 가능) ====
dataset_dir = "./data"
print(f"Using dataset directory: {dataset_dir}")

# 현재는 dummy input (실제 학습 데이터는 나중에 사용 가능)
model = models.resnet18()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Dummy 입력 데이터 생성
x = torch.randn(5, 3, 224, 224, device=device)
target = torch.randn(5, 1000, device=device)

# Forward + Backward
criterion = nn.MSELoss()
output = model(x)
loss = criterion(output, target)
loss.backward()

# Gradient 수집 및 저장
deterministic_grad = {name: param.grad.clone().cpu().numpy() for name, param in model.named_parameters()}

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "./deterministic_output"
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, f"deterministic_grad_resnet_{timestamp}.npz")
np.savez_compressed(save_path, **deterministic_grad)

print(f"[Saved] Gradient saved to {save_path}")
