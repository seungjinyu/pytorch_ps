import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import random
import os
import time

# 환경 통제 (deterministic 설정)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 고정 입력 데이터 (입력은 항상 동일하게 유지)
x = torch.randn(5, 3, 224, 224)
target = torch.randn(5, 1000)

# 랜덤하게 seed 생성
seed = int(time.time() * 1000) % 100000
print(f"Using seed: {seed}")

# seed 저장
os.makedirs("output", exist_ok=True)
with open(f"output/seed_{seed}.txt", "w") as f:
    f.write(str(seed))

# 랜덤 seed 적용
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 모델 초기화 (seed 영향 받음)
model = models.resnet18()
model.to(device)
x = x.to(device)
target = target.to(device)

# 손실함수
criterion = nn.MSELoss()

# forward + backward
output = model(x)
loss = criterion(output, target)
loss.backward()

# gradient 저장
grad = {name: param.grad.clone().cpu().numpy() for name, param in model.named_parameters()}

grad_filename = f"output/grad_seed_{seed}.npz"
np.savez_compressed(grad_filename, **grad)
print(f"✅ Gradient saved to {grad_filename}")
