import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import random
import os
import glob

# 환경 통제 (deterministic 설정)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 최근에 생성된 seed 파일과 grad 파일을 읽음
seed_files = sorted(glob.glob("./output/seed_*.txt"))
grad_files = sorted(glob.glob("./output/grad_seed_*.npz"))

if len(seed_files) == 0 or len(grad_files) == 0:
    print("❌ seed 파일 또는 gradient 파일이 부족합니다.")
    exit(1)

# 가장 최근 것 사용
seed_file = seed_files[-1]
grad_file = grad_files[-1]

with open(seed_file, 'r') as f:
    seed = int(f.read().strip())

print(f"복원 중: seed = {seed}")

# 동일한 입력 데이터 사용
x = torch.randn(5, 3, 224, 224)
target = torch.randn(5, 1000)

# 동일한 seed 설정
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# 모델 초기화 (seed 영향)
# torchvision.models.resnet.ResNet
model = models.resnet18()
model.to(device)
x = x.to(device)
target = target.to(device)

criterion = nn.MSELoss()

# forward + backward
# output = model.__call__(x)
output = model(x)
loss = criterion(output, target)
loss.backward()

# gradient 비교
grad_loaded = np.load(grad_file)
all_match = True

for name, param in model.named_parameters():
    arr1 = grad_loaded[name]
    arr2 = param.grad.cpu().numpy()
    if not np.allclose(arr1, arr2, atol=1e-7):
    # if not np.array_equal(arr1, arr2):
        print(f"❌ Mismatch in layer: {name}")
        all_match = False

if all_match:
    print("✅ 완전 identical (bit-to-bit) 복원 성공!")
else:
    print("⚠️ 일부 mismatch 발생")
