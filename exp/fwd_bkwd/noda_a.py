import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
import zmq
import pickle
import numpy as np
import io

# ======================
# 설정
# ======================
torch.manual_seed(0)
torch.set_num_threads(1)
device = torch.device("cpu")

# ======================
# 모델 및 옵티마이저
# ======================
model = models.mobilenet_v2(weights=None).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ======================
# CIFAR-10 데이터 (64x64 변환)
# ======================
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()
])
dataset = datasets.CIFAR10(root="../data", train=True, transform=transform, download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# 배치 하나 추출
inputs, labels = next(iter(loader))
inputs, labels = inputs.to(device), labels.to(device)

# ======================
# Forward + Backward
# ======================
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()

# ======================
# 초기 파라미터 저장 및 전송용 준비
# ======================
torch.save(model.state_dict(), "init_state.pt")
with open("init_state.pt", "rb") as f:
    init_bytes = f.read()

# ======================
# Gradient 추출 (이름 기준)
# ======================
named_grads = {
    name: p.grad.cpu().numpy() if p.grad is not None else None
    for name, p in model.named_parameters()
}

# Node A도 optimizer step 수행
optimizer.step()
torch.save(model.state_dict(), "model_A.pt")

# ======================
# ZeroMQ: init_state + gradient 전송
# ======================
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://10.32.137.71:5555")  # ← Node B IP 주소로 바꿔야 함

print("📤 Node A: 초기화 + gradient 전송 중...")
socket.send_multipart([
    init_bytes,
    pickle.dumps(named_grads)
])

# Node B로부터 업데이트된 모델 수신
reply = socket.recv()
state_dict_b = pickle.loads(reply)
print("📥 Node A: Node B 모델 수신 완료")

# ======================
# 모델 비교
# ======================
state_dict_a = model.state_dict()
total = 0
matched = 0
mismatched_keys = []

for key in state_dict_a:
    a_tensor = state_dict_a[key]
    b_tensor = state_dict_b[key]
    total += 1
    if torch.allclose(a_tensor, b_tensor, atol=1e-6):
        matched += 1
    else:
        mismatched_keys.append(key)

match_rate = matched / total * 100
print(f"\n✅ MATCHED: {matched}/{total} parameters ({match_rate:.2f}%)")

if mismatched_keys:
    print(f"❌ MISMATCHED KEYS ({len(mismatched_keys)}):")
    for key in mismatched_keys:
        print(f" - {key}")
else:
    print("🎉 All parameters match exactly!")
