import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets

import zmq 
import os 
import pickle
import numpy as np
import csv
from datetime import datetime

# ======================
# Function: Save gradient to CSV with timestamp
# ======================
def save_grad_to_csv(grad_tensor, prefix="grad_node"):
    os.makedirs("experiment_grads", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.csv"
    path = os.path.join("experiment_grads", filename)
    flat = grad_tensor.view(-1).cpu().numpy() if isinstance(grad_tensor, torch.Tensor) else grad_tensor.reshape(-1)
    with open(path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["grad_value"])
        for val in flat:
            writer.writerow([val])
    print(f"💾 Saved: {path}")
    return path

# ======================
# 환경 변수 및 ZeroMQ 연결
# ======================
NODE_B_IP = os.environ.get("NODE_B_IP", "localhost")

context = zmq.Context() 
socket = context.socket(zmq.REQ)
socket.connect(f"tcp://{NODE_B_IP}:5555")
print(f"🔗 Connected to Node B at {NODE_B_IP}:5555")

# ======================
# 기본 설정
# ======================
torch.manual_seed(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
device = torch.device("cpu")
print(f"🖥️  Running on {device}")

# ======================
# 모델 및 데이터 설정
# ======================
model = models.mobilenet_v2(weights=None).to(device)
model.train()

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()
])

# 데이터셋 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, "../../data"))

dataset = datasets.CIFAR10(root=data_dir, train=True, transform=transform, download=True)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

# ======================
# 입력 데이터 준비
# ======================
inputs, labels = next(iter(loader))
inputs, labels = inputs.to(device), labels.to(device)

# ======================
# 모델 Forward (detach해서 보냄)
# ======================
outputs = model(inputs)
outputs.retain_grad()

payload = {
    "outputs": outputs.detach().cpu(),  # graph는 제거된 상태
    "labels": labels.cpu()
}

serialized = pickle.dumps(payload)
print(f"📤 Node A: sending forward outputs... ({len(serialized)/1024:.2f} KB)")
socket.send(serialized)

# ======================
# Node B의 결과 수신 및 비교
# ======================
reply = socket.recv()
print("📥 Node A: received gradient from Node B")
data_b = pickle.loads(reply)
grad_b = torch.tensor(data_b["outputs_grad"])

# Node A에서 backward 수행
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(outputs, labels)
loss.backward()
grad_a = outputs.grad

# ======================
# CSV 저장
# ======================
save_grad_to_csv(grad_a, prefix="grad_node_a")
save_grad_to_csv(grad_b, prefix="grad_node_b")

# ======================
# Gradient 비교
# ======================
if torch.allclose(grad_a, grad_b, atol=1e-6):
    print("🎉 Gradient match!")
else:
    diff = torch.abs(grad_a - grad_b)
    print("❌ Gradient mismatch!")
    print(f"Max diff: {torch.max(diff):.4e}")
