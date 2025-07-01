import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import zmq, pickle

# 설정
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
device = torch.device("cpu")

# 모델 & 옵티마이저
model = models.mobilenet_v2(weights=None).to(device)
model.train()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 데이터
inputs = torch.randn(2, 3, 64, 64).to(device)
labels = torch.randint(0, 10, (2,)).to(device)

# forward
outputs = model(inputs)
outputs.retain_grad()
loss_fn = nn.CrossEntropyLoss()

# B에게 보낼 output만
payload = {
    "outputs": outputs.detach().cpu(),
    "labels": labels.cpu()
}

# ZeroMQ 송신
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://10.32.137.71:5555")
socket.send(pickle.dumps(payload))
print("📤 Sent outputs to Node B")

# A에서도 ∂L/∂outputs 계산
loss_a = loss_fn(outputs, labels)
loss_a.backward()
grad_a = outputs.grad.detach().clone().cpu()

# B로부터 받은 grad 수신
reply = socket.recv()
grad_b = pickle.loads(reply)["outputs_grad"]
grad_b_tensor = torch.tensor(grad_b)

# 1️⃣ gradient 비교
print("\n📊 Comparing ∂L/∂outputs")
if torch.allclose(grad_a, grad_b_tensor, atol=1e-6):
    print("✅ A와 B의 outputs gradient 일치")
else:
    diff = (grad_a - grad_b_tensor).abs()
    print(f"❌ Gradient mismatch, max diff: {diff.max():.4e}")

# 2️⃣ outputs.backward(grad_from_b) 수행 → parameter.grad 생성
model.zero_grad()
outputs = model(inputs)  # 다시 forward
outputs.backward(grad_b_tensor)

# 3️⃣ parameter update
optimizer.step()
print("✅ A: model parameters updated using B's gradient")
