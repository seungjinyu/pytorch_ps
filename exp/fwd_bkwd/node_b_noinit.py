import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import zmq
import pickle

# ======================
# 설정
# ======================
torch.manual_seed(0)
torch.set_num_threads(1)
device = torch.device("cpu")

# ======================
# 모델 및 옵티마이저 초기화 (자체 초기화)
# ======================
model = models.mobilenet_v2(weights=None).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
model.train()



# ======================
# ZeroMQ 소켓 설정
# ======================
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("🟢 Node B: 대기 중...")

while True:
    # gradient 수신
    # Node B: optimizer 상태도 동기화
    data = pickle.loads(socket.recv())
    named_grads = data["grads"]
    optimizer.load_state_dict(data["opt_state"])

    # grad_bytes = socket.recv()
    # named_grads = pickle.loads(grad_bytes)
    print("📥 Node B: gradient 수신 완료")

    # .grad 수동 할당
    for name, p in model.named_parameters():
        if name in named_grads and named_grads[name] is not None:
            p.grad = torch.tensor(named_grads[name], dtype=torch.float32).to(device)
        else:
            p.grad = None

    # optimizer step
    optimizer.step()
    torch.save(model.state_dict(), "model_B.pt")

    # Node A로 모델 회신
    socket.send(pickle.dumps(model.state_dict()))
    print("📤 Node B: 모델 회신 완료\n")
