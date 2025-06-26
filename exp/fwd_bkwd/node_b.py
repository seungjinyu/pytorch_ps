import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import zmq
import pickle
import io

# ======================
# 설정
# ======================
torch.manual_seed(0)
torch.set_num_threads(1)
device = torch.device("cpu")

# ======================
# 모델 및 옵티마이저 초기화
# ======================
model = models.mobilenet_v2(weights=None).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ======================
# ZeroMQ 소켓 설정
# ======================
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("🟢 Node B: 대기 중...")

while True:
    # ======================
    # 초기 파라미터 + gradient 수신
    # ======================
    msg_parts = socket.recv_multipart()
    init_bytes, grad_bytes = msg_parts

    # 초기 파라미터 로딩
    buffer = io.BytesIO(init_bytes)
    init_state = torch.load(buffer)
    model.load_state_dict(init_state)

    # gradient 역직렬화
    named_grads = pickle.loads(grad_bytes)
    print("📥 Node B: 초기화 + gradient 수신 완료")

    # ======================
    # .grad 수동 할당
    # ======================
    for name, p in model.named_parameters():
        if name in named_grads and named_grads[name] is not None:
            p.grad = torch.tensor(named_grads[name], dtype=torch.float32).to(device)
        else:
            p.grad = None

    # optimizer step
    optimizer.step()
    torch.save(model.state_dict(), "model_B.pt")

    # ======================
    # Node A로 모델 회신
    # ======================
    socket.send(pickle.dumps(model.state_dict()))
    print("📤 Node B: 모델 회신 완료\n")
