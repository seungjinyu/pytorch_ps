# node_b_res.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import zmq
import pickle
import numpy as np

# ===== 랜덤 시드 고정 =====
torch.manual_seed(42)

# ===== ResNet18 정의 (CIFAR-10 수정) =====
def get_resnet18():
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model

model = get_resnet18()
model.train()

# ===== ZeroMQ 통신 설정 =====
context = zmq.Context()
receiver = context.socket(zmq.PULL)
receiver.bind("tcp://*:5555")  # A가 PUSH하는 곳

sender = context.socket(zmq.PUSH)
sender.connect("tcp://<NODE_A_IP>:5556")  # A가 PULL하는 곳

# ===== Gradient 수신 =====
print("📡 Waiting for gradient from Node A...")
raw_data = receiver.recv()
received_grads = pickle.loads(raw_data)
print("📥 Gradient received")

# ===== Optimizer 설정 =====
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ===== Gradient 적용 =====
for name, param in model.named_parameters():
    if name in received_grads:
        grad_tensor = torch.tensor(received_grads[name])
        param.grad = grad_tensor

optimizer.step()
optimizer.zero_grad()

# ===== 업데이트된 파라미터 전송 =====
state_to_send = {k: v.cpu() for k, v in model.state_dict().items()}
sender.send(pickle.dumps(state_to_send))
print("📤 Updated model sent to Node A")
