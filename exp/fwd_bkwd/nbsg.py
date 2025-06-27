import torch
import torch.nn as nn
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
# ZeroMQ 설정
# ======================
BIND_IP = os.environ.get("NODE_B_BIND_IP", "*")
PORT = 5555

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind(f"tcp://{BIND_IP}:{PORT}")
print(f"🟢 Node B: Listening at tcp://{BIND_IP}:{PORT}")

# ======================
# 기본 설정
# ======================
torch.manual_seed(0)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
device = torch.device("cpu")

model = nn.Identity()  # 실제 모델 필요 없음
loss_fn = nn.CrossEntropyLoss()

while True:
    # ======= 요청 수신 =======
    raw = socket.recv()
    print("📥 Received input from Node A")
    data = pickle.loads(raw)

    outputs = data["outputs"].to(device).detach().requires_grad_()
    labels = data["labels"].to(device)

    # ======= Backward =======
    loss = loss_fn(outputs, labels)
    loss.backward()

    grad = outputs.grad.detach().cpu()

    # ======= 저장 =======
    save_grad_to_csv(grad, prefix="grad_node_b")

    # ======= 회신 =======
    response = {
        "outputs_grad": grad.numpy()
    }
    socket.send(pickle.dumps(response))
    print("📤 Sent gradient back to Node A\n")
