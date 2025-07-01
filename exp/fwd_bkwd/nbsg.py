import torch
import torch.nn as nn
import zmq, pickle

# 설정
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)
device = torch.device("cpu")

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")
print("🟢 Node B: waiting...")

loss_fn = nn.CrossEntropyLoss()

while True:
    raw = socket.recv()
    data = pickle.loads(raw)

    outputs = data["outputs"].to(device).detach().requires_grad_()
    labels = data["labels"].to(device)

    loss = loss_fn(outputs, labels)
    loss.backward()

    grad = outputs.grad.detach().cpu().numpy()
    socket.send(pickle.dumps({"outputs_grad": grad}))
    print("📤 Sent ∂L/∂outputs to Node A\n")
