# node_a_res.py (seed 고정, init_state 없이 실행)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import zmq
import pickle

# ===== 랜덤 시드 고정 =====
torch.manual_seed(42)

# ===== CIFAR-10 설정 =====
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
train_dataset = datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True)

# ===== ResNet18 모델 정의 =====
def get_resnet18():
    model = models.resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # CIFAR-10용
    model.maxpool = nn.Identity()
    return model

model = get_resnet18()
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ===== ZeroMQ 통신 설정 =====
context = zmq.Context()
sender = context.socket(zmq.PUSH)
sender.connect("tcp://<NODE_B_IP>:5555")  # Node B 주소 입력

receiver = context.socket(zmq.PULL)
receiver.bind("tcp://*:5556")

# ===== Forward & Gradient 생성 =====
images, labels = next(iter(train_loader))
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()

# ===== Gradient 전송 =====
named_grads = {
    name: param.grad.cpu().numpy()
    for name, param in model.named_parameters()
    if param.grad is not None
}
sender.send(pickle.dumps(named_grads))
print("📤 Gradient sent")

# ===== 업데이트된 파라미터 수신 =====
new_state_dict = pickle.loads(receiver.recv())
print("📥 Updated model received")

model.load_state_dict(new_state_dict)

# ===== 파라미터 비교 =====
def compare_models(model_a, state_b):
    total, match = 0, 0
    for name, param in model_a.state_dict().items():
        total += 1
        if name not in state_b:
            continue
        if torch.allclose(param.cpu(), state_b[name], atol=1e-6):
            match += 1
    print(f"✅ MATCHED: {match}/{total} ({100 * match / total:.2f}%)")

compare_models(model, new_state_dict)
