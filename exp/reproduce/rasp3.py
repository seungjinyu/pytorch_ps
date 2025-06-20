import torch

# 고정된 random seed + deterministic mode
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

# 아주 간단한 1-layer 모델
model = torch.nn.Linear(10, 2)

# 고정된 입력과 타겟
inputs = torch.randn(3, 10)
labels = torch.tensor([0, 1, 0])

# Loss
criterion = torch.nn.CrossEntropyLoss()

# Forward + backward
model.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()

# Gradient 확인
for name, param in model.named_parameters():
    print(f"{name} grad:\n{param.grad}\n")
