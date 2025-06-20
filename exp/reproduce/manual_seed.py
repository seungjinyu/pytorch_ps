# exp for manual seed 

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] =":4096:8"

import torch 
import torch.nn as nn
import torchvision
# Image preprocessing 
import torchvision.transforms as transforms
import numpy as np
from datetime import datetime 

# Determinisitic config 
# random seed fixed
torch.manual_seed(42)
# True -> False Does not change the gradient
torch.use_deterministic_algorithms(True)
# True -> False does not change the gradient
torch.backends.cudnn.deterministic = True
# False -> True the gradient changes
torch.backends.cudnn.benchmark = False 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on {device}")

transform = transforms.Compose([
        transforms.ToTensor()
])

# Data 
trainset = torchvision.datasets.CIFAR10(root='./data', train=True , download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20,shuffle=False)

model = torchvision.models.resnet18()
model = model.to(device)
model.eval()

# Loss function
criterion = nn.CrossEntropyLoss()

dataiter = iter(trainloader)
images , labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# 
model.zero_grad()
# forward
outputs = model(images)

# loss function 
loss = criterion(outputs, labels)
# loss = criterion.__call__(outputs, labels)
# backward 
loss.backward()


# Timestamp 
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = os.path.join("./deterministic_grad", timestamp)
os.makedirs(save_dir, exist_ok=True)

for name, param in model.named_parameters():
    if param.grad is not None:
        grad = param.grad.detach().cpu().numpy()
        fname = os.path.join(save_dir, f"grad_{name.replace(',','_')}.npy")
        np.save(fname, grad)
        print(f"Saved: {fname}, shape: {grad.shape}")