import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet34
from torch.utils.data import DataLoader
import sys
import jutils

import time

logger = jutils.DualLogger("output_log.txt")
sys.stdout = logger

def count_parameters_and_gradients(model):
    total_params = 0
    total_grads = 0
    layer_idx = 1
    for name, param in model.named_parameters():
        num_params = param.numel()
        grad_params = param.grad.numel() if param.grad is not None else 0
        print(f"[{layer_idx} {name:<40}] shape={tuple(param.shape)}, params={num_params}, grad={grad_params}")
        layer_idx+=1
        total_params += num_params
        total_grads += grad_params
    param_size_mb = total_params * 4 / (1024 ** 2)
    grad_size_mb = total_grads * 4 / (1024 ** 2)
    print(f"[Epoch Summary] Total Parameters: {total_params}, {param_size_mb:.2f} MB")
    print(f"[Epoch Summary] Total Gradients : {total_grads}, {grad_size_mb:.2f} MB")

def compress_topk(grand_tensor, k_ratio=0.01):
    flat_grad = grand_tensor.view(-1)
    k = max(1,int(k_ratio *flat_grad.numel()))
    topk_vals, topk_indices = torch.topk(flat_grad.abs(),k)
    real_vals = flat_grad[topk_indices]
    return real_vals, topk_indices,grand_tensor.shape

def decompress_topk(real_vals, indices, shape):
    flat = torch.zeros(torch.prod(torch.tensor(shape)), device=real_vals.device)
    flat[indices] = real_vals
    return flat.view(shape)

def main():
    batch_size = 128
    learning_rate = 0.01
    num_epochs = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root="../../data", train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="../../data", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Set up the model to ResNet-34")
    model = resnet34(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)


    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")
        count_parameters_and_gradients(model)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed Time : {elapsed_time:.4f} seconds")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), 'resnet34_cifar10.pth')

if __name__ == '__main__':
    main()
    logger.log.close()


