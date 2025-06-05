import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
# 
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import platform


device = ""

if platform.system() == "Darwin":
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
elif platform.system() == "Linux":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_param_and_grad_stats(model):
    total_params = 0
    total_grads = 0
    total_param_bytes = 0
    total_grad_bytes = 0

    for name, param in model.named_parameters():
        numel = param.numel()
        grad_numel = param.grad.numel() if param.grad is not None else 0
        total_params += numel
        total_grads += grad_numel

        param_bytes = numel * 4 / 1024 / 1024  # float32 = 4 bytes
        grad_bytes = grad_numel * 4 / 1024 / 1024

        total_param_bytes += param_bytes
        total_grad_bytes += grad_bytes

        print(f"{name:<40} {str(param.shape):<25} - param: {param_bytes:.2f} MB, grad: {grad_bytes:.2f} MB")

    print(f"\n[Epoch Summary]")
    print(f"Total Parameters: {total_params:,} - {total_param_bytes:.2f} MB")
    print(f"Total Gradients : {total_grads:,} - {total_grad_bytes:.2f} MB")



def main() :
    # setting parameters
    batch_size = 128
    learning_rate = 0.01
    num_epochs = 1
    print(f"Running on {platform.system()} and using {device}")

    # CIFAR-10 for transform
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    # dataset loading 
    train_dataset = torchvision.datasets.CIFAR10(root="../data",train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root="../data",train=False, transform=transform, download= True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True,num_workers=2)

    print("Set up the model to resnet18")
    model = resnet18(weights= None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    print("Set up the criterion is CrossEntropyLoss")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9 ,weight_decay=5e-4)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs= model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss:{running_loss/len(train_loader):.4f}")

        print_param_and_grad_stats(model)

        model.eval()
        correct , total = 0,0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, prediected = outputs.max(1)
                total += targets.size(0)
                correct += prediected.eq(targets).sum().item()
        print(f"Test Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(),'resnet18_cifra10.pth')

if __name__ == '__main__':
    # windows and mac needs this 
    # why? -> spawn() runs the script and makes and child process
    main()