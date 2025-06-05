import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
# 
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
import platform
import time 
import os

device = ""

# root dir setting
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

# device setting
if platform.system() == "Darwin":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
elif platform.system() == "Linux":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compress_topk(grad_tensor, k_ratio=0.01):
    flat_grad = grad_tensor.view(-1)
    total_size = flat_grad.numel()
    k = max(1, int(k_ratio * total_size))

    # top-k 압축
    topk_vals, topk_indices = torch.topk(flat_grad.abs(), k)
    real_vals = flat_grad[topk_indices]

    # 🔽 크기 출력
    original_size_bytes = flat_grad.numel() * 4  # float32 = 4 bytes
    compressed_size_bytes = real_vals.numel() * 4 + topk_indices.numel() * 4  # 값 + 인덱스

    compression_ratio = compressed_size_bytes / original_size_bytes

    print(f"[Compression] Original: {original_size_bytes} bytes | Compressed: {compressed_size_bytes} bytes | Ratio: {compression_ratio:.2%}")

    return real_vals, topk_indices, grad_tensor.shape


def decompress_topk(real_vals, indices, shape):
    flat = torch.zeros(torch.prod(torch.tensor(shape)), device=real_vals.device)
    flat[indices] = real_vals
    return flat.view(shape)

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
    train_dataset = torchvision.datasets.CIFAR10(root=root_dir,train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root=root_dir,train=False, transform=transform, download= True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True,num_workers=2)

    print("Set up the model to resnet18")
    model = resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)

    print("Set up the criterion is CrossEntropyLoss")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9 ,weight_decay=5e-4)

    for epoch in range(num_epochs):
        start_time = time.time()
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
        end_time = time.time()

        elapsed_time = end_time - start_time
        print(f"Elapsed time {elapsed_time:.4f} seconds")

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
        
        com_start_time = time.time()
        # worker compress
        grads = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                values, indices, shape = compress_topk(param.grad, k_ratio=0.01)
                grads.append((name, values.cpu(), indices.cpu(), shape))  # 보내기 용도
        com_end_time = time.time()
        com_elapsed_time = com_end_time - com_start_time
        print(f"Com elapsed time {com_elapsed_time}")

        decom_start_time = time.time()
        # ps decompress
        for name, values, indices, shape in grads:
            param = model.state_dict()[name]
            decompressed_grad = decompress_topk(values.to(device), indices.to(device), shape)
            param.grad = decompressed_grad
        decom_end_time = time.time()
        decom_elapsed_time = decom_end_time - decom_start_time
        
        print(f"Decxom elapsed time {decom_elapsed_time}")

    torch.save(model.state_dict(),'resnet18_cifra10.pth')

if __name__ == '__main__':
    # windows and mac needs this 
    # why? -> spawn() runs the script and makes and child process
    main()