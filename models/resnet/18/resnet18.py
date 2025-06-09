import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
import time
import os
import random
import resnet18_utils as ru
import ctypes
import numpy as np

# root dir setting
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))

# device setting
device = ru.setting_platform()

arm_check = ru.is_arm()
print(arm_check)
if arm_check:
    print("setting up dylib")
    dylib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "./libneon_abs.dylib"))
    neon_lib = ctypes.cdll.LoadLibrary(dylib_dir)
    neon_lib.abs_neon.argtypes = [np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),ctypes.c_int]

# compress with external dylib
def neon_abs_neon(input_tensor: torch.Tensor) -> torch.Tensor:
    # 반드시 flatten + float32 + contiguous + CPU 변환까지
    input_tensor = input_tensor.contiguous().view(-1).to("cpu").float()
    input_np = input_tensor.numpy()
    output_np = np.zeros_like(input_np, dtype=np.float32)

    neon_lib.abs_neon(
        input_np,
        output_np,
        ctypes.c_int(input_np.size)
    )
    return torch.from_numpy(output_np).to(input_tensor.device)


def compress_topk_neon(grad_tensor, k_ratio=0.01):
    flat_grad = grad_tensor.view(-1)
    abs_grad = neon_abs_neon(flat_grad)
    total_size = flat_grad.numel()
    k = max(1, int(k_ratio * total_size))

    topk_vals, topk_indices = torch.topk(abs_grad, k)
    real_vals = flat_grad[topk_indices]

    # original_size_bytes = total_size * 4
    # compressed_size_bytes = real_vals.numel() * 4 + topk_indices.numel() * 4
    # compression_ratio = compressed_size_bytes / original_size_bytes

    # print(f"[Compression] Original: {original_size_bytes} bytes | Compressed: {compressed_size_bytes} bytes | Ratio: {compression_ratio:.2%}")
    return real_vals, topk_indices, grad_tensor.shape



def main() :
    # setting parameters
    batch_size = 128
    learning_rate = 0.01
    num_epochs = 1
    print(f"Running on {device}")

    # CIFAR-10 for transform
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])

    # train dataset loading
    train_dataset = torchvision.datasets.CIFAR10(root=root_dir,train=True, transform=transform, download=True)

    # 1/10
    num_samples = len(train_dataset)
    random.seed(42)
    indices = random.sample(range(num_samples),num_samples // 10)
    train_dataset = Subset(train_dataset, indices)

    # test data loading
    test_dataset = torchvision.datasets.CIFAR10(root=root_dir,train=False, transform=transform, download= True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True,num_workers=0)

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
        ru.print_param_and_grad_stats(model)



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
            print(f"param.device: {param.device}")
            if param.grad is not None:
                values, indices, shape = ru.compress_topk(param.grad, k_ratio=0.01)
                grads.append((name, values.cpu(), indices.cpu(), shape))  # 보내기 용도
        com_end_time = time.time()
        com_elapsed_time = com_end_time - com_start_time
        print(f"\n[Com elapsed time {com_elapsed_time}]")
        ru.print_compressed_grad_stats(grads)

        if arm_check:
        # Neon compress
            neon_com_start_time = time.time()
            grads = []
            for name, param in model.named_parameters():
                print(f"param.device: {param.device}")
                if param.grad is not None:
                    values, indices, shape = compress_topk_neon(param.grad, k_ratio=0.01)
                    grads.append((name, values.cpu(), indices.cpu(), shape))  # 보내기 용도
            neon_com_end_time = time.time()
            neon_com_elapsed_time = neon_com_end_time - neon_com_start_time
            print(f"\n[Neon Com elapsed time {neon_com_elapsed_time}]")
            ru.print_compressed_grad_stats(grads)


        decom_start_time = time.time()
        # ps decompress
        for name, values, indices, shape in grads:
            param = model.state_dict()[name]
            decompressed_grad = ru.decompress_topk(values.to(device), indices.to(device), shape)
            param.grad = decompressed_grad
        decom_end_time = time.time()
        decom_elapsed_time = decom_end_time - decom_start_time
        print(f"Decom elapsed time {decom_elapsed_time}")



    # print("Saved the model into pth")
    # torch.save(model.state_dict(),'resnet18_cifra10.pth')

if __name__ == '__main__':
    # windows and mac needs this
    # why? -> spawn() runs the script and makes and child process
    main()
