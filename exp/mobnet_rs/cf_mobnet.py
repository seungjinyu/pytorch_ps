import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import time
import os
import tracemalloc
from torch.profiler import profile, record_function, ProfilerActivity

# =======================
# 시스템 정보 수집 함수들
# =======================

def get_thread_count():
    with open("/proc/self/status", "r") as f:
        for line in f:
            if line.startswith("Threads:"):
                return int(line.split()[1])
    return -1

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_str = f.readline()
            return int(temp_str) / 1000.0
    except:
        return -1.0

# =======================
# 설정 및 데이터 로딩
# =======================

num_thread = int(os.environ.get("NUM_THREAD", 2))
torch.set_num_threads(num_thread)
torch.set_num_interop_threads(1)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(weights=None, num_classes=10)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

tracemalloc.start()
os.makedirs("log", exist_ok=True)

log_path = "log/train_log_with_temp.csv"
with open(log_path, "w") as logf:
    logf.write("Step,Loss,Threads,CPU_Temp,Mem_Current_MB,Mem_Peak_MB,Time_ms\n")

    model.train()
    print(f"Using device: {device}, Threads: {num_thread}")
    print("Training...")

    try:
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True
        ) as prof:

            for i, (inputs, labels) in enumerate(train_loader):
                t0 = time.time()
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with record_function("forward"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                with record_function("backward"):
                    loss.backward()

                optimizer.step()
                t1 = time.time()

                # 시스템 정보 수집
                current, peak = tracemalloc.get_traced_memory()
                thread_count = get_thread_count()
                cpu_temp = get_cpu_temp()

                logf.write(f"{i},{loss.item():.4f},{thread_count},{cpu_temp:.1f},{current/1024/1024:.2f},{peak/1024/1024:.2f},{(t1-t0)*1000:.2f}\n")

                print(f"[Step {i}] Loss: {loss.item():.4f} | Threads: {thread_count} | Temp: {cpu_temp:.1f}°C | Mem: {current/1024/1024:.2f} MB")

                if i >= 100:
                    break

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("OOM detected!")
            with open("log/error.log", "w") as errf:
                errf.write(f"OOM at step {i}: {str(e)}\n")
        else:
            raise

print("Training complete.")
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
