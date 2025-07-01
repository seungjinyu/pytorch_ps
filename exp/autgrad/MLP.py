import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity


from datetime import datetime
import time

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
trace_file = f"profile_trace_simple_{timestamp}.json"


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28,128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128,10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = SimpleNN()
model.train()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


with profile(
    activities=[
        ProfilerActivity.CPU
    ],
    record_shapes=True,
    with_stack=True,
    profile_memory=True,
    with_flops=True
) as prof:
    for epoch in range(5):
        for step, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            with torch.autograd.profiler.record_function("model_forward"):
                outputs = model(images)
            with torch.autograd.profiler.record_function("model_loss"):
                loss = criterion(outputs, labels)
            with torch.autograd.profiler.record_function("model_backward"):
                loss.backward()
            optimizer.step()

            if step >10:
                break
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# model.eval()
# correct = 0
# total = 0

# with torch.no_grad():
#     for images, labels in test_loader:
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Accuracy: {100 * correct / total:.2f}%")

start_time = time.time()
prof.export_chrome_trace(trace_file)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

print(f"✅ JSON trace saved to {trace_file}")
