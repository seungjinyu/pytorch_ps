import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import queue
import time

# Hyperparams
SIGN_FLIP_THRESHOLD = 0.1  # 10% sign flip threshold
ABS_DELTA_THRESHOLD = 0.01  # absolute delta threshold
EPOCHS = 5
BATCH_SIZE = 64
LR = 0.01
STEPS_PER_EPOCH = 100
NUM_WORKERS = 3

# Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16*16*16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv(x)))
        x = x.view(-1, 16*16*16)
        x = self.fc(x)
        return x

# Generate synthetic random data
def get_batch():
    inputs = torch.randn(BATCH_SIZE, 3, 32, 32)
    labels = torch.randint(0, 10, (BATCH_SIZE,))
    return inputs, labels

def compute_gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

# Worker process
def worker(worker_id, ps_request_q, ps_response_q):
    model = SimpleCNN()
    optimizer = optim.SGD(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        for step in range(STEPS_PER_EPOCH):
            inputs, labels = get_batch()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            grad_norm = compute_gradient_norm(model)
            ps_request_q.put((worker_id, grad_norm))

            try:
                response = ps_response_q.get(timeout=5)
            except queue.Empty:
                print(f"[Worker {worker_id}] No response from PS")
                continue

            if response == "SEND_FULL":
                full_grad = [p.grad.data.clone() for p in model.parameters()]
                ps_request_q.put((worker_id, full_grad))
            else:
                pass  # Skip sending full gradient

            optimizer.step()

# Parameter Server process
def parameter_server(ps_request_q, ps_response_q):
    reference_dict = {}
    stats = {}

    while True:
        try:
            msg = ps_request_q.get(timeout=10)
        except queue.Empty:
            print("[PS] No more messages. Exiting.")
            break

        worker_id, content = msg

        if isinstance(content, float):
            if worker_id not in reference_dict:
                reference_dict[worker_id] = None
                stats[worker_id] = {"total": 0, "skipped": 0, "full_sent": 0}
                ps_response_q.put("SEND_FULL")
            else:
                ps_response_q.put("SEND_FULL")
        elif isinstance(content, list):
            stats[worker_id]["total"] += 1
            if reference_dict[worker_id] is None:
                reference_dict[worker_id] = content
                print(f"[PS] Worker {worker_id}: Initial full gradient stored.")
                stats[worker_id]["full_sent"] += 1
                continue

            prev_grad = reference_dict[worker_id]
            curr_grad = content

            total_elements = 0
            sign_flips = 0
            abs_deltas = 0

            for prev_tensor, curr_tensor in zip(prev_grad, curr_grad):
                prev_flat = prev_tensor.view(-1)
                curr_flat = curr_tensor.view(-1)
                total_elements += prev_flat.numel()

                sign_flips += torch.sum(torch.sign(prev_flat) != torch.sign(curr_flat)).item()
                abs_deltas += torch.sum(torch.abs(prev_flat - curr_flat) > ABS_DELTA_THRESHOLD).item()

            flip_ratio = sign_flips / total_elements
            abs_delta_ratio = abs_deltas / total_elements

            print(f"[PS] Worker {worker_id}: Flip ratio: {flip_ratio:.3f}, Abs delta ratio: {abs_delta_ratio:.3f}")

            if flip_ratio > SIGN_FLIP_THRESHOLD or abs_delta_ratio > SIGN_FLIP_THRESHOLD:
                ps_response_q.put("SEND_FULL")
                reference_dict[worker_id] = curr_grad
                stats[worker_id]["full_sent"] += 1
                print(f"[PS] Worker {worker_id}: Significant change, updated reference.")
            else:
                ps_response_q.put("SKIP")
                stats[worker_id]["skipped"] += 1
        else:
            print("[PS] Unknown message type")

    # Print statistics after all workers finished
    print("\n=== Statistics ===")
    for wid, s in stats.items():
        print(f"Worker {wid}: Total={s['total']}, Full Sent={s['full_sent']}, Skipped={s['skipped']}")

def main():
    mp.set_start_method("spawn")
    ps_request_q = mp.Queue()
    ps_response_q = mp.Queue()

    ps_proc = mp.Process(target=parameter_server, args=(ps_request_q, ps_response_q))
    ps_proc.start()
    time.sleep(1)

    workers = []
    for worker_id in range(NUM_WORKERS):
        p = mp.Process(target=worker, args=(worker_id, ps_request_q, ps_response_q))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    ps_proc.join()

if __name__ == "__main__":
    main()
