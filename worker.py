import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import os
from rpc_utils import no_op

ps_ref = None  # 전역으로 선언

def fetch_weights():
    return ps_ref.rpc_sync().get_weights()

def send_gradients(grads):
    return ps_ref.rpc_sync().apply_gradients(grads)

def run_worker():
    global ps_ref
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)
    ps_ref = rpc.remote("ps", no_op)

    x = torch.tensor([[1.0, 2.0]])
    y = torch.tensor([[1.0]])

    weights = rpc.rpc_sync("ps", fetch_weights)

    model = nn.Linear(2, 1)
    with torch.no_grad():
        for p, w in zip(model.parameters(), weights):
            p.copy_(w)

    pred = model(x)
    loss = nn.MSELoss()(pred, y)
    loss.backward()

    grads = [p.grad for p in model.parameters()]
    rpc.rpc_sync("ps", send_gradients, args=(grads,))

    print(f"Worker {rank} done.")
    rpc.shutdown()

if __name__ == "__main__":
    run_worker()
