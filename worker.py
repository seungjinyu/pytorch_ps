import torch
import torch.distributed.rpc as rpc
import argparse
import atexit
from model import SimpleNet
from rpc_api import get_weights, update_weights

def run_worker(rank, world_size):
    print(f"[INFO] Initializing RPC for Worker {rank}...")
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

    model = SimpleNet()

    for step in range(5):
        weights = get_weights()
        model.load_state_dict(weights)

        x = torch.randn(4, 10)
        y = torch.randn(4, 1)

        output = model(x)
        loss = ((output - y) ** 2).mean()

        model.zero_grad()
        loss.backward()

        grad_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_dict[name] = param.grad.cpu()

        print(f"[Worker {rank}] Sending gradients to PS (step {step+1})")
        update_weights(grad_dict)

    rpc.shutdown()

def graceful_shutdown():
    print("[INFO] Gracefully shutting down Worker RPC...")
    try:
        rpc.shutdown()
    except:
        pass

atexit.register(graceful_shutdown)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    args = parser.parse_args()
    run_worker(args.rank, args.world_size)
