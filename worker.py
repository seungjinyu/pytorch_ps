import torch
import torch.distributed.rpc as rpc
import torch.optim as optim
import atexit
import argparse
from model import SimpleNet
from rpc_api import get_weights, update_weights

def run_worker(rank, world_size):
    print(f"[INFO] Initializing RPC for Worker {rank}...")
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

    model = SimpleNet()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for i in range(5):
        weights = rpc.rpc_sync("ps", get_weights, args=())
        model.load_state_dict(weights)

        # fake training
        data = torch.randn(4, 10)
        target = torch.randn(4, 1)
        output = model(data)
        loss = ((output - target) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        rpc.rpc_sync("ps", update_weights, args=(model.state_dict(),))
        print(f"[Worker {rank}] Step {i+1} finished.")

    rpc.shutdown()

def graceful_shutdown():
    print("[INFO] Gracefully shutting down Worker RPC...")
    try:
        rpc.shutdown()
    except:
        print("[WARN] RPC shutdown error: RPC has not been initialized.")

atexit.register(graceful_shutdown)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    args = parser.parse_args()
    run_worker(args.rank, args.world_size)
