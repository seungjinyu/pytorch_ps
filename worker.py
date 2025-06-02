import torch
import torch.distributed.rpc as rpc
import torch.nn.functional as F
import argparse
import atexit
from model import SimpleNet

def run_worker(rank, world_size):
    print(f"[INFO] Initializing RPC for Worker {rank}...")
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)

    ps_rref = rpc.rpc_sync("ps", lambda: rpc.get_rref_owner(rpc.RRef(lambda: None)))  # just trigger ps
    ps_rref = rpc.rpc_sync("ps", get_ps_rref)

    for step in range(5):
        # 1. Get weights
        weights = rpc.rpc_sync("ps", lambda ps: ps.get_weights(), args=(ps_rref,))
        model = SimpleNet()
        model.load_state_dict(weights)

        # 2. Train
        data = torch.randn(4, 10)
        target = torch.randn(4, 1)
        output = model(data)
        loss = F.mse_loss(output, target)

        model.zero_grad()
        loss.backward()
        grads = [p.grad for p in model.parameters()]

        # 3. Send gradients to PS
        rpc.rpc_sync("ps", lambda ps, g: ps.apply_gradients(g), args=(ps_rref, grads,))
        print(f"[Worker {rank}] Sent gradients to PS")

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
