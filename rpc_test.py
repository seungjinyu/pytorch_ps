import os
import argparse
import time
import torch.distributed.rpc as rpc

def remote_add(x, y):
    print(f"[{rpc.get_worker_info().name}] remote_add called with: {x}, {y}")
    return x + y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--master_addr", type=str, default="localhost")
    parser.add_argument("--master_port", type=str, default="29500")
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=8)
    rpc.init_rpc(
        name=args.name,
        rank=args.rank,
        world_size=args.world_size,
        rpc_backend_options=options
    )
    print(f"[{args.name}] RPC initialized")

    if args.name == "worker1":
        # worker1 calls a function on worker0
        fut = rpc.rpc_async("worker0", remote_add, args=(3, 4))
        result = fut.wait()
        print(f"[{args.name}] Received result: {result}")

    # Wait to ensure all work finishes
    time.sleep(2)
    rpc.shutdown()

if __name__ == "__main__":
    main()
