# utils.py
import os
import argparse
import torch.distributed.rpc as rpc

def init_rpc_with_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--master_addr", type=str, default="ps")
    parser.add_argument("--master_port", type=str, default="29500")
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=16)
    rpc.init_rpc(
        name="ps" if args.role == "ps" else f"worker{args.rank}",
        rank=args.rank,
        world_size=args.world_size,
        rpc_backend_options=options,
    )

    print("Initialization finished")
    return args
