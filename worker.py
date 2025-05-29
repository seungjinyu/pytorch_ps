import argparse
import torch
import torch.distributed.rpc as rpc
import time

def train():
    # 여기에 실제 모델 학습 or RPC 호출 구현 가능
    while True:
        result = rpc.rpc_sync("ps", torch.add, args=(torch.tensor(1), torch.tensor(2)))
        print(f"[Worker] Got result from PS: {result}")
        time.sleep(1)

def start_worker(rank):
    rpc.init_rpc(f"worker{rank}", rank=rank, world_size=3)
    train()
    rpc.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    start_worker(args.rank)
