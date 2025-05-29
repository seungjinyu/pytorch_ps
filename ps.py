import torch.distributed.rpc as rpc
import time

def start_ps():
    rpc.init_rpc("ps", rank=0, world_size=3)
    print("Parameter Server started")

    try:
        while True:
            print("Getting request from the worker")
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down PS...")

    rpc.shutdown()


if __name__ == "__main__":
    start_ps()
