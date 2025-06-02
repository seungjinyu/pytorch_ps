import torch.distributed.rpc as rpc
import atexit
from rpc_api import get_weights, update_weights

def run_ps():
    print("[INFO] Initializing RPC for Parameter Server...")
    rpc.init_rpc("ps", rank=0, world_size=3)
    print("[INFO] Parameter Server is ready.")
    rpc.shutdown()

def graceful_shutdown():
    print("[INFO] Gracefully shutting down Parameter Server RPC...")
    try:
        rpc.shutdown()
    except:
        print("[WARN] RPC shutdown error: RPC has not been initialized.")

atexit.register(graceful_shutdown)

if __name__ == "__main__":
    run_ps()
