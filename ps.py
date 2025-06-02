import torch
import torch.distributed.rpc as rpc
import torch.nn as nn
import torch.optim as optim
import atexit

from model import SimpleNet  # 입력: 10, 출력: 1인 모델

class ParameterServer:
    def __init__(self):
        self.model = SimpleNet()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def apply_gradients(self, grads):
        for p, g in zip(self.model.parameters(), grads):
            p.grad = g
        self.optimizer.step()
        self.optimizer.zero_grad()
        print("[PS] Applied gradients and updated parameters.")

    def get_weights(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

ps_rref = None

def run_ps():
    global ps_rref
    print("[INFO] Initializing RPC for Parameter Server...")
    rpc.init_rpc("ps", rank=0, world_size=3)
    ps_rref = rpc.RRef(ParameterServer())
    print("[PS] Parameter Server is running.")
    rpc.shutdown()

def get_ps_rref():
    return ps_rref

def graceful_shutdown():
    print("[INFO] Gracefully shutting down PS RPC...")
    try:
        rpc.shutdown()
    except:
        pass

atexit.register(graceful_shutdown)

if __name__ == "__main__":
    run_ps()
