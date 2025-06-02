import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
import time
import os
from rpc_utils import no_op

class ParameterServer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(2, 1)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def get_weights(self):
        return [p.detach().clone() for p in self.model.parameters()]

    def apply_gradients(self, grads):
        for param, grad in zip(self.model.parameters(), grads):
            param.grad = grad
        self.optimizer.step()
        self.optimizer.zero_grad()

ps_instance = None  # 전역 인스턴스 참조

def get_weights():
    return ps_instance.get_weights()

def apply_gradients(grads):
    return ps_instance.apply_gradients(grads)

def run_ps():
    global ps_instance
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    rpc.init_rpc("ps", rank=rank, world_size=world_size)

    ps_instance = ParameterServer()
    print("Parameter Server ready.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down PS.")
    finally:
        rpc.shutdown()

if __name__ == "__main__":
    run_ps()
