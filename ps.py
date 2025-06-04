import torch
import torch.distributed.rpc as rpc
import torch.optim as optim
import atexit
from model import SimpleNet

# 🔄 PS에서 사용할 모델과 옵티마이저는 전역 변수로 유지
model = SimpleNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def get_weights():
    print("[PS] Sending weights...")
    return {k: v.cpu() for k, v in model.state_dict().items()}

def apply_gradients(grad_dict):
    print("[PS] Applying gradients...")
    for name, param in model.named_parameters():
        if name in grad_dict:
            param.grad = grad_dict[name]
    optimizer.step()
    optimizer.zero_grad()
    print("[PS] Weights updated.")

def run_ps():
    print("[INFO] Initializing RPC for Parameter Server...")
    rpc.init_rpc("ps", rank=0, world_size=3)
    print("[INFO] Parameter Server is running.")
    rpc.shutdown()

def graceful_shutdown():
    print("[INFO] Gracefully shutting down PS RPC...")
    try:
        rpc.shutdown()
    except:
        pass

atexit.register(graceful_shutdown)

if __name__ == "__main__":
    run_ps()
