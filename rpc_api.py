# rpc_api.py
import torch.distributed.rpc as rpc

from ps import get_weights as ps_get_weights
from ps import apply_gradients as ps_apply_gradients

def get_weights():
    return rpc.rpc_sync("ps", ps_get_weights)  # ✅ lambda 제거

def update_weights(grads):
    return rpc.rpc_sync("ps", ps_apply_gradients, args=(grads,))
