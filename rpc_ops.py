# rpc_ops.py
def fetch_weights(ps_rref):
    return ps_rref.rpc_sync().get_weights()

def apply_gradients(ps_rref, grads):
    return ps_rref.rpc_sync().apply_gradients(grads)
