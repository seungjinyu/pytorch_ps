from model import SimpleNet

# PS에 저장될 모델
model = SimpleNet()

def get_weights():
    return {k: v.cpu() for k, v in model.state_dict().items()}

def update_weights(new_weights):
    model.load_state_dict(new_weights)
    print("[PS] Weights updated.")
