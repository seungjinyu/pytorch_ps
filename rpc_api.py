from model import SimpleNet

# PS에 저장될 모델
model = SimpleNet()

def get_weights():
    print("[PS] Sending model weights PARAMETERS")
    for name, param in model.state_dict().items():
        print(f"[PS] Param: {name}\n{name}")
    return {k: v.cpu() for k, v in model.state_dict().items()}

def update_weights(new_weights):
    
    print("[PS] Received updated PARAMETER")

    for name, param in new_weights.items():
        print(f"[PS] Updated Param : {name}\n{param}")
    model.load_state_dict(new_weights)

    print("[PS] Weights updated.")
