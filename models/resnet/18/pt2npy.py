import torch
import numpy as np
import os
import json

# from datetime import datetime

# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 파일 경로
model_name = "resnet18_best_epoch47_20250612_182403.pt"

pt_path = f"./ex_models/{model_name}"
output_dir = f"./epoch_data/model_npy_{model_name}"

os.makedirs(output_dir, exist_ok=True)

state_dict = torch.load(pt_path, map_location="cpu")

shape_info = {}

# state_dict 로드
state_dict = torch.load(pt_path, map_location="cpu")

# 저장
for name, tensor in state_dict.items():
    npy_name = name.replace('.', '_') + ".npy"
    npy_path = os.path.join(output_dir, npy_name)

    # save data
    np.save(npy_path, tensor.cpu().numpy())

    # shape 저장
    shape_info[npy_name] = {
        "original_name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype)
    }

    print(f"✅ Saved: {npy_name} | shape={tuple(tensor.shape)}")

with open(os.path.join(output_dir,"0shapes.json"),"w") as f:
    json.dump(shape_info, f, indent =2 )

print("0Shape.json saved.")