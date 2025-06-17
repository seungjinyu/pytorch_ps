import os
import torch
import numpy as np

# 경로 설정
model_path = os.path.join("data", "lstm_model_weights.pth")
output_dir = os.path.join("data", "weights")
os.makedirs(output_dir, exist_ok=True)

# 모델 로드
state_dict = torch.load(model_path, map_location="cpu")

# 각 파라미터 저장
print(f"📦 모델에서 {len(state_dict)}개의 파라미터를 추출합니다.\n")
for name, tensor in state_dict.items():
    npy_path = os.path.join(output_dir, f"{name}.npy")
    np_val = tensor.cpu().numpy()
    np.save(npy_path, np_val)
    print(f"✅ 저장: {npy_path}  | shape: {np_val.shape}  | dtype: {np_val.dtype}")

print(f"\n🎉 모든 파라미터를 '{output_dir}' 디렉터리에 저장 완료.")
