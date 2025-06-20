import numpy as np
import os
import glob

# 저장된 gradient 파일 경로
output_dir = "./deterministic_output"
files = sorted(glob.glob(os.path.join(output_dir, "deterministic_grad_resnet_*.npz")))

if len(files) < 2:
    print("❌ 최소 2개의 gradient 파일이 필요합니다.")
    exit(1)

# 가장 최근 2개 선택
file1, file2 = files[-2], files[-1]

print(f"비교 중: \n - {file1} \n - {file2}")

grad1 = np.load(file1)
grad2 = np.load(file2)

all_match = True
mismatch_count = 0

for key in grad1.files:
    arr1 = grad1[key]
    arr2 = grad2[key]
    if not np.allclose(arr1, arr2, atol=1e-7):
        print(f"❌ Mismatch in layer: {key}")
        mismatch_count += 1
        all_match = False

if all_match:
    print("✅ 모든 gradient가 동일합니다.")
else:
    print(f"총 {mismatch_count}개의 레이어에서 mismatch 발생")
