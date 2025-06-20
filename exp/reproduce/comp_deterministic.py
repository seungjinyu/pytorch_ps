import numpy as np
import os
import glob

# gradient 저장 폴더
output_dir = "./deterministic_output"
files = sorted(glob.glob(os.path.join(output_dir, "deterministic_grad_resnet_*.npz")))

if len(files) < 2:
    print("❌ 최소 2개의 gradient 파일이 필요합니다.")
    exit(1)

file1, file2 = files[-2], files[-1]

print(f"비교 중: \n - {file1} \n - {file2}")

grad1 = np.load(file1)
grad2 = np.load(file2)

mismatch_layers = []
identical_layers = []

total_mismatch_bytes = 0
total_identical_bytes = 0




for key in grad1.files:
    arr1 = grad1[key]
    arr2 = grad2[key]
    size_bytes = arr1.size * 4  # float32 = 4 bytes

    # np.array_equal(arr1, arr2)
    if not np.allclose(arr1, arr2, atol=1e-05):
        mismatch_layers.append((key, arr1.shape, size_bytes))
        total_mismatch_bytes += size_bytes
    else:
        identical_layers.append((key, arr1.shape, size_bytes))
        total_identical_bytes += size_bytes

# 요약 출력
print("\n=== MISMATCH LAYERS ===")
for name, shape, size in mismatch_layers:
    print(f"{name:40s}  shape: {shape}  size: {size/1024/1024:.3f} MB")

print("\n=== IDENTICAL LAYERS ===")
for name, shape, size in identical_layers:
    print(f"{name:40s}  shape: {shape}  size: {size/1024/1024:.3f} MB")

print("\n=== SUMMARY ===")
print(f"Mismatch Layers: {len(mismatch_layers)}")
print(f"Identical Layers: {len(identical_layers)}")
print(f"Mismatch Total Size: {total_mismatch_bytes/1024/1024:.3f} MB")
print(f"Identical Total Size: {total_identical_bytes/1024/1024:.3f} MB")
print(f"Total Gradients Size: {(total_mismatch_bytes+total_identical_bytes)/1024/1024:.3f} MB")
