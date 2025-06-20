import numpy as np
import os

dir1 = "./rasp3"
dir2 = "./rasp4"

# 두 디렉토리의 모든 npy 파일 목록 (정렬해서 순서 맞추기)
files1 = sorted([f for f in os.listdir(dir1) if f.endswith(".npy")])
files2 = sorted([f for f in os.listdir(dir2) if f.endswith(".npy")])

# 파일 수 확인
if files1 != files2:
    print("Warning: 파일 이름이 두 디렉토리에서 다릅니다!")
    print("rasp3:", files1)
    print("rasp4:", files2)
    exit(1)

# 비교
for fname in files1:
    path1 = os.path.join(dir1, fname)
    path2 = os.path.join(dir2, fname)
    arr1 = np.load(path1)
    arr2 = np.load(path2)

    exact_equal = np.array_equal(arr1, arr2)
    close_equal = np.equal(arr1, arr2, atol=1e-8)

    print(f"{fname}: array_equal={exact_equal}, allclose={close_equal}")
