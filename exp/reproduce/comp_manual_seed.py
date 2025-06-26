import os
import numpy as np

# 기본 경로
base_dir = "./deterministic_grad"

# timestamp 폴더들 정렬
experiments = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

if len(experiments) < 2:
    raise ValueError("비교할 실험 폴더가 2개 이상 필요합니다!")

# 기준: 첫번째 실험을 기준으로 모든 다른 실험과 비교
ref_dir = os.path.join(base_dir, experiments[0])

for target_exp in experiments[1:]:
    print(f"\nComparing [{experiments[0]}] vs [{target_exp}]")
    target_dir = os.path.join(base_dir, target_exp)
    diffnum = 0
    for fname in os.listdir(ref_dir):
        if not fname.endswith(".npy"):
            continue

        ref_path = os.path.join(ref_dir, fname)
        tgt_path = os.path.join(target_dir, fname)

        if not os.path.exists(tgt_path):
            print(f"Missing file in target: {fname}")
            continue

        ref_arr = np.load(ref_path)
        tgt_arr = np.load(tgt_path)
        
        if np.allclose(ref_arr, tgt_arr, atol=1e-7):
        # if np.array_equal(ref_arr, tgt_arr):
            continue
        else:    
            diffnum+=1
            diff = np.abs(ref_arr - tgt_arr)
            max_diff = diff.max()
            status = f"⚠ DIFFERENT (max diff={max_diff:.2e})"
            print(f"{fname}: {status}")
    if diffnum == 0:
        print("EE is the SAME")    
    else:
        print("Some gradients changed")
        
        
        
        
