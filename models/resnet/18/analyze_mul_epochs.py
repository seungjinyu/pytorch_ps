import os
import numpy as np
import pandas as pd
from collections import defaultdict

base_dir = os.path.join(os.getcwd(), "epoch_data")

# 실험 디렉토리 수집 (예: 20250617_195508 등)
experiment_dirs = [
    os.path.join(base_dir, d) for d in sorted(os.listdir(base_dir))
    if os.path.isdir(os.path.join(base_dir, d))
]

# 결과 누적 딕셔너리
summary_data = {
    "Experiment": [],
    "Layer": [],
    "Type": [],
    "Mean_L2": [],
    "Std_L2": [],
    "Mean_MAD": [],
    "Std_MAD": []
}

prefixes = ["param", "grad", "delta"]

for exp_path in experiment_dirs:
    # epoch_* 디렉토리 탐색
    epoch_dirs = sorted([
        d for d in os.listdir(exp_path)
        if os.path.isdir(os.path.join(exp_path, d)) and d.startswith("epoch_")
    ], key=lambda x: int(x.split("_")[1]))

    if len(epoch_dirs) < 2:
        continue

    for prefix in prefixes:
        layer_stats = defaultdict(lambda: {"l2": [], "mad": []})

        for i in range(1, len(epoch_dirs)):
            prev_dir = os.path.join(exp_path, epoch_dirs[i - 1])
            curr_dir = os.path.join(exp_path, epoch_dirs[i])

            for file in os.listdir(curr_dir):
                if not file.startswith(f"{prefix}_") or not file.endswith(".npy"):
                    continue
                name = file[len(f"{prefix}_"):-len(".npy")]
                prev_file = os.path.join(prev_dir, f"{prefix}_{name}.npy")
                curr_file = os.path.join(curr_dir, f"{prefix}_{name}.npy")

                if not os.path.exists(prev_file):
                    continue

                prev = np.load(prev_file)
                curr = np.load(curr_file)
                diff = curr - prev
                l2 = np.linalg.norm(diff)
                mad = np.mean(np.abs(diff))

                layer_stats[name]["l2"].append(l2)
                layer_stats[name]["mad"].append(mad)

        for name, stats in layer_stats.items():
            summary_data["Experiment"].append(os.path.basename(exp_path))
            summary_data["Layer"].append(name)
            summary_data["Type"].append(prefix)
            summary_data["Mean_L2"].append(np.mean(stats["l2"]))
            summary_data["Std_L2"].append(np.std(stats["l2"]))
            summary_data["Mean_MAD"].append(np.mean(stats["mad"]))
            summary_data["Std_MAD"].append(np.std(stats["mad"]))

# CSV로 요약
df = pd.DataFrame(summary_data)
# DataFrame을 CSV로 저장
summary_csv = os.path.join(base_dir, "multi_experiment_summary.csv")
df.to_csv(summary_csv, index=False)

print(f"[✓] 분석 요약 저장 완료: {summary_csv}")
