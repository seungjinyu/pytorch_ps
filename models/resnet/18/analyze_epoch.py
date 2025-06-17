import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict

# 전체 실험 루트
base_dir = os.path.join(os.getcwd(), "epoch_data")

# 모든 실험 디렉토리 찾기
experiment_dirs = [
    d for d in sorted(os.listdir(base_dir))
    if os.path.isdir(os.path.join(base_dir, d)) and
    any(name.startswith("epoch_") for name in os.listdir(os.path.join(base_dir, d)))
]

if not experiment_dirs:
    raise ValueError("epoch_ 디렉토리를 포함한 실험 디렉토리를 찾을 수 없습니다.")

for exp in experiment_dirs:
    exp_dir = os.path.join(base_dir, exp)
    output_dir = os.path.join(exp_dir, "param_diff")
    os.makedirs(output_dir, exist_ok=True)

    # epoch 디렉토리 정렬
    epoch_dirs = sorted(
        [d for d in os.listdir(exp_dir) if d.startswith("epoch_")],
        key=lambda x: int(x.split("_")[1])
    )

    param_names, grad_names, delta_names = set(), set(), set()
    for epoch in epoch_dirs:
        path = os.path.join(exp_dir, epoch)
        for file in os.listdir(path):
            if file.startswith("param_") and file.endswith(".npy"):
                param_names.add(file[len("param_"):-len(".npy")])
            elif file.startswith("grad_") and file.endswith(".npy"):
                grad_names.add(file[len("grad_"):-len(".npy")])
            elif file.startswith("delta_") and file.endswith(".npy"):
                delta_names.add(file[len("delta_"):-len(".npy")])

    def analyze_and_plot(name, prefix, save_prefix):
        history = []
        for i in range(1, len(epoch_dirs)):
            prev_path = os.path.join(exp_dir, f"epoch_{i-1}", f"{prefix}_{name}.npy")
            curr_path = os.path.join(exp_dir, f"epoch_{i}", f"{prefix}_{name}.npy")

            if os.path.exists(prev_path) and os.path.exists(curr_path):
                prev = np.load(prev_path)
                curr = np.load(curr_path)
                diff = curr - prev

                l2 = np.linalg.norm(diff)
                mad = np.mean(np.abs(diff))
                stats = (np.min(diff), np.max(diff), np.mean(diff), np.std(diff))
                history.append((i, l2, mad, *stats))

        if history:
            csv_path = os.path.join(output_dir, f"{save_prefix}_{name}.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Epoch", "L2_Distance", "MAD", "Min", "Max", "Mean", "Std"])
                writer.writerows(history)

            epochs, l2s, mads, *_ = zip(*history)
            plt.figure()
            plt.plot(epochs, l2s, label="L2 Distance")
            plt.plot(epochs, mads, label="MAD")
            plt.title(f"{save_prefix.upper()} Change: {name}")
            plt.xlabel("Epoch")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"{save_prefix}_{name}.png"))
            plt.close()

    for name in sorted(param_names):
        analyze_and_plot(name, "param", "param")
    for name in sorted(grad_names):
        analyze_and_plot(name, "grad", "grad")
    for name in sorted(delta_names):
        analyze_and_plot(name, "delta", "delta")

    # 그룹 분석
    shape_group_history = {
        "param": defaultdict(list),
        "grad": defaultdict(list),
        "delta": defaultdict(list)
    }

    def analyze_group(name, prefix):
        for i in range(1, len(epoch_dirs)):
            prev_path = os.path.join(exp_dir, f"epoch_{i-1}", f"{prefix}_{name}.npy")
            curr_path = os.path.join(exp_dir, f"epoch_{i}", f"{prefix}_{name}.npy")
            if os.path.exists(prev_path) and os.path.exists(curr_path):
                prev = np.load(prev_path)
                curr = np.load(curr_path)
                diff = curr - prev
                shape_str = str(diff.shape)
                l2 = np.linalg.norm(diff)
                mad = np.mean(np.abs(diff))
                shape_group_history[prefix][shape_str].append((i, l2, mad, name))

    for name in sorted(param_names):
        analyze_group(name, "param")
    for name in sorted(grad_names):
        analyze_group(name, "grad")
    for name in sorted(delta_names):
        analyze_group(name, "delta")

    group_dir = os.path.join(output_dir, "grouped")
    os.makedirs(group_dir, exist_ok=True)

    def plot_group(group_dict, prefix):
        for shape_str, records in group_dict.items():
            l2_map, mad_map = defaultdict(list), defaultdict(list)
            for epoch, l2, mad, name in records:
                l2_map[name].append((epoch, l2))
                mad_map[name].append((epoch, mad))

            shape_clean = shape_str.replace(", ", "_").replace("(", "").replace(")", "")
            # L2
            plt.figure()
            for name, points in l2_map.items():
                epochs, vals = zip(*points)
                plt.plot(epochs, vals, label=name)
            plt.title(f"{prefix.upper()} L2 (Shape {shape_str})")
            plt.xlabel("Epoch"); plt.ylabel("L2")
            plt.legend(fontsize='x-small'); plt.grid(); plt.tight_layout()
            plt.savefig(os.path.join(group_dir, f"{prefix}_group_l2_{shape_clean}.png"))
            plt.close()
            # MAD
            plt.figure()
            for name, points in mad_map.items():
                epochs, vals = zip(*points)
                plt.plot(epochs, vals, label=name)
            plt.title(f"{prefix.upper()} MAD (Shape {shape_str})")
            plt.xlabel("Epoch"); plt.ylabel("MAD")
            plt.legend(fontsize='x-small'); plt.grid(); plt.tight_layout()
            plt.savefig(os.path.join(group_dir, f"{prefix}_group_mad_{shape_clean}.png"))
            plt.close()

    plot_group(shape_group_history["param"], "param")
    plot_group(shape_group_history["grad"], "grad")
    plot_group(shape_group_history["delta"], "delta")

    print(f"[✓] 분석 및 시각화 완료: {exp_dir}")
