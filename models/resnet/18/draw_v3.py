import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 이미지 저장 폴더 설정
image_dir = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(image_dir, exist_ok=True)

# 최신 CSV 파일 탐색
data_dir = os.path.join(os.path.dirname(__file__), "data")
csv_files = [f for f in os.listdir(data_dir) if f.startswith("compression_results_") and f.endswith(".csv")]
latest_csv = max(csv_files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
csv_path = os.path.join(data_dir, latest_csv)
print(f"Loading: {csv_path}")

# CSV 불러오기
df = pd.read_csv(csv_path)

# 컬럼 이름 확인 및 필요한 열만 변환
expected_columns = [
    "Epoch", "Algorithm", "Original Size", "Compressed Size",
    "Compression Ratio", "Delta Time", "Compress Time",
    "Decompress Time", "Reconstruct Time", "Total Time"
]

missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    raise ValueError(f"CSV is missing columns: {missing_cols}")

# 타입 변환
for col in expected_columns[2:]:  # 숫자열만
    df[col] = df[col].astype(float)

# 알고리즘 목록
algorithms = df["Algorithm"].unique()

# 그래프 함수
def plot_metric(metric, ylabel, title, filename):
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        subset = df[df["Algorithm"] == algo]
        plt.plot(subset["Epoch"], subset[metric], marker='o', label=algo)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename_with_time = f"{timestamp}_{filename}"
    path = os.path.join(image_dir, filename_with_time)
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()

# 각각의 메트릭 그래프 저장
plot_metric("Compression Ratio", "Ratio", "Compression Ratio (Delta)", "delta_compression_ratio.png")
plot_metric("Delta Time", "Seconds", "Delta Calculation Time", "delta_time.png")
plot_metric("Compress Time", "Seconds", "Compression Time", "compression_time.png")
plot_metric("Decompress Time", "Seconds", "Decompression Time", "decompression_time.png")
plot_metric("Reconstruct Time", "Seconds", "Gradient Reconstruction Time", "reconstruct_time.png")
plot_metric("Total Time", "Seconds", "Total Time per Epoch", "total_time.png")
