import os, numpy as np, time, csv
import bz2, zlib
import zstandard as zstd
from collections import defaultdict

base_dir = os.path.join(os.getcwd(), "epoch_data")
target_epochs = ["epoch_0", "epoch_49", "epoch_99"]  # 🎯 압축할 epoch만 선택

compression_log = []

# 실험 디렉토리 순회
for exp in sorted(os.listdir(base_dir)):

    print(f"Folder {exp}")
    exp_dir = os.path.join(base_dir, exp)

    if not os.path.isdir(exp_dir):
        continue

    for epoch in target_epochs:
        print(f"Currently on {epoch}")
        epoch_path = os.path.join(exp_dir, epoch)
        if not os.path.exists(epoch_path):
            continue

        zip_dir = os.path.join(epoch_path, "zip")
        os.makedirs(zip_dir, exist_ok=True)

        for prefix in ["grad", "delta"]:
            shape_groups = defaultdict(list)
            shape_bytes = defaultdict(int)

            for fname in os.listdir(epoch_path):
                if fname.startswith(f"{prefix}_") and fname.endswith(".npy"):
                    fpath = os.path.join(epoch_path, fname)
                    arr = np.load(fpath)
                    shape = str(arr.shape)
                    shape_groups[shape].append(arr)
                    shape_bytes[shape] += arr.nbytes

            for shape_str, arrays in shape_groups.items():
                stack = np.stack(arrays)
                data_bytes = stack.tobytes()
                total_raw_bytes = shape_bytes[shape_str]

                for algo in ["bz2", "zstd", "zlib"]:
                    start = time.time()

                    if algo == "bz2":
                        compressed = bz2.compress(data_bytes)
                        ext = ".bz2"
                    elif algo == "zstd":
                        cctx = zstd.ZstdCompressor(level=3)
                        compressed = cctx.compress(data_bytes)
                        ext = ".zst"
                    elif algo == "zlib":
                        compressed = zlib.compress(data_bytes, level=6)
                        ext = ".zlib"

                    elapsed = time.time() - start
                    shape_tag = shape_str.replace("(", "").replace(")", "").replace(", ", "x").replace(",", "x")
                    out_path = os.path.join(zip_dir, f"{prefix}_shape_{shape_tag}_compressed{ext}")
                    with open(out_path, "wb") as f:
                        f.write(compressed)

                    compression_log.append({
                        "Type": prefix,
                        "Experiment": exp,
                        "Epoch": epoch,
                        "Shape": shape_str,
                        "Algorithm": algo,
                        "NumArrays": len(arrays),
                        "OriginalBytes": total_raw_bytes,
                        "CompressedBytes": len(compressed),
                        "CompressionRatio": len(compressed) / total_raw_bytes,
                        "CompressionTimeSec": elapsed
                    })

# CSV 저장
log_path = os.path.join(base_dir, "partial_compression_summary.csv")
with open(log_path, "w", newline="") as csvfile:
    fieldnames = [
        "Type", "Experiment", "Epoch", "Shape", "Algorithm", "NumArrays",
        "OriginalBytes", "CompressedBytes",
        "CompressionRatio", "CompressionTimeSec"
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in compression_log:
        writer.writerow(row)

print(f"[✓] 선택한 epoch 압축 완료. 요약 CSV: {log_path}")
