import os, numpy as np, time, csv
import bz2, zlib
import zstandard as zstd
from collections import defaultdict

base_dir = os.path.join(os.getcwd(), "epoch_data")
experiments = sorted([
    d for d in os.listdir(base_dir)
    if os.path.isdir(os.path.join(base_dir, d)) and
       any(name.startswith("epoch_") for name in os.listdir(os.path.join(base_dir, d)))
])

compression_log = []

for exp in experiments:
    exp_dir = os.path.join(base_dir, exp)
    epoch_dirs = sorted(
        [d for d in os.listdir(exp_dir) if d.startswith("epoch_")],
        key=lambda x: int(x.split("_")[1])
    )

    for epoch in epoch_dirs:
        epoch_path = os.path.join(exp_dir, epoch)
        zip_dir = os.path.join(epoch_path, "zip")
        os.makedirs(zip_dir, exist_ok=True)
        
        print(f"Zipping Epoch {epoch}")

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
log_path = os.path.join(base_dir, "bn_compression_summary.csv")
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

print(f"[✓] grad/delta 전체 shape 기반 압축 완료. 요약 CSV: {log_path}")
