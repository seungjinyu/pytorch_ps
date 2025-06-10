import csv
from collections import defaultdict
import glob
import os

def summarize_latest_compression_csv():
    # 최신 파일 검색
    csv_files = sorted(glob.glob("compression_results_*.csv"), reverse=True)
    if not csv_files:
        print("CSV file does not exist")
        return

    latest_file = csv_files[0]
    print(f"🔍 Recent csv file: {latest_file}")

    stats = {
        "grad": defaultdict(list),
        "delta": defaultdict(list)
    }

    with open(latest_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            kind = row["Type"]  # 'grad' or 'delta'
            algo = row["Algorithm"]
            ratio = float(row["Compression Ratio"])
            time_taken = float(row["Time (s)"])
            stats[kind][algo].append((ratio, time_taken))

    # 요약 출력
    for kind in ["grad", "delta"]:
        print(f"\n📦 {kind.upper()} Compression Average Result:")
        for algo, values in stats[kind].items():
            ratios, times = zip(*values)
            avg_ratio = sum(ratios) / len(ratios)
            avg_time = sum(times) / len(times)
            print(f"[{algo.upper()}] Average Compression Ratio: {avg_ratio:.2%} | Average Compression Time: {avg_time:.4f} sec")

if __name__ == "__main__":
    summarize_latest_compression_csv()
