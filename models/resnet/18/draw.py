import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# 최신 CSV 자동 탐색
csv_candidates = glob.glob("compression_results_*.csv")
if not csv_candidates:
    raise FileNotFoundError("No compression_results_*.csv files found in the current directory.")
latest_csv = max(csv_candidates, key=os.path.getmtime)
print(f"[INFO] Using latest CSV file: {latest_csv}")
df = pd.read_csv(latest_csv)

# 열 이름 및 타입 정리
if "Compress Time (s)" not in df.columns and "Time (s)" in df.columns:
    df.rename(columns={"Time (s)": "Compress Time (s)"}, inplace=True)

df['Original Size'] = df['Original Size'].astype(int)
df['Compressed Size'] = df['Compressed Size'].astype(int)
df['Compression Ratio'] = df['Compression Ratio'].astype(float)
df['Compress Time (s)'] = pd.to_numeric(df['Compress Time (s)'], errors='coerce').fillna(0.0)
df['Delta Calc Time (s)'] = pd.to_numeric(df.get('Delta Calc Time (s)'), errors='coerce').fillna(0.0)
df['Total Time (s)'] = df['Compress Time (s)'] + df['Delta Calc Time (s)']

algorithms = df['Algorithm'].unique().tolist()
types = ['grad', 'delta']
epochs = sorted(df['Epoch'].unique())
# 1. 막대 차트: delta 알고리즘별 평균 압축률 (Epoch 1과 마지막)
for epoch in [epochs[0], epochs[-1]]:
    plt.figure(figsize=(10, 6))
    # delta만 필터링
    data = df[(df['Type'] == 'delta') & (df['Epoch'] == epoch)]
    grouped = data.groupby('Algorithm')['Compression Ratio'].mean()
    bars = plt.bar([f"{alg} (E{epoch})" for alg in grouped.index], grouped.values,
                   color=['orange' if epoch == epochs[-1] else 'steelblue'])

    plt.title(f"Delta Compression Ratio by Algorithm")
    plt.xlabel("Algorithm (Epoch)")
    plt.ylabel("Compression Ratio")
    plt.ylim(0, max(1.05, grouped.max() + 0.05))  # y축 범위 1.05 이상 확보

    # 압축률 표시
    for bar, ratio in zip(bars, grouped.values):
        height = bar.get_height()
        if ratio > 1.0:
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{ratio:.3f} ↑", ha='center', color='gray')
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f"{ratio:.3f}", ha='center')

    plt.legend([f"Epoch {epoch}"])
    plt.tight_layout()
    plt.savefig(f'images/delta_compression_ratio_bar_epoch{epoch}.png')
    plt.close()


# 2. 산점도: 압축률 vs Compress Time
for t in types:
    plt.figure(figsize=(10, 6))
    data = df[(df['Type'] == t) & (df['Epoch'] == epochs[-1])].groupby('Algorithm').agg({
        'Compression Ratio': 'mean',
        'Compress Time (s)': 'mean'
    }).reset_index()
    plt.scatter(data['Compress Time (s)'], data['Compression Ratio'])
    for i, row in data.iterrows():
        plt.annotate(row['Algorithm'], (row['Compress Time (s)'], row['Compression Ratio']), xytext=(5, 5), textcoords='offset points')
    plt.title(f'{t.capitalize()} Compression Ratio vs Compress Time (Epoch {epochs[-1]})')
    plt.xlabel('Compress Time (s)')
    plt.ylabel('Compression Ratio')
    plt.tight_layout()
    plt.savefig(f'images/{t}_ratio_vs_compress_time_scatter.png')
    plt.close()

# 3. 산점도: 압축률 vs Total Time
for t in types:
    plt.figure(figsize=(10, 6))
    data = df[(df['Type'] == t) & (df['Epoch'] == epochs[-1])].groupby('Algorithm').agg({
        'Compression Ratio': 'mean',
        'Total Time (s)': 'mean'
    }).reset_index()
    plt.scatter(data['Total Time (s)'], data['Compression Ratio'])
    for i, row in data.iterrows():
        plt.annotate(row['Algorithm'], (row['Total Time (s)'], row['Compression Ratio']), xytext=(5, 5), textcoords='offset points')
    plt.title(f'{t.capitalize()} Compression Ratio vs Total Time (Epoch {epochs[-1]})')
    plt.xlabel('Total Time (s)')
    plt.ylabel('Compression Ratio')
    plt.tight_layout()
    plt.savefig(f'images/{t}_ratio_vs_total_time_scatter.png')
    plt.close()

# 4. 에포크별 압축률 변화 (≥ 30MB 데이터만)
for t in types:
    plt.figure(figsize=(10, 6))
    large_data = df[(df['Original Size'] >= 30 * 1024 * 1024) & (df['Type'] == t)]

    if large_data.empty:
        print(f"[Warning] No large tensor data found for {t}. Using all data instead.")
        large_data = df[df['Type'] == t]

    for alg in algorithms:
        data = large_data[large_data['Algorithm'] == alg].groupby('Epoch')['Compression Ratio'].mean()
        if not data.empty:
            plt.plot(data.index, data.values, marker='o', label=alg)
            for x, y in zip(data.index, data.values):
                plt.text(x, y + 0.01, f'{y:.3f}', ha='center')
    plt.title(f'{t.capitalize()} Compression Ratio Over Epochs (≥ 30MB tensors)')
    plt.xlabel('Epoch')
    plt.ylabel('Compression Ratio')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'images/{t}_ratio_over_epochs_large_tensor_only.png')
    plt.close()

# 5. 에폭별 시간 분석 (Delta, Compression, Total)
epoch_summary = df.groupby('Epoch').agg({
    'Delta Calc Time (s)': 'mean',
    'Compress Time (s)': 'mean'
}).reset_index()
epoch_summary['Total Time (s)'] = epoch_summary['Compress Time (s)'] + epoch_summary['Delta Calc Time (s)']

fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axs[0].plot(epoch_summary['Epoch'], epoch_summary['Delta Calc Time (s)'], marker='o', color='tab:blue', label='Delta Time')
axs[0].set_title("Delta Time per Epoch")
axs[0].set_ylabel("Seconds")
axs[0].set_yscale('log')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(epoch_summary['Epoch'], epoch_summary['Compress Time (s)'], marker='o', color='tab:orange', label='Compress Time')
axs[1].set_title("Compress Time per Epoch")
axs[1].set_ylabel("Seconds")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(epoch_summary['Epoch'], epoch_summary['Total Time (s)'], marker='o', color='tab:green', label='Total Time')
axs[2].set_title("Total Time per Epoch")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("Seconds")
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig("images/epoch_time_breakdown_subplot.png")
plt.close()

# 6. 데이터 크기 분포 파이 차트
size_counts = df['Original Size'].value_counts()
pie_data = size_counts.head(7)
plt.figure(figsize=(8, 8))
plt.pie(pie_data.values, labels=[f'{s/1024:.2f} KB' for s in pie_data.index], autopct='%1.1f%%')
plt.title('Original Size Distribution')
plt.tight_layout()
plt.savefig('images/size_distribution_pie.png')
plt.close()

print("✅ All charts saved to 'images/' folder.")
