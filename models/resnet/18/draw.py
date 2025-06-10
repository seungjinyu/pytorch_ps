import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSV 파일 로드 (로컬 경로에 맞게 수정)
file_path = "compression_results_20250609_205742.csv"
df = pd.read_csv(file_path)

# 데이터 전처리
df['Original Size'] = df['Original Size'].astype(int)
df['Compressed Size'] = df['Compressed Size'].astype(int)
df['Compression Ratio'] = df['Compression Ratio'].astype(float)
df['Time (s)'] = df['Time (s)'].astype(float)

# 알고리즘 및 타입 정의
algorithms = ['zlib', 'bz2', 'lzma', 'lz4', 'zstd', 'snappy', 'blosc']
types = ['grad', 'delta']
epochs = [1, 50]

# 1. 막대 차트: 알고리즘별 평균 압축률 (Epoch 1과 50)
for t in types:
    plt.figure(figsize=(10, 6))
    for epoch in epochs:
        data = df[(df['Type'] == t) & (df['Epoch'] == epoch)].groupby('Algorithm')['Compression Ratio'].mean()
        plt.bar([f"{alg} (E{epoch})" for alg in algorithms], data, label=f'Epoch {epoch}')
    plt.title(f'{t.capitalize()} Compression Ratio by Algorithm')
    plt.xlabel('Algorithm (Epoch)')
    plt.ylabel('Compression Ratio')
    for i, v in enumerate(data):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'images/{t}_compression_ratio_bar.png')
    plt.close()

# 2. 산점도: 압축률 vs 시간 (Epoch 50)
for t in types:
    plt.figure(figsize=(10, 6))
    data = df[(df['Type'] == t) & (df['Epoch'] == 50)].groupby('Algorithm').agg({'Compression Ratio': 'mean', 'Time (s)': 'mean'}).reset_index()
    plt.scatter(data['Time (s)'], data['Compression Ratio'])
    for i, row in data.iterrows():
        plt.annotate(row['Algorithm'], (row['Time (s)'], row['Compression Ratio']), xytext=(5, 5), textcoords='offset points')
    plt.title(f'{t.capitalize()} Compression Ratio vs Time (Epoch 50)')
    plt.xlabel('Time (s)')
    plt.ylabel('Compression Ratio')
    plt.tight_layout()
    plt.savefig(f'images/{t}_ratio_vs_time_scatter.png')
    plt.close()

# 3. 라인 차트: 42.95MB 근처 데이터의 에포크별 압축률 변화
for t in types:
    plt.figure(figsize=(10, 6))
    # 42.95MB 근처 (30,000,000 ~ 44,000,000) 데이터 필터링
    large_data = df[(df['Original Size'] >= 42900000) & (df['Original Size'] <= 43000000) & (df['Type'] == t)]

    if large_data.empty:
        print(f"[Warning] No data found for {t} near 42.95MB. Using all available data.")
        large_data = df[df['Type'] == t]  # fallback

    for alg in algorithms:
        data = large_data[large_data['Algorithm'] == alg].groupby('Epoch')['Compression Ratio'].mean()
        if not data.empty:
            plt.plot(data.index, data.values, marker='o', label=alg)
            for x, y in zip(data.index, data.values):
                plt.text(x, y + 0.01, f'{y:.3f}', ha='center')
        else:
            print(f"Warning: No data for algorithm {alg} in {t}")
    plt.title(f'{t.capitalize()} Compression Ratio Over Epochs (near 42.95MB)')
    plt.xlabel('Epoch')
    plt.ylabel('Compression Ratio')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'images/{t}_ratio_over_epochs_line.png')
    plt.close()

# 4. 파이 차트: 데이터 크기 분포
size_counts = df['Original Size'].value_counts()
total = size_counts.sum()
pie_data = size_counts.head(7)
plt.figure(figsize=(8, 8))
plt.pie(pie_data.values, labels=[f'{s/1024:.2f} KB' for s in pie_data.index], autopct='%1.1f%%')
plt.title('Original Size Distribution')
plt.tight_layout()
plt.savefig('size_distribution_pie.png')
plt.close()

print("Charts have been saved as PNG files.")