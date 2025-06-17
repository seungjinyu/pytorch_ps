import pandas as pd
import matplotlib.pyplot as plt
import os

base_dir ="epoch_data"
csv_path = os.path.join("epoch_data", "multi_experiment_summary.csv")
df = pd.read_csv(csv_path)

# 카테고리 지정
df["Category"] = df["Layer"].apply(lambda x: "conv" if "conv" in x else ("bn" if "bn" in x else "other"))

# param 타입만 사용
param_df = df[df["Type"] == "param"].copy()

# 1. Top-10 L2
top_l2 = param_df.sort_values("Mean_L2", ascending=False).head(10)

# 2. 안정적인 레이어 (표준편차 기준)
stable_layers = param_df.sort_values("Std_L2").head(10)

# 3. 카테고리 그룹 평균
category_group = param_df.groupby("Category")[["Mean_L2", "Mean_MAD"]].mean(numeric_only=True).reset_index()

# 바차트 1: Top L2
plt.figure(figsize=(10, 4))
plt.bar(top_l2["Layer"], top_l2["Mean_L2"], color='steelblue')
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 Layers by Mean L2 (param)")
plt.tight_layout()
plt.savefig("top10_mean_l2_param.png")
plt.close()

# 바차트 2: Most Stable
plt.figure(figsize=(10, 4))
plt.bar(stable_layers["Layer"], stable_layers["Std_L2"], color='orange')
plt.xticks(rotation=45, ha='right')
plt.title("Top 10 Most Stable Layers (Std L2, param)")
plt.tight_layout()
plt.savefig("most_stable_layers.png")
plt.close()

# 바차트 3: Category 별 평균
plt.figure(figsize=(6, 4))
bar_width = 0.35
index = range(len(category_group))
plt.bar(index, category_group["Mean_L2"], bar_width, label='Mean_L2')
plt.bar([i + bar_width for i in index], category_group["Mean_MAD"], bar_width, label='Mean_MAD')
plt.xticks([i + bar_width / 2 for i in index], category_group["Category"])
plt.title("Mean L2 and MAD by Layer Category")
plt.legend()
plt.tight_layout()
plt.savefig("category_mean_l2_mad.png")
plt.close()

# 요약 CSV 저장
summary_csv_path = os.path.join("epoch_data", "summary_analysis_result.csv")
category_group.to_csv(summary_csv_path, index=False)


print(f"[✓] 분석 요약 저장 완료: {summary_csv_path}")

