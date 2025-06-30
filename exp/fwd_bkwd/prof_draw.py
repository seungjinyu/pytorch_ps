import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys

# ====== 경로 설정 ======
json_path = "profile_trace_20250627_163521.json"  # ← 여기 파일명을 맞춰줘

# ====== JSON 로드 ======
with open(json_path, "r") as f:
    raw = json.load(f)

# ====== 이벤트 추출 ======
events = raw["traceEvents"]
valid = [
    e for e in events
    if e.get("ph") == "X" and "ts" in e and "dur" in e
]

if not valid:
    raise ValueError("❌ No events with 'ph'='X' and both 'ts' and 'dur' found.")

# ====== DataFrame 구성 ======
df = pd.DataFrame(valid)
df["name"] = df["name"].astype(str)
df["ts"] = df["ts"].astype(float)
df["dur"] = df["dur"].astype(float)
df["start_ms"] = df["ts"] / 1000
df["dur_ms"] = df["dur"] / 1000
df["end_ms"] = df["start_ms"] + df["dur_ms"]

# 스레드별로 구분
df["tid"] = df["tid"].astype(str)
threads = sorted(df["tid"].unique())
tid_map = {tid: i for i, tid in enumerate(threads)}
df["thread_id"] = df["tid"].map(tid_map)

# ====== 시각화 ======
fig, ax = plt.subplots(figsize=(12, 6))

colors = plt.cm.tab20.colors
name_color = {}
y_offset = 0.4

for idx, row in df.iterrows():
    print(f"iter rows {idx}")
    name = row["name"]
    color = name_color.setdefault(name, colors[len(name_color) % len(colors)])
    ax.barh(
        y=row["thread_id"],
        width=row["dur_ms"],
        left=row["start_ms"],
        height=y_offset,
        color=color,
        edgecolor='black'
    )

print("58")
# Y축: 스레드
ax.set_yticks(list(tid_map.values()))
ax.set_yticklabels(list(tid_map.keys()))
ax.set_xlabel("Time (ms)")
ax.set_title("PyTorch Profiler Gantt Timeline")

# 범례
legend_patches = [
    mpatches.Patch(color=c, label=n)
    for n, c in name_color.items()
]
plt.legend(handles=legend_patches, loc="upper right", fontsize='small')

plt.tight_layout()
plt.show()
