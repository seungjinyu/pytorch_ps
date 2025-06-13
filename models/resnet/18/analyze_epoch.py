import numpy as np
import os
import math
from collections import Counter

def calculate_entropy(data: bytes) -> float:
    """바이트 단위로 엔트로피 계산"""
    if not data:
        return 0.0
    counter = Counter(data)
    length = len(data)
    return -sum((count / length) * math.log2(count / length) for count in counter.values())

def analyze_gradients_entropy(epoch_dir: str):
    print(f"📂 Analyzing gradients in: {epoch_dir}")
    for filename in os.listdir(epoch_dir):
        if filename.endswith(".npy"):
            path = os.path.join(epoch_dir, filename)
            tensor = np.load(path)
            entropy = calculate_entropy(tensor.tobytes())
            print(f"{filename:40s} | shape={tensor.shape} | entropy={entropy:.4f} bits/byte")

# 사용 예시
epoch_path = "epoch_data/epoch_38_20250612_174454"
analyze_gradients_entropy(epoch_path)
