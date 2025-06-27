import torch
from torchvision import models

model = models.mobilenet_v2(weights=None)
all_named = set(dict(model.named_parameters()).keys()) | set(dict(model.named_buffers()).keys())

# mismatch 목록을 불러온다고 가정
with open("mismatch_log.txt") as f:
    mismatched = set()
    for line in f:
        if "]:" in line:
            parts = line.split("]:")[-1]
            mismatched |= set(map(str.strip, parts.split(',')))

matching = sorted(all_named - mismatched)

# 출력
for name in matching:
    print(name)

