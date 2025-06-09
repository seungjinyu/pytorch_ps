import torch
import zlib, bz2, lzma
import lz4.frame
import snappy
import zstandard as zstd
import blosc
import time

tensor = torch.randn(100_000_000, dtype=torch.float32)
tensor_bytes = tensor.numpy().tobytes()
original_size = len(tensor_bytes)

print(f"Original Size : {original_size} bytes \n")

compressors = {
    "zlib": lambda d: zlib.compress(d),
    "bz2": lambda d: bz2.compress(d),
    "lzma": lambda d: lzma.compress(d),
    "lz4": lambda d: lz4.frame.compress(d),
    "zstd": lambda d: zstd.ZstdCompressor().compress(d),
    "snappy": lambda d: snappy.compress(d),
    "blosc": lambda d: blosc.compress(d, typesize=4),
}

print(f"{'Algorithm':8} | {'Size (bytes)':>12} | {'Ratio':>6} | {'Time (s)':>8}")
print("-" * 44)

for name, func in compressors.items():
    try:
        start = time.time()
        compressed = func(tensor_bytes)
        end = time.time()
        compressed_size = len(compressed)
        ratio = compressed_size / original_size
        elapsed = end - start
        print(f"{name:8} | {compressed_size:12} | {ratio:.2%} | {elapsed:.4f}")
    except Exception as e:
        print(f"{name:8} | {'ERROR':>12} |{'--':>6} | {str(e)}")
