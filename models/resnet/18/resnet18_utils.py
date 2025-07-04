import platform
import os
import torch
import ctypes
import numpy as np
import zlib
import bz2
import zstandard as zstd
import time


def is_arm():
    machine = platform.machine().lower()
    return machine == "arm64" or machine == "aarch64"

# setting platform set ups the gpu by the platform
def setting_platform():
    # device setting
    if platform.system() == "Darwin":
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        return device
    elif platform.system() == "Linux":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device
    else:
        device = torch.device("cpu")
        return device

def print_param_and_grad_stats(model):
    total_params = 0
    total_grads = 0
    total_param_bytes = 0
    total_grad_bytes = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
        if param.grad is not None:
            total_grads += param.grad.numel()

    print(f"\n[Epoch Summary]")
    print(f"Total Parameters: {total_params:,} - {total_params * 4/ 1e6:.2f} MB")
    print(f"Total Gradients : {total_grads:,} - {total_grads * 4 /1e6:.2f} MB")

def print_compressed_grad_stats(compressed_grads):
    print("\n[Compressed Gradient Summary]")
    total_vals = 0
    total_indices = 0
    total_bytes = 0

    # for name, values, indices, shape in compressed_grads:
    #     num_vals = values.numel()
    #     num_indices = indices.numel()
    #     size_vals = num_vals * 4 / 1024 / 1024  # float32
    #     size_indices = num_indices * 4 / 1024 / 1024  # int32
    #     size_total = size_vals + size_indices

    #     total_vals += num_vals
    #     total_indices += num_indices
    #     total_bytes += size_total

    #     print(f"{name:<40} shape: {shape} - values: {size_vals:.2f} MB, indices: {size_indices:.2f} MB, total: {size_total:.2f} MB")

    print(f"\n[Compressed Total]")
    print(f"Values: {total_vals:,}, Indices: {total_indices:,}, Size: {total_bytes:.2f} MB")

# pytorch topk compression
def compress_topk(grad_tensor, k_ratio=0.01):
    flat_grad = grad_tensor.view(-1)
    total_size = flat_grad.numel()
    k = max(1, int(k_ratio * total_size))

    # top-k 압축
    topk_vals, topk_indices = torch.topk(flat_grad.abs(), k)
    real_vals = flat_grad[topk_indices]

    # original_size_bytes = flat_grad.numel() * 4  # float32 = 4 bytes
    # compressed_size_bytes = real_vals.numel() * 4 + topk_indices.numel() * 4  # 값 + 인덱스

    # compression_ratio = compressed_size_bytes / original_size_bytes

    # print(f"[Compression] Original: {original_size_bytes} bytes | Compressed: {compressed_size_bytes} bytes | Ratio: {compression_ratio:.2%}")

    return real_vals, topk_indices, grad_tensor.shape

# decompress_topk
def decompress_topk(real_vals, indices, shape):
    flat = torch.zeros(torch.prod(torch.tensor(shape)),device=real_vals.device)
    flat[indices] = real_vals
    return flat.view(shape)

# Compress
def compress(data: bytes, algo: str):
    start = time.time()
    if algo == "zlib": c = zlib.compress(data)
    elif algo == "bz2": c = bz2.compress(data)
    elif algo == "zstd": c = zstd.ZstdCompressor().compress(data)
    else: raise ValueError()
    return c, time.time() - start

# Decompress
def decompress(data: bytes, algo: str):
    start = time.time()
    if algo == "zlib": d = zlib.decompress(data)
    elif algo == "bz2": d = bz2.decompress(data)
    elif algo == "zstd": d = zstd.ZstdDecompressor().decompress(data)
    else: raise ValueError()
    return d, time.time() - start
