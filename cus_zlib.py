import zlib
import numpy as np
import torch

tensor = torch.rand(1000)
byte_data = tensor.numpy().astype(np.float32).tobytes()

compressed = zlib.compress(byte_data)
decompressed = zlib.decompress(compressed)
restored = torch.from_numpy(np.frombuffer(decompressed, dtype= np.float32))

print("복원 정확도:", torch.allclose(tensor, restored))

original_size = len(byte_data)
compressed_size = len(compressed)

compression_rate = compressed_size / original_size

print("원본 크기 :", original_size)
print("압축 크기 :", compressed_size)
print("압축률:", compression_rate)

print(torch.allclose(tensor, restored))


tensor = torch.tensor([1.0]*10 + [2.0]*10, dtype=torch.float32)
data = tensor.numpy().tobytes()
man_compressed = zlib.compress(data)

print(f"원본: {len(data)} bytes")        # 80 bytes
print(f"압축 후: {len(man_compressed)} bytes")  # 15 bytes 등
