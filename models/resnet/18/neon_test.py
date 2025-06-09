import ctypes
import numpy as np
import time

# shared library 불러오기
lib = ctypes.CDLL('./libneon_abs.dylib')

# 함수 정의
abs_neon = lib.abs_neon
abs_neon.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
abs_neon.restype = None

# 테스트용 배열
size = 1000000
x = np.random.uniform(-100, 100, size).astype(np.float32)
y = np.empty_like(x)

# 실행
start = time.time()
abs_neon(x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),size)
end = time.time()

print(f"NEON abs completed in {end - start:.6f} seconds")
print(f"First 5 abs values: {y[:5]}")
