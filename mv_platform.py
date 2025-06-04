import platform
import struct
import sys
import os

print(f"Python version: {platform.python_version()}")
print(f"Architecture: {platform.machine()}")
print(f"Interpreter: {platform.python_implementation()}")
print(f"Is 64-bit: {struct.calcsize('P') * 8 == 64}")
print(f"Executable path: {sys.executable}")
os.system(f"file {sys.executable}")
