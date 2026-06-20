import sys
import os
import time

print("Starting imports...")

start = time.time()
import torch
print(f"Imported torch in {time.time() - start:.2f}s")

start = time.time()
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
print(f"Imported matplotlib in {time.time() - start:.2f}s")

start = time.time()
sys.path.append(os.path.abspath(r'C:\Users\Shrish\Desktop\semantic-comm\actual_project'))
from src.model import SemanticEncoder
print(f"Imported src.model in {time.time() - start:.2f}s")

print("All imports successful.")
