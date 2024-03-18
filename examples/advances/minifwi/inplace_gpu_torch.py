import numpy as np
import torch
import timeit

def inplace_fn_torch(x):
    for _ in range(100):
        x = x + x * x + x * x * x
        x[0, 0] = 0
    return x
    
y = np.random.randn(1000, 1000).astype(dtype='float32')
y_torch = torch.tensor(y).cuda()

print("With inplace operations:")

tt = timeit.timeit(lambda: inplace_fn_torch(y_torch), number=10)
print(f"PyTorch: {tt * 1000} msec")

def noinplace_fn(x):
    for _ in range(100):
        x = x + x * x + x * x * x
    return x


print("Without inplace operations:")

tt = timeit.timeit(lambda: noinplace_fn(y_torch), number=10)
print(f"PyTorch: {tt * 1000} msec")
