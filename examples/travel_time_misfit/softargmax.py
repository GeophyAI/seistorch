import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

import sys
sys.path.append("../..")
from seistorch.signal import ricker_wave
from seistorch.show import SeisShow
from seistorch.signal import filter

# Define the parameters
fm = 10
dt = 0.001
nt = 1000
delay1 = 200
delay2 = 500

# Generate the wavelet
w1 = ricker_wave(fm, dt, nt, delay=delay1)
w2 = ricker_wave(fm, dt, nt, delay=delay2)
w2 = torch.diff(w2, prepend=torch.Tensor([0.]))

w1 = w1.numpy()
w2 = w2.numpy()

w1 = filter(np.expand_dims(w1, 0), dt, N=3, freqs=(2,8), axis=0, mode="bandpass")
w2 = filter(np.expand_dims(w2, 0), dt, N=3, freqs=(2,8), axis=0, mode="bandpass")

w1 = torch.from_numpy(w1[0])
w2 = torch.from_numpy(w2[0])

w1 = w1/torch.max(torch.abs(w1))
w2 = w2/torch.max(torch.abs(w2))

# Plot the wavelets
plt.figure(figsize=(8, 4))
plt.plot(w1, label="w1")
plt.plot(w2, label="w2")
plt.legend()
plt.show()

# plt.plot()

def argmax(x):
    return torch.argmax(x)

def differentiable_trvaletime_difference(input, beta=1):
    *_, n = input.shape
    input = F.softmax(beta * input, dim=-1)
    # plt.plot(input[0], label="Probability")
    # plt.legend()
    # plt.show()
    indices = torch.linspace(0, 1, n).to(input.device)
    result = torch.sum((n) * input * indices, dim=-1)
    return result      

cc = F.conv1d(w1.unsqueeze(0), w2.unsqueeze(0).unsqueeze(0), padding=nt-1)
plt.figure(figsize=(8, 4))
plt.plot(cc[0], label="cc")
plt.legend()
plt.show()

travel_time_diff_by_argmax = (argmax(cc)-nt+1)*dt
travel_time_diff_by_softmax = (differentiable_trvaletime_difference(cc)-nt)*dt

print(f"Travel time difference by argmax: {travel_time_diff_by_argmax}")
print(f"Travel time difference by softmax: {travel_time_diff_by_softmax}")


