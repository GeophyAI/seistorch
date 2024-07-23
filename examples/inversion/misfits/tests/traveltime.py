import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics.pairwise import cosine_similarity
import time

save_path = r"traveltime"
import os
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Func for Ricker
def ricker_wavelet(t, f):
    return (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)

# Func for calculating travel time
def calculate_travetime(signal1, signal2, dt=0.002):
    nt = signal2.size
    return (np.argmax(np.convolve(signal1, signal2))-nt)*dt

# Generate a Ricker
t = np.linspace(-1, 1, 1000, endpoint=False)
frequency = 10 
original_signal = ricker_wavelet(t, frequency)

fig, axes = plt.subplots(1,2, figsize=(6,4))
ax = axes[0]
line_original, = ax.plot(t, original_signal, label='Original Signal', color='blue')
line_delayed, = ax.plot(t, original_signal, label='Delayed Signal', color='red', linestyle='dashed')
# cs, = axes[1].scatter([], [], label='Cosine Similarity', color='green')
ax.legend()
DELAY = []
LOSS = []
def update(delay):
    # Delay the signal
    delayed_signal = np.roll(original_signal, delay)

    # Calculate the travel time between the original and delayed signals
    traveltime = calculate_travetime(original_signal, delayed_signal)

    DELAY.append(delay)
    LOSS.append(traveltime)
    # Update the plot
    line_original.set_ydata(original_signal)
    line_delayed.set_ydata(delayed_signal)
    axes[1].plot(DELAY, LOSS, c='b')
    axes[1].set_xlim(-100, 100)
    axes[1].set_xlabel("Delay")
    axes[1].set_ylabel("Travel Time")
    plt.tight_layout()

    return line_original, line_delayed

# Generate the animation
delay_values = np.arange(-100, 100, 10)
animation = FuncAnimation(fig, update, interval=1000,
                          frames=delay_values, blit=True, repeat=False)

# Save the animation
animation.save(f'{save_path}/traveltime.gif', writer='imagemagick', fps=100)

plt.show()
