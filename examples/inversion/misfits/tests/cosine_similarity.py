import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics.pairwise import cosine_similarity
import time
# Func for Ricker wavelet
def ricker_wavelet(t, f):
    return (1 - 2 * (np.pi * f * t) ** 2) * np.exp(-(np.pi * f * t) ** 2)

# Calculate the cosine similarity between two signals
def calculate_cosine_similarity(signal1, signal2):
    return cosine_similarity(signal1.reshape(1, -1), signal2.reshape(1, -1))[0, 0]

# Generate a Ricker
t = np.linspace(-1, 1, 1000, endpoint=False)
frequency = 10  # 频率
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
    delayed_signal = -np.roll(original_signal, delay)*np.random.random()

    # Calculate the cosine similarity between the original and delayed signals
    similarity = calculate_cosine_similarity(original_signal, delayed_signal)

    DELAY.append(delay)
    LOSS.append(similarity)
    # Update the plot
    line_original.set_ydata(original_signal)
    line_delayed.set_ydata(delayed_signal)
    axes[1].plot(DELAY, LOSS, c='b')
    axes[1].set_xlim(-100, 100)
    axes[1].set_xlabel("Delay")
    axes[1].set_ylabel("Cosine Similarity")
    plt.tight_layout()

    return line_original, line_delayed

# Generate the animation
delay_values = np.arange(-100, 100, 4)
animation = FuncAnimation(fig, update, interval=500,
                          frames=delay_values, blit=True, repeat=False)

# Save the animation
animation.save('signal_comparison_animation_rand.gif', writer='imagemagick', fps=10)

plt.show()
