import os, re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

image_folder = 'figures'

image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

image_files.sort()

iteration_numbers = [int(re.search(r'\d+', f).group()) for f in image_files]


output_video_path = 'output_video.mp4'

first_image_path = os.path.join(image_folder, image_files[0])
img = plt.imread(first_image_path)
height, width, _ = img.shape

fig, ax = plt.subplots()
ax.set_axis_off()

def init():
    im = ax.imshow(np.zeros((height, width, 3)), animated=True)
    title = ax.set_title('')
    plt.tight_layout()

    return [im, title]

def update(frame):
    image_path = os.path.join(image_folder, image_files[frame])
    img = plt.imread(image_path)
    im = ax.imshow(img, animated=True)
    title = ax.set_title(f'Iteration: {iteration_numbers[frame]}')
    plt.tight_layout()
    return [im, title]

ani = FuncAnimation(fig, update, frames=len(image_files), init_func=init, blit=True)

ani.save(output_video_path, writer='ffmpeg', fps=5)

print(f'Video saved to: {output_video_path}')
