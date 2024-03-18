import os, re
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def read_npy_files(path):

    npy_files = sorted(glob.glob(os.path.join(path, "para*.npy")))
    npy_files.sort(key=lambda x: int(re.search(r'E(\d+)', x).group(1)))
    return npy_files[::10]

def update(frame):
    # ax.cla()
    ax.clear()
    # print(frame)
    # file_path = os.path.join(path, frame)
    pmln=50
    expand=50
    array = np.load(frame)[pmln:-pmln, pmln+expand:-pmln-expand]
    iteration = int(re.search(r'E(\d+)', frame).group(1))
    dh=20
    extent = [0, array.shape[1]*dh, array.shape[0]*dh, 0]
    kwargs=dict(extent=extent, cmap="seismic", aspect="auto", vmin=1500, vmax=5500)
    ax.imshow(array, **kwargs)
    ax.set_title(f"Iteration {iteration}")
    # plt.colorbar(mappable=ax.images[0], ax=ax)
    # plt.colorbar(mappable=ax.images[0], ax=ax)
    # set colorbar
    # plt.tight_layout()
    frame_counter[0] += 1

if __name__ == "__main__":
    path = "siren"
    npy_files = read_npy_files(path)

    # Set up the figure and axis
    fig, ax = plt.subplots(1,1,figsize=(6, 4))

    # Create an animation
    frame_counter = [0]
    animation = FuncAnimation(fig, update, frames=npy_files, repeat=False)

    # Display the animation
    # plt.show()

    # Save animation as an MP4 file
    animation.save(f'{path}.mp4', writer='ffmpeg', fps=24)
