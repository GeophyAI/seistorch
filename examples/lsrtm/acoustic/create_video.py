import os, re
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation

def read_npy_files(path):

    npy_files = sorted(glob.glob(os.path.join(path, "model*.pt")))
    npy_files.sort(key=lambda x: int(re.search(r'E(\d+)', x).group(1)))
    return npy_files

def update(frame):
    plt.clf()
    # print(frame)
    # file_path = os.path.join(path, frame)
    pmln=50
    expand=50
    array = torch.load(frame, 'cpu')['vp'][pmln:-pmln, pmln+expand:-pmln-expand]
    dh=20
    extent = [0, array.shape[1]*dh, array.shape[0]*dh, 0]
    kwargs=dict(extent=extent, cmap="seismic", aspect="auto", vmin=1500, vmax=5500)
    plt.imshow(array, **kwargs)
    plt.title("Iteration {}".format(frame_counter[0]))
    plt.colorbar()
    # set colorbar
    # plt.tight_layout()
    frame_counter[0] += 1

if __name__ == "__main__":
    path = "/home/shaowinw/synthetic/marmousi2d/multiples_fixed/habc/with_multiples/results"
    npy_files = read_npy_files(path)

    # Set up the figure and axis
    fig, ax = plt.subplots(1,1,figsize=(6, 4))

    # Create an animation
    frame_counter = [0]
    animation = FuncAnimation(fig, update, frames=npy_files, repeat=False)

    # Display the animation
    # plt.show()

    # Save animation as an MP4 file
    animation.save('output.mp4', writer='ffmpeg', fps=24)

