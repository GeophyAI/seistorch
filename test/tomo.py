import torch
import numpy as np
from skimage.graph import route_through_array
import heapq
import matplotlib.pyplot as plt

def eikonal_fmm(velocities, dx, dt, sources, device='cuda'):
    ntime_samples, n_traces = velocities.shape

    # Convert velocities to slownesses
    slownesses = 1 / velocities

    # Compute maximum possible traveltime
    max_traveltime = ntime_samples * dt

    # Initialize traveltimes tensor with maximum possible traveltime
    traveltimes = torch.full((ntime_samples, n_traces), max_traveltime, device=device)

    # Initialize the FMM queue
    queue = []

    # Add the sources to the FMM queue with zero traveltime
    for x, t in sources:
        traveltimes[x, t] = 0
        heapq.heappush(queue, (0, (x, t)))

    # FMM algorithm
    while queue:
        traveltime, (x, t) = heapq.heappop(queue)

        for dx_step, dt_step in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            x_neighbor, t_neighbor = x + dx_step, t + dt_step

            if 0 <= x_neighbor < ntime_samples and 0 <= t_neighbor < n_traces:
                slowness = slownesses[x_neighbor, t_neighbor]
                d_traveltime = dx * slowness if dx_step != 0 else dt * slowness

                neighbor_traveltime = traveltime + d_traveltime

                if neighbor_traveltime < traveltimes[x_neighbor, t_neighbor]:
                    traveltimes[x_neighbor, t_neighbor] = neighbor_traveltime
                    heapq.heappush(queue, (neighbor_traveltime, (x_neighbor, t_neighbor)))

    return traveltimes

def traveltime_inversion(traveltimes, dx, dt, nsources):
    ntime_samples, n_traces = traveltimes.shape
    initial_model = torch.zeros((ntime_samples, n_traces), dtype=torch.float32)

    # Invert traveltimes to obtain the initial model for FWI
    for x in range(ntime_samples):
        for t in range(n_traces):
            initial_model[x, t] = torch.mean(traveltimes[:, x, t])

    # Convert traveltimes to velocities
    initial_model = (dx + dt) / initial_model

    return initial_model

seismic_record = np.load("/mnt/data/wangsw/inversion/marmousi_20m/data/marmousi_acoustic.npy").squeeze()

seismic_record = torch.from_numpy(seismic_record)

# Input parameters
nsources, ntime_samples, n_traces = seismic_record.shape

# Create a velocity model as a torch.Tensor with shape (ntime_samples, n_traces)
velocities = torch.ones((ntime_samples, n_traces)) * 2000  # 2000 m/s homogeneous model

# Define spatial and temporal steps
dx = 20
dt = 0.001

# Create the source coordinates tensor with shape (nsources, 2)
sources = np.array([[1, 1], [5, 1], [10, 1], [15, 1]])

# Compute the traveltimes for each source using the Fast Marching Method
traveltimes_list = []
for source in sources:
    traveltimes = eikonal_fmm(velocities, dx, dt, [source])
    traveltimes_list.append(traveltimes)

traveltimes = torch.stack(traveltimes_list)
print(traveltimes)
initial_model = traveltime_inversion(traveltimes, dx, dt, nsources)
print(initial_model.shape)
plt.imshow(initial_model)
plt.show()