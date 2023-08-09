import pickle
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Specify the file path where you want to save the list

# 692, 2244, 5m
shots = 10
dx = dz = 10 # meter
shot_interval = 128*dx//10 # meter
receiver_depth = dz # meter
shot_depth = dz # meter
receiver_locx = np.linspace(0, 200, 200)
receiver_counts = receiver_locx.size


"""Generate source list"""
sources = []
for s in np.linspace(0, 200, 10):
    sources.append([s, 1])
# for shot in range(shots):
#     shot_z = shot_depth//dz
#     shot_x = 1+shot*shot_interval//dx
#     sources.append([shot_x, shot_z])

"""Generate receiver list"""
receivers = []
recv_x = receiver_locx.tolist()
recv_z = [1 for i in range(len(recv_x))]
receivers = [[recv_x, recv_z]]*len(sources)
# for shot in range(shots):
#     recv_z = [1 for i in range(receiver_counts)]
#     recv_x = [i for i in range(receiver_counts)]
#     recv_x = receiver_locx.tolist()
#     receivers.append([recv_x, recv_z])

def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)

write_pkl("./geometry/layer2d/sources.pkl", sources)
write_pkl("./geometry/layer2d/receivers.pkl", receivers)

"""OBC"""
for src in sources:
    plt.scatter(src[0], src[1])
plt.show()
for rec in receivers:
    plt.scatter(rec[0], rec[1])
plt.show()

# nx, nz = 1122, 346

