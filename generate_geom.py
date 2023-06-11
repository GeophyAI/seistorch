import pickle
import pickle


# Specify the file path where you want to save the list

# 692, 2244, 5m
shots = 111
dx = dz = 5 # meter
shot_interval = 100 # meter
receiver_depth = 480
receiver_counts = 2244


"""Generate source list"""
sources = []
for shot in range(shots):
    shot_z = 0
    shot_x = 1+shot*shot_interval//dx
    sources.append([shot_x, shot_z])

"""Generate receiver list"""
receivers = []

for shot in range(shots):
    recv_z = [receiver_depth//dz for i in range(receiver_counts)]
    recv_x = [i for i in range(receiver_counts)]
    receivers.append([recv_x, recv_z])

def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)

write_pkl("./geometry/marmousi_obn_5m/sources.pkl", sources)
write_pkl("./geometry/marmousi_obn_5m/receivers.pkl", receivers)

"""OBC"""

# nx, nz = 1122, 346

