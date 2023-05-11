import pickle
import pickle


# Specify the file path where you want to save the list
file_path = 'list_file.pkl'

shots = 111
"""Generate source list"""
sources = []
for shot in range(shots):
    shot_z = 1
    shot_x = 1+shot*5
    sources.append([shot_x, shot_z])


"""Generate receiver list"""
receivers = []
receiver_counts = 561

for shot in range(shots):
    recv_z = [24 for i in range(receiver_counts)]
    recv_x = [i for i in range(receiver_counts)]
    receivers.append([recv_x, recv_z])

def write_pkl(path: str, data: list):
    # Open the file in binary mode and write the list using pickle
    with open(path, 'wb') as f:
        pickle.dump(data, f)

write_pkl("../geometry/marmousi_obn/sources.pkl", sources)
write_pkl("../geometry/marmousi_obn/receivers.pkl", receivers)