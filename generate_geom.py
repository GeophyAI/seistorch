import pickle
import pickle


# Specify the file path where you want to save the list

# shots = 111
# """Generate source list"""
# sources = []
# for shot in range(shots):
#     shot_z = 3
#     shot_x = 1+shot*10
#     sources.append([shot_x, shot_z])


# """Generate receiver list"""
# receivers = []
# receiver_counts = 561*2

# for shot in range(shots):
#     recv_z = [3 for i in range(receiver_counts)]
#     recv_x = [i+1 for i in range(receiver_counts)]
#     receivers.append([recv_x, recv_z])

# def write_pkl(path: str, data: list):
#     # Open the file in binary mode and write the list using pickle
#     with open(path, 'wb') as f:
#         pickle.dump(data, f)

# write_pkl("./geometry/marmousi_surface_10m/sources.pkl", sources)
# write_pkl("./geometry/marmousi_surface_10m/receivers.pkl", receivers)

"""OBC"""

nx, nz = 1122, 346

