import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

import sys
sys.path.append("../../../..")
from seistorch.show import SeisShow

show = SeisShow()

def read_pkl(path):
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def getmap(recs_x):
    """This function will return a map from loc index to all valid shot index.

    Args:
        recs_x (np.ndarray): 2D array of receiver x coordinates in grid.

    Returns:
        dict: map from loc index to all valid shot index.
        Example: {0: [0, 1, 2, 3], 
                  1: [1, 2, 3, 4], 
                  ...}
    """
    nshots, nrecs = recs_x.shape
    recmap = dict()
    for i in range(nshots):
        for j in range(nrecs):
            rec_x = recs_x[i][j]
            if rec_x in recmap:
                recmap[rec_x].append(i)
            else:
                recmap[rec_x] = [i]
    return recmap

def find_intersection(*lists):
    if not lists:
        return []

    # Convert the first list to a set
    result_set = set(lists[0])

    # Find the intersection with each subsequent list
    for lst in lists[1:]:
        result_set.intersection_update(lst)

    # Convert the final set result back to a list
    intersection_result = list(result_set)
    
    return intersection_result

# Input the receiver x and output the valid shot index
def getvalidrec(rec_x):

    if isinstance(rec_x, int):
        rec_x = [rec_x]

    valid_shots = []

    for x in rec_x:
        if x not in valid_rec_x:
            raise ValueError(f"Invalid receiver x: {x}")
        else:
            valid_shots.append(recmap[x])

    return find_intersection(*valid_shots)

recs = read_pkl("./geometry/receivers.pkl")
srcs = read_pkl("./geometry/sources.pkl")
vel = np.ones((10, 561))

assert len(recs)==len(srcs), "Number of shots and receivers should be the same."

recs = np.array(recs)
srcs = np.array(srcs)

# Show the towed geoetry
show.geometry(vel, srcs, recs, savepath="geometry_towed.gif", dh=20, interval=1)

for i in range(3):

    fig,ax=plt.subplots(figsize=(10,3))
    for j in range(i+1):
        ax.plot(recs[j][0][::2], recs[j][1][::2]+j*0.8, 'v', label=f'recs of shot {j}')
        ax.plot(srcs[j][0], srcs[j][1]+j*0.8, 'r*', label=f'src of shot {j}')

    ax.set_xlabel('x in grid')
    ax.set_yticks([])
    ax.legend()
    ax.set_xlim(0, 201)
    ax.set_ylim(0, 3.)
    fig.savefig(f"geometry_towed_{i:02d}_shot.png", dpi=300, bbox_inches='tight')
    plt.show()

# BIG interval
step=6
for i in range(0,18,step):

    fig,ax=plt.subplots(figsize=(10,3))
    for j in range(i+1):
        ax.plot(recs[j][0][::2], recs[j][1][::2]+j*0.8, 'v')
        ax.plot(srcs[j][0], srcs[j][1]+j*0.8, 'r*')

    ax.vlines(max(recs[0][0]), 0, 15, color='k', linestyle='--')
    ax.vlines(min(recs[i][0]), 0, 15, color='k', linestyle='--')

    ax.set_xlabel('x in grid')
    ax.set_yticks([])
    ax.legend()
    ax.set_xlim(0, 201)
    ax.set_ylim(0, 12.)
    fig.savefig(f"Big_interval-_towed_{i:02d}_shot.png", dpi=300, bbox_inches='tight')
    plt.show()

# show move
step=1
for i in range(0,len(srcs),step):

    fig,ax=plt.subplots(figsize=(10,3))
    for j in range(i+1):
        ax.plot(recs[j][0][::2], recs[j][1][::2]+j*0.8, 'v')
        ax.plot(srcs[j][0], srcs[j][1]+j*0.8, 'r*')

    # if i in [0,6]: vmin=0.5; vmax=6; maxv = max(recs[i-step][0])
    # if i == 12: vmin=6.5; vmax=11; maxv = max(recs[i-step+1][0])

    # ax.vlines(maxv, vmin, vmax, color='k', linestyle='--')
    # ax.vlines(min(recs[i][0]), vmin, vmax, color='k', linestyle='--')

    ax.set_xlabel('x in grid')
    ax.set_yticks([])
    ax.legend()
    ax.set_xlim(0, 201)
    # ylim_min = min(recs[i][1][::2])
    # ylim_max = max(recs[i][1][::2])
    ax.set_ylim(0, 12.)
    # fig.savefig(f"Move_Big_interval-_towed_{i:02d}_shot.png", dpi=300, bbox_inches='tight')
    plt.show()


# X location of receivers
recs_x = recs[:,0,:]

valid_rec_x, counts = np.unique(recs_x, return_counts=True)

recmap = getmap(recs_x)

# plt.plot(valid_rec_x, counts, 'o')
# plt.show()

# REC = valid_rec_x[60:180].tolist()
# REC = valid_rec_x.tolist()#sorted(np.random.choice(valid_rec_x[10:-10], 30).tolist())#[20:30]
# plt.plot(REC, np.zeros_like(REC), 'o')
# plt.show()

# batch_size = 10
# nshots = len(REC)//batch_size
# for batch in range(nshots):
#     _rec = REC[batch*batch_size:(batch+1)*batch_size]
#     valid_shots = getvalidrec(_rec)
#     print("Valid shots:", valid_shots)

# print("REC:", REC)
# valid_shots = getvalidrec(REC)
# print("Valid shots:", valid_shots)
# print(len(valid_shots))

