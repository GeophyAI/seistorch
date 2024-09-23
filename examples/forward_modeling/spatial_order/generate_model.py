import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
nz, nx = 256, 256

# Create a simple velocity model
vp = np.ones((nz, nx), dtype=np.float32)*1500

srcs = [[128, 128]]
recs = [[[128], [128]]]

os.makedirs('velocity', exist_ok=True)
os.makedirs('geometry', exist_ok=True)

np.save('velocity/vp.npy', vp)
with open('geometry/srcs.pkl', 'wb') as f:
    pickle.dump(srcs, f)
with open('geometry/recs.pkl', 'wb') as f:
    pickle.dump(recs, f)
