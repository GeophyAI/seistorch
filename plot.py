import matplotlib, glob, os
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
import argparse
# import torch, segyio
# parser = argparse.ArgumentParser()
# parser.add_argument('frequency_index', type=int, 
#                     help='index of multi-scale inversion')
# parser.add_argument('epoch', type=int, 
#                     help='iteration of current frequency band')

# args = parser.parse_args() 

# d = np.load("/mnt/data/wangsw/inversion/bp/data/acoustic_bp_part.npy")

# print(d.shape, d.max(), d.min())
# dx = d[...,0]
# dz = d[...,0]

# no = 55
# fig,axes = plt.subplots(1,2, figsize=(8,5))
# vmin, vmax = np.percentile(dx[no], [2, 98])
# axes[0].imshow(dx[no].squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.seismic)
# vmin, vmax = np.percentile(dz[no], [2, 98])
# axes[1].imshow(dz[no].squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.seismic)
# plt.show()
# # # plt.savefig("test_15hz.png")
# exit()

F = 0
epoch = 499
PMLN = 50
EXPAND = 0
# # epoch = args.epoch

# # F=args.frequency_index
# #root_path = r"/public1/home/wangsw/FWI/EFWI/Marmousi/marmousi_10m/ss"
root_path = r"/mnt/data/wangsw/inversion/bp/compare_loss/envelope"
#r"/mnt/data/wangsw/inversion/marmousi/elastic/oldcodes"
loss = root_path.split("/")[-1]
coding = "."
grad_vp = np.load(f"{root_path}/{coding}/gradvpF{F:02d}E{epoch:02d}.npy")[PMLN:-PMLN,PMLN:-PMLN]
grad_vs = np.load(f"{root_path}/{coding}/gradvpF{F:02d}E{epoch:02d}.npy")[PMLN:-PMLN,PMLN:-PMLN]
# print(grad_vs.min(), grad_vs.max(), grad_vp.max(), grad_vp.min())

fig,axes = plt.subplots(1,2, figsize=(10,3))
vmin, vmax = np.percentile(grad_vp, [2, 98])
ax0=axes[0].imshow(grad_vp.squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.seismic)
vmin, vmax = np.percentile(grad_vs, [2, 98])
ax1=axes[1].imshow(grad_vs.squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.seismic)
axes[0].set_title(root_path.split("/")[-1])
axes[1].set_title(root_path.split("/")[-1])
plt.colorbar(ax1)
plt.colorbar(ax0)
plt.show()


# Acoustic case
# true_vp = np.load("/mnt/data/wangsw/inversion/overthrust/velocity/true_vp.npy")#[:,expand:-expand]
# true_vp = np.load("/mnt/data/wangsw/inversion/marmousi_20m/velocity/true_vp.npy")
true_vp = np.load("/mnt/data/wangsw/inversion/bp/velocity_part/true_vel_z50m_x50m.npy")
init_vp = np.load("/mnt/data/wangsw/inversion/bp/velocity_part/init_vel_z50m_x50m.npy")
vp = np.load(f"{root_path}/{coding}/paravpF{F:02d}E{epoch:02d}.npy")[PMLN:-PMLN,PMLN+EXPAND:-PMLN-EXPAND]
vs = np.load(f"{root_path}/{coding}/paravpF{F:02d}E{epoch:02d}.npy")[PMLN:-PMLN,PMLN+EXPAND:-PMLN-EXPAND]
print(vp.shape, vp.max(), vp.min())

plt.plot(true_vp[:,200])
plt.plot(vp[:,200])
plt.plot(init_vp[:,200])
plt.show()
 
# vp[0:48] = 1500.
fig,axes = plt.subplots(1,2, figsize=(10,3))
vmin, vmax = (true_vp.min(), true_vp.max())#(1.5, 5.500)
# vmin, vmax=vp.max(), vp.min()
ax0=axes[0].imshow(vp.squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.jet)
# vmin, vmax = (0, 5500/1.73)#(1.5, 5.500)
ax1=axes[1].imshow(true_vp.squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.jet)
axes[0].set_title(loss+"_vp")
axes[1].set_title(loss+"_vs")
plt.colorbar(ax0);plt.colorbar(ax1)
plt.tight_layout()
plt.show()

loss = np.load(f"{root_path}/loss.npy")
for i in range(loss.shape[0]):
    loss[i]/=loss[i].max()
plt.plot(loss.flatten())
plt.show()
# fig.savefig(f"./figures/{loss}.png")

