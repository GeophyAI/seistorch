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


# d = np.load("/mnt/others/DATA/Inversion/RNN/data/marmousi_acoustic.npy")
# print(d.shape)
# dx = d[:,:,:,0]
# dz = d[:,:,:,0]
# no = 0
# fig,axes = plt.subplots(1,2, figsize=(10,5))
# vmin, vmax = np.percentile(dx[no], [2, 98])
# axes[0].imshow(dx[no].squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.seismic)
# vmin, vmax = np.percentile(dz[no], [2, 98])
# axes[1].imshow(dz[no].squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.seismic)
# plt.show()

# loss = np.load("/public1/home/wangsw/FWI/AFWI/Hessian_free/loss.npy")
# print(loss.shape)
# print(loss[0][0])

F = 4
epoch = 39
PMLN = 50
# epoch = args.epoch
# F=args.frequency_index
#root_path = r"/public1/home/wangsw/FWI/EFWI/Marmousi/marmousi_10m/l1reg"
root_path = r"/mnt/data/wangsw/inversion/marmousi/results/dynamic_codingEFWI_l2"
coding = "."
#grad_vp = np.load("{root_path}/gradvsE00S70.npy")
grad_vp = np.load(f"{root_path}/{coding}/gradvpF{F:02d}E{epoch:02d}.npy")[PMLN:-PMLN,PMLN:-PMLN]
grad_vs = np.load(f"{root_path}/{coding}/gradvsF{F:02d}E{epoch:02d}.npy")[PMLN:-PMLN,PMLN:-PMLN]
print(grad_vs.min(), grad_vs.max(), grad_vp.max(), grad_vp.min())
fig,axes = plt.subplots(1,2, figsize=(10,3))
vmin, vmax = np.percentile(grad_vp, [2, 98])
ax0=axes[0].imshow(grad_vp.squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.jet)
vmin, vmax = np.percentile(grad_vs, [2, 98])
ax1=axes[1].imshow(grad_vs.squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.jet)
plt.colorbar(ax1)
plt.colorbar(ax0)
plt.show()

# Acoustic case
true_vp = np.load("/mnt/data/wangsw/inversion/marmousi/velocity/true_vp.npy")
vp = np.load(f"{root_path}/{coding}/velvpF{F:02d}E{epoch:02d}.npy")[PMLN:-PMLN,PMLN:-PMLN]
vs = np.load(f"{root_path}/{coding}/velvsF{F:02d}E{epoch:02d}.npy")[PMLN:-PMLN,PMLN:-PMLN]

fig,axes = plt.subplots(1,2, figsize=(10,3))
vmin, vmax = (true_vp.min(), true_vp.max())#(1.5, 5.500)
ax0=axes[0].imshow(vp.squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.seismic)
vmin, vmax = (0, true_vp.max()/1.73)#(1.5, 5.500)
ax1=axes[1].imshow(vs.squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.seismic)
plt.colorbar(ax0);plt.colorbar(ax1)
plt.tight_layout()
plt.show()

# true_vp = np.load("/public1/home/wangsw/FWI/EFWI/Marmousi/marmousi_10m/velocity/true_vp.npy")
# true_vs = np.load("/public1/home/wangsw/FWI/EFWI/Marmousi/marmousi_10m/velocity/true_vs.npy")

# vp = np.load(f"{root_path}/{coding}/velvpF{F:02d}E{epoch:02d}.npy")[PMLN:-PMLN,PMLN:-PMLN]
# vs = np.load(f"{root_path}/{coding}/velvsF{F:02d}E{epoch:02d}.npy")[PMLN:-PMLN,PMLN:-PMLN]
# print(vp.min(), vp.max(), vs.max(), vs.min())
# fig,axes = plt.subplots(2,2, figsize=(10,8))
# vmin, vmax = (true_vp.min(), true_vp.max())#(1.5, 5.500)
# ax0=axes[0,0].imshow(true_vp.squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.seismic)
# ax1=axes[1,0].imshow(vp.squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.seismic)
# vmin, vmax = (true_vs.min(), true_vs.max())#(1.500/1.73, 5.500/1.73)
# ax2 = axes[0,1].imshow(true_vs.squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.seismic)
# ax3 = axes[1,1].imshow(vs.squeeze(), vmin=vmin, vmax=vmax, aspect='auto', cmap=plt.cm.seismic)
# plt.colorbar(ax1);plt.colorbar(ax0)
# plt.colorbar(ax2);plt.colorbar(ax3)
# plt.tight_layout()
# plt.show()

def loss_norm(path, freqnorm = False, eps=1e-20):
    loss = np.load(path)
    #loss = np.sum(loss, axis=2)
    for i in range(loss.shape[0]):
        loss[i] = loss[i]/(loss[i].max()+eps)
        if freqnorm and i >0:
            loss[i] = loss[i] * loss[i-1][-1]
    return loss.flatten()

root_path = r"/mnt/data/wangsw/inversion/marmousi/results/dynamic_codingAFWI_l2"
root_path2 = r"/mnt/data/wangsw/inversion/marmousi/results/dynamic_coding_env"
plt.plot(loss_norm(f"{root_path}/loss.npy"))
plt.plot(loss_norm(f"{root_path2}/loss.npy"))
plt.legend(["STD", "Gram-invvp_only"])
plt.show()

