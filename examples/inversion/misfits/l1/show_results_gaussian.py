import numpy as np
import matplotlib.pyplot as plt
import torch, glob, os

npml = 50
expand = 50
dh = 20
savepath = 'figures/gaussian'
os.makedirs(savepath, exist_ok=True)
true = np.load('../../../models/marmousi_model/true_vp.npy')[:,expand:-expand]
init = np.load('../../../models/marmousi_model/linear_vp.npy')[:,expand:-expand]
l2n = torch.load('./results/l2/model_F04E49.pt')['vp'].cpu().detach()[npml:-npml, npml+expand:-npml-expand]
l1n = torch.load('./results/l1/model_F04E49.pt')['vp'].cpu().detach()[npml:-npml, npml+expand:-npml-expand]
l2 = torch.load('./results/l2_gaussian/model_F04E49.pt')['vp'].cpu().detach()[npml:-npml, npml+expand:-npml-expand]
l1 = torch.load('./results/l1_gaussian/model_F04E49.pt')['vp'].cpu().detach()[npml:-npml, npml+expand:-npml-expand]
fig, axes=plt.subplots(2,2,figsize=(8,6))
vmin,vmax=true.min(),true.max()
extent = [0, true.shape[1]*dh, true.shape[0]*dh, 0]
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax, "extent":extent}
axes[0,0].imshow(true,**kwargs)
axes[0,0].set_title("True")
axes[0,1].imshow(init,**kwargs)
axes[0,1].set_title("Initial")
axes[1,0].imshow(l2,**kwargs)
axes[1,0].set_title("L2")
axes[1,1].imshow(l1,**kwargs)
axes[1,1].set_title("L1")
for ax in axes.ravel():
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
plt.tight_layout()
plt.savefig(f"{savepath}/Inverted_withnoise.png",dpi=300)

fig, axes=plt.subplots(2,2,figsize=(8,6))
vmin,vmax=true.min(),true.max()
extent = [0, true.shape[1]*dh, true.shape[0]*dh, 0]
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax, "extent":extent}
axes[0,0].imshow(true,**kwargs)
axes[0,0].set_title("True")
axes[0,1].imshow(init,**kwargs)
axes[0,1].set_title("Initial")
axes[1,0].imshow(l2n,**kwargs)
axes[1,0].set_title("L2")
axes[1,1].imshow(l1n,**kwargs)
axes[1,1].set_title("L1")
for ax in axes.ravel():
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Depth (m)")
plt.tight_layout()
plt.savefig(f"{savepath}/Inverted_without_noise.png",dpi=300)

# Compare the results (data with gaussian noise)
fig, ax = plt.subplots(1,1,figsize=(5,3))
dh=20
z=np.arange(0,true.shape[0]*dh,dh)
trace_no = 200
ax.plot(z, true[:,trace_no], label="True")
ax.plot(z, init[:,trace_no], label="Initial")
ax.plot(z, l2[:,trace_no], label="L2")
ax.plot(z, l1[:,trace_no], label="L1")
ax.set_xlabel("Depth (m)")
ax.set_ylabel("Velocity (m/s)")
ax.legend()
ax.set_title("Comparison (with gaussian noise)")
plt.tight_layout()
plt.savefig(f"{savepath}/Traces_with_noise.png",dpi=300)
plt.show()

# Compare the results (data with gaussian noise)
fig, ax = plt.subplots(1,1,figsize=(5,3))
dh=20
z=np.arange(0,true.shape[0]*dh,dh)
trace_no = 200
ax.plot(z, true[:,trace_no], label="True")
ax.plot(z, init[:,trace_no], label="Initial")
ax.plot(z, l2n[:,trace_no], label="L2")
ax.plot(z, l1n[:,trace_no], label="L1")
ax.set_xlabel("Depth (m)")
ax.set_ylabel("Velocity (m/s)")
ax.legend()
ax.set_title("Comparison (without gaussian noise)")
plt.tight_layout()
plt.savefig(f"{savepath}/Traces_without_noise.png",dpi=300)
plt.show()

# Compare the results (with/without noise)
fig, ax = plt.subplots(1,1,figsize=(5,3))
dh=20
z=np.arange(0,true.shape[0]*dh,dh)
trace_no = 200
ax.plot(z, true[:,trace_no], label="True")
ax.plot(z, init[:,trace_no], label="Initial")
ax.plot(z, l1[:,trace_no], label="L1 with noise")
ax.plot(z, l1n[:,trace_no], label="L1 without noise")
ax.set_xlabel("Depth (m)")
ax.set_ylabel("Velocity (m/s)")
ax.legend()
ax.set_title("Comparison")
plt.tight_layout()
plt.savefig(f"{savepath}/Traces_with(out)_noise_l1.png",dpi=300)
plt.show()

# Compare the results (with/without noise)
fig, ax = plt.subplots(1,1,figsize=(5,3))
dh=20
z=np.arange(0,true.shape[0]*dh,dh)
trace_no = 200
ax.plot(z, true[:,trace_no], label="True")
ax.plot(z, init[:,trace_no], label="Initial")
ax.plot(z, l2[:,trace_no], label="L2 with noise")
ax.plot(z, l2n[:,trace_no], label="L2 without noise")
ax.set_xlabel("Depth (m)")
ax.set_ylabel("Velocity (m/s)")
ax.legend()
ax.set_title("Comparison")
plt.tight_layout()
plt.savefig(f"{savepath}/Traces_with(out)_noise_l2.png",dpi=300)
plt.show()

def cal_model_error(path, true):
    files = sorted(glob.glob(path+'/model_*.pt'))
    error = []
    for file in files:
        model = torch.load(file)['vp'].cpu().detach().numpy()[npml:-npml, npml+expand:-npml-expand]
        error.append(np.sum((model-true)**2))
    return np.array(error)

l2_with = cal_model_error('./results/l2_gaussian', true)
l2_without = cal_model_error('./results/l2', true)
l1_with = cal_model_error('./results/l1_gaussian', true)
l1_without = cal_model_error('./results/l1', true)

fig, ax=plt.subplots(1,1,figsize=(5,3))
ax.plot(l2_with, label="L2 with noise")
ax.plot(l2_without, label="L2 without noise")
ax.plot(l1_with, label="L1 with noise")
ax.plot(l1_without, label="L1 without noise")
ax.set_xlabel("Epoch")
ax.set_ylabel("Error")
ax.legend()
plt.tight_layout()
plt.savefig(f"{savepath}/ModelError.png",dpi=300, bbox_inches='tight')
