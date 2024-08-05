import numpy as np
import matplotlib.pyplot as plt
import torch

bwidth = 50
expand = 50
epoch = 99
dh = 20
grad_vp = torch.load(f'./results/grad_vp_F00E00.pt').cpu().detach().numpy()[bwidth:-bwidth, bwidth+expand:-bwidth-expand]
grad_rx = torch.load(f'./results/grad_rx_F00E00.pt').cpu().detach().numpy()[bwidth:-bwidth, bwidth+expand:-bwidth-expand]
grad_rz = torch.load(f'./results/grad_rz_F00E00.pt').cpu().detach().numpy()[bwidth:-bwidth, bwidth+expand:-bwidth-expand]
extent = [0, grad_vp.shape[1]*dh, grad_vp.shape[0]*dh, 0]
fig, axes=plt.subplots(3,1,figsize=(8,10))
titles = ["Gradient of vp", "Gradient of rx", "Gradient of rz"]
for ax, grad, title in zip(axes, [grad_vp, grad_rx, grad_rz], titles):
    vmin,vmax=np.percentile(grad,[2, 98])
    kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax,"extent":extent}
    ax.imshow(grad,**kwargs)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
plt.tight_layout()
plt.savefig("figures/Gradient_at_first_epoch.png",dpi=300)

plt.show()
model = torch.load(f'./results/model_F00E{epoch:02d}.pt')
vp = model['vp'].cpu().detach().numpy()[bwidth:-bwidth, bwidth+expand:-bwidth-expand]
rx = model['rx'].cpu().detach().numpy()[bwidth:-bwidth, bwidth+expand:-bwidth-expand]
rz = model['rz'].cpu().detach().numpy()[bwidth:-bwidth, bwidth+expand:-bwidth-expand]
fig, axes=plt.subplots(3,1,figsize=(6,8))
titles = ["vp", "rx", "rz"]
for ax, model, title in zip(axes, [vp, rx, rz], titles):
    kwargs={"cmap":"seismic","aspect":"auto","extent":extent}
    if title == "vp":
        vmin,vmax=1500, 5500
        kwargs["vmin"]=vmin
        kwargs["vmax"]=vmax
    else:
        vmin, vmax=np.percentile(model,[2, 98])
        kwargs["vmin"]=vmin
        kwargs["vmax"]=vmax
        kwargs["cmap"]='gray'
    ax.imshow(model,**kwargs)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title(title)
plt.tight_layout()
plt.savefig("figures/Inverted_Profile.png",dpi=300, bbox_inches='tight')
plt.show()

true_vp = np.load("models/true_vp.npy")[:,expand:-expand]
init_vp = np.load("models/smooth_vp.npy")[:,expand:-expand]
invt_vp = vp
true_rx = np.load("models/rx.npy")[:,expand:-expand]
init_rx = np.load("models/zero_m.npy")[:,expand:-expand]
invt_rx = rx
true_rz = np.load("models/rz.npy")[:,expand:-expand]
init_rz = np.load("models/zero_m.npy")[:,expand:-expand]
invt_rz = rz

fig, axes=plt.subplots(3,1,figsize=(6,8))
titles = ["vp", "rx", "rz"]
trace_no = 300
zz = np.arange(true_vp.shape[0])*dh
axes[0].plot(zz, true_vp[:,trace_no], label="True")
axes[0].plot(zz, init_vp[:,trace_no], label="Initial")
axes[0].plot(zz, invt_vp[:,trace_no], label="Inverted")
axes[0].set_title("vp")
axes[0].set_xlabel("z (m)")
axes[0].legend()

xx = np.arange(true_vp.shape[1])*dh
axes[1].plot(xx, true_rx[50], label="True")
axes[1].plot(xx, init_rx[50], label="Initial")
axes[1].plot(xx, invt_rx[50], label="Inverted")
axes[1].set_title("rx")
axes[1].set_xlabel("x (m)")
axes[1].legend()

axes[2].plot(zz, true_rz[:, trace_no], label="True")
axes[2].plot(zz, init_rz[:, trace_no], label="Initial")
axes[2].plot(zz, invt_rz[:, trace_no], label="Inverted")
axes[2].set_title("rz")
axes[2].set_xlabel("z (m)")
axes[2].legend()

plt.tight_layout()
plt.savefig("figures/Comparison.png",dpi=300)
plt.show()