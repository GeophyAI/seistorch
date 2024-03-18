import numpy as np
import glob
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


cg_invs = glob.glob("./results_cg*/")
sd_invs = glob.glob("./results_sd*/")

fig, ax=plt.subplots(1,1,figsize=(6,3))
for cg_inv in cg_invs:
    loss = np.load(f"{cg_inv}/loss.npy")
    loss /= np.max(loss, axis=1, keepdims=True)
    ax.plot(loss.flatten(), label=cg_inv)

for sd_inv in sd_invs:
    loss = np.load(f"{sd_inv}/loss.npy")
    loss /= np.max(loss, axis=1, keepdims=True)
    ax.plot(loss.flatten(), label=sd_inv)

plt.legend(ncol=3, loc=3, bbox_to_anchor=(0, 1))
plt.tight_layout()
fig.savefig("Loss.png",dpi=300)
plt.show()

# # Compare the results
npml = 50
expand = 50
F=2
E=38

true = np.load('../../../models/marmousi_model/true_vp.npy')[:,expand:-expand]
init = np.load('../../../models/marmousi_model/linear_vp.npy')[:,expand:-expand]

trace = 200
fig, ax = plt.subplots(1,1,figsize=(6,3))
ax.plot(true[:,trace],label="True")
ax.plot(init[:,trace],label="Initial")
for cg_inv, sd_inv in zip(cg_invs, sd_invs):
    cg_inv_result = np.load(f'{cg_inv}/paravpF{F:02d}E{E:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
    sd_inv_result = np.load(f'{sd_inv}/paravpF{F:02d}E{E:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
    ax.plot(cg_inv_result[:,trace],label=cg_inv)
    ax.plot(sd_inv_result[:,trace],label=sd_inv)
    ax.legend()
plt.tight_layout()
fig.savefig("Inverted_line.png",dpi=300)

def load_model_error_from_tensorboard(path):
    tb = EventAccumulator(path, size_guidance={'scalars': 0})
    tb.Reload()
    tb_steps = tb.Scalars('model_error/vp')
    tb_steps = [i.step for i in tb_steps]
    tb_loss = tb.Scalars('model_error/vp')
    tb_loss = [i.value for i in tb_loss]
    return tb_steps, tb_loss

fig, ax=plt.subplots(1,1,figsize=(6,3))
for cg_inv, sd_inv in zip(cg_invs, sd_invs):
    cg_steps, cg_loss = load_model_error_from_tensorboard(f'{cg_inv}/logs')
    sd_steps, sd_loss = load_model_error_from_tensorboard(f'{sd_inv}/logs')
    ax.plot(cg_steps, cg_loss, label=cg_inv)
    ax.plot(sd_steps, sd_loss, label=sd_inv)
ax.legend()
plt.tight_layout()
fig.savefig("Model_error.png",dpi=300)
plt.show()

inverted_cg = np.load(f'./results_cg/paravpF{F:02d}E{E:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
inverted_sd = np.load(f'./results_sd/paravpF{F:02d}E{E:02d}.npy')[npml:-npml, npml+expand:-npml-expand]

def show_and_save(data, title=""):
    fig, ax=plt.subplots(1,1,figsize=(4,3))
    vmin,vmax=1500,5500
    kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
    ax.imshow(data,**kwargs)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(f"{title}.png",dpi=300)

for cg_inv, sd_inv in zip(cg_invs, sd_invs):
    cg = np.load(f'{cg_inv}/paravpF{F:02d}E{E:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
    sd = np.load(f'{sd_inv}/paravpF{F:02d}E{E:02d}.npy')[npml:-npml, npml+expand:-npml-expand]

    show_and_save(cg, title=cg_inv)
    show_and_save(sd, title=sd_inv)