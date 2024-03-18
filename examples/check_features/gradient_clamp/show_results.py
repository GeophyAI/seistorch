import numpy as np
import matplotlib.pyplot as plt

npml = 50
expand = 50
F = 2
E = 49
true = np.load('../../models/marmousi_model/true_vp.npy')[:,expand:-expand]
init = np.load('../../models/marmousi_model/linear_vp.npy')[:,expand:-expand]
noclamp = np.load(f'./results_noclamp/paravpF{F:02d}E{E:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
withclamp = np.load(f'./results_withclamp/paravpF{F:02d}E{E:02d}.npy')[npml:-npml, npml+expand:-npml-expand]

fig, axes=plt.subplots(4,1,figsize=(6,10))
vmin,vmax=true.min(),true.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].imshow(true,**kwargs)
axes[0].set_title("True")
axes[1].imshow(init,**kwargs)
axes[1].set_title("Initial")
axes[2].imshow(noclamp,**kwargs)
axes[2].set_title("noclamp")
axes[3].imshow(withclamp,**kwargs)
axes[3].set_title("withclamp")
plt.tight_layout()
plt.savefig("Inverted.png",dpi=300)

# Show the model error
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_model_error_from_tensorboard(path):
    tb = EventAccumulator(path, size_guidance={'scalars': 0})
    tb.Reload()
    tb_steps = tb.Scalars('model_error/vp')
    tb_steps = [i.step for i in tb_steps]
    tb_loss = tb.Scalars('model_error/vp')
    tb_loss = [i.value for i in tb_loss]
    return tb_steps, tb_loss

noclamp_steps, noclamp_loss = load_model_error_from_tensorboard(f'./results_noclamp/logs')
withclamp_steps, withclamp_loss = load_model_error_from_tensorboard(f'./results_withclamp/logs')

fig, ax=plt.subplots(1,1,figsize=(6,3))
ax.plot(noclamp_steps, noclamp_loss, label="noclamp")
ax.plot(withclamp_steps, withclamp_loss, label="withclamp")
ax.legend()
plt.tight_layout()
plt.savefig("Model_error.png",dpi=300)
plt.show()

# Show the gradient at at first iteration
npml = 50
expand = 50
F = 0
E = 0
noclamp = np.load(f'./results_noclamp/gradvpF{F:02d}E{E:02d}.npy')[npml:-npml, npml+expand:-npml-expand]
vmin, vmax=np.percentile(noclamp, [2,98])
clamped_grad = np.clip(noclamp, vmin, vmax)

fig, axes=plt.subplots(3,1,figsize=(6,9))
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].set_title("no clamp")
axes[1].set_title("after clamp")
axes[2].set_title("error")
vmin,vmax=np.percentile(noclamp, [0,100])
plt.colorbar(axes[0].imshow(noclamp,**kwargs),ax=axes[0])
plt.colorbar(axes[1].imshow(clamped_grad,**kwargs),ax=axes[1])
plt.colorbar(axes[2].imshow(clamped_grad-noclamp,**kwargs),ax=axes[2])
plt.tight_layout()
plt.savefig("Gradientat0epoch.png",dpi=300)
plt.show()
