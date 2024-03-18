import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_model_error_from_tensorboard(path):
    tb = EventAccumulator(path, size_guidance={'scalars': 0})
    tb.Reload()
    tb_steps = tb.Scalars('model_error/vp')
    tb_steps = [i.step for i in tb_steps]
    tb_loss = tb.Scalars('model_error/vp')
    tb_loss = [i.value for i in tb_loss]
    return tb_steps, tb_loss

type1 = 'siren'
type2 = 'siren_scale'
pmln = 50
expand=50
true = np.load('../../models/marmousi_model_half/true_vp.npy')[:,expand:-expand]
epoch = 100
trace = 100
fig, ax=plt.subplots(1,1,figsize=(6,4))
ax.plot(true[:,trace],label="True")
for _type in [type1, type2]:
    inverted = np.load(f'./{_type}/paravpF00E{epoch:02d}.npy')[pmln:-pmln, pmln+expand:-pmln-expand]
    assert inverted.shape == true.shape
    ax.plot(inverted[:,trace],label=_type)
ax.legend()
# fig.savefig("Inverted_profile.png",dpi=300)
plt.show()

fig, axes=plt.subplots(3,1,figsize=(8,10))
vmin,vmax=true.min(),true.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
axes[0].imshow(true,**kwargs)
axes[0].set_title("True")
for i, _type in enumerate([type1, type2]):
    inverted = np.load(f'./{_type}/paravpF00E{epoch:02d}.npy')[pmln:-pmln, pmln+expand:-pmln-expand]
    axes[i+1].imshow(inverted,**kwargs)
    axes[i+1].set_title(_type)
plt.tight_layout()
# plt.savefig("Inverted.png",dpi=300)
plt.show()

# load model error
fig, ax=plt.subplots(1,1,figsize=(6,4))
for _type in [type1, type2]:
    tb_steps, tb_loss = load_model_error_from_tensorboard(f'./{_type}/logs')
    ax.plot(tb_steps, tb_loss, label=_type)
ax.legend()
plt.show()
    