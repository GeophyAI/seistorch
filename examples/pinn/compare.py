import torch, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch.nn as nn

# Define the PINN
class PINN(torch.nn.Module):
    def __init__(self, layers):
        super().__init__() 
        self.act = nn.Tanh()
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]) 
        'Xavier Normal Initialization'
        for i in range(len(layers)-1):
            nn.init.xavier_normal_(self.layers[i].weight.data, gain=1.0)
            # set biases to zero
            nn.init.zeros_(self.layers[i].bias.data)   

    def forward(self, x, z, t):
        a = torch.concatenate((x, z, t), 1)
        for i in range(len(self.layers)-2):  
            z = self.layers[i](a)              
            a = self.act(z)    
        a = self.layers[-1](a)
        return a
    
pinn = torch.load('pretrained')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_tensor(d):
    return torch.from_numpy(d).float().to(device, dtype=torch.float32)


def load_wf(path, pmln, down_scale=4):
    d = np.load(path)[0,pmln:-pmln,pmln:-pmln]
    d = d[::down_scale, ::down_scale]
    return d

# load the config from file
import yaml
with open('forward.yml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
end_time = config['geom']['nt']*config['geom']['dt']
vel=np.load(config['geom']['truePath']['vp'])
gridnz, gridnx=vel.shape
dh = config['geom']['h']/1000.
(nz, nx)=(gridnz*dh, gridnx*dh)
print(f"The size of the model is {nz} km x {nx} km")
print(f"The modeling time is {end_time} s")

wf_files = sorted(glob.glob('wavefield/*.npy'))

dh = config['geom']['h']#/1000.
dt = config['geom']['dt']
nt = config['geom']['nt']
pmln = config['geom']['pml']['N']

rec_pred = []
grid1m_x = int(nx/(dh/1000))
grid1m_z = int(nz/(dh/1000))
xx, zz = np.meshgrid(np.linspace(0,nx,grid1m_x),np.linspace(0,nz,grid1m_z))
xx0 = to_tensor(xx.reshape((-1,1)))
zz0 = to_tensor(zz.reshape((-1,1)))
shape = xx.shape
extent = [0, nx/1000, nz/1000, 0]

# Create a figure and axes for the animation
fig, axes = plt.subplots(1, 3, figsize=(8, 3))
rec_pred = []
rec_true = []
def update(t):
    if t % 10 == 0:
        tt = t * dt * np.ones(shape)
        tt = to_tensor(tt.reshape((-1, 1)))
        with torch.no_grad():
            u_pred = pinn(xx0, zz0, tt).detach().cpu().numpy().reshape(shape)
        u_true = load_wf(wf_files[t + 400], pmln=pmln, down_scale=1)
        vmin, vmax = np.min(u_true), np.max(u_true)
        kwargs = {'vmin': vmin, 'vmax': vmax, 'extent': extent, 'cmap': 'seismic', 'aspect': 'auto'}

        # Clear previous frames
        for ax in axes:
            ax.clear()

        # Plot new frames
        axes[0].imshow(u_pred, **kwargs)
        axes[1].imshow(u_true, **kwargs)
        axes[2].imshow(u_pred - u_true, **kwargs)

        for ax, title in zip(axes.ravel(), ['PINN', 'FD', 'Difference']):
            ax.set_title(f"{title} at {t * dt:.2f} s")
            ax.set_xlabel('x (km)')
            ax.set_ylabel('z (km)')
            ax.axis('off')
        fig.tight_layout()
    

# Create an animation
ani = animation.FuncAnimation(fig, update, frames=nt-400, interval=10)
# Save the animation as an MP4 file
# ani.save('PINN.mp4', writer='ffmpeg', fps=200, dpi=300)

# Show the record
# rec_pred = np.array(rec_pred)
# rec_true = np.array(rec_true)
# fig, ax = plt.subplots(1,2,figsize=(6,4))
# ax[0].imshow(rec_pred, cmap="seismic", aspect="auto")
# ax[0].set_title("PINN")
# ax[1].imshow(rec_true, cmap="seismic", aspect="auto")
# ax[1].set_title("FD")
# plt.tight_layout()
# plt.show()

