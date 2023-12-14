# This code implements a physics-informed neural network (PINN) for the
# second order acoustic wave equation.

#%% Import packages
import torch, tqdm, glob, yaml, pickle
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from pyDOE import lhs

# %% Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)

# %% Define necessary functions

def ricker_wave(fm, dt, nt, delay = 80, dtype='tensor', inverse=False):
    """
        Ricker-like wave.
    """
    print(f"Wavelet inverse:{inverse}")
    ricker = []
    delay = delay * dt 
    t = np.arange(0, nt*dt, dt)

    c = np.pi * fm * (t - delay) #  delay
    p = -1 if inverse else 1
    ricker = p*(1-2*np.power(c, 2)) * np.exp(-np.power(c, 2))

    if dtype == 'numpy':
        return np.array(ricker).astype(np.float32)
    else:
        return torch.from_numpy(np.array(ricker).astype(np.float32))

def to_tensor(d):
    return torch.from_numpy(d).float().to(device, dtype=torch.float32)
# load the configure file
def load_config(path):
    with open('forward.yml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config
# load the pkl file
def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_wf(path, pmln, down_scale=4):
    d = np.load(path)[0,pmln:-pmln,pmln:-pmln]
    d = d[::down_scale, ::down_scale]
    return d

def load_rec(path):
    return np.load(path, allow_pickle=True)

def pde(x, z, t, vel=1.5):
    u = pinn(x, z, t)
    kwargs = {'create_graph': True, 'retain_graph': True}
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), **kwargs)[0]
    u_tt = torch.autograd.grad(u_t, t, grad_outputs=torch.ones_like(u_t), **kwargs)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), **kwargs)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), **kwargs)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u), **kwargs)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=torch.ones_like(u_z), **kwargs)[0]

    # 2D acoustic wave equation u_tt=c**2*(u_xx+u_zz)
    u_pde = vel**2*(u_xx+u_zz) - u_tt
    return u_pde

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

# %% Define the path of the wavefields and record
cfg = load_config('forward.yml')
record = np.load(cfg['geom']['obsPath'], allow_pickle=True)
wf_files = sorted(glob.glob('wavefield/*.npy'))

#%% Configures for the PINN
# Since the wavefield u in 2D is a function of (x, z, t), 
# so the input dimension is 3, and the output dimension is 1.
num_hidden_layers = 5
neurons_per_layer = 20
layers = [3] + [neurons_per_layer]*num_hidden_layers + [1]
epochs = 200000
lr=1e-4

wf_scale = 4 # The downsample rate of the wavefields
rec_scale = 10 # The downsample rate of the record

# Data usage
use_u0 = False
use_u1 = False
use_rec = True
use_pde = True
use_wavelet = False
use_vel = False

# epoch = 50000 
# Parameters in the forward modeling
dt = cfg['geom']['dt'] # unit: sec
dh = cfg['geom']['h']/1000 # unit: km
nt = cfg['geom']['nt'] # number of time steps in recording
nt_pinn = int(nt//rec_scale) # number of time steps in PINN
pmln = cfg['geom']['pml']['N'] # number of PML layers
endt = dt*nt # unit: sec
vel = np.load(cfg['geom']['truePath']['vp']) # The true velocity
grid_nz, grid_nx = vel.shape # The grid size of the velocity model
nz, nx = grid_nz*dh, grid_nx*dh # The physical size of the velocity model
v = torch.rand(1).to(device) # The velocity model
v.requires_grad = True
# Parameters for training
pde_samples = int(2e4)
t0 = 0.4 # unit: sec, initial condition 1 at t=0.4s
t1 = 0.5 # unit: sec, initial condition 2 at t=0.5s
t_test = 0.8 # unit: sec, test data at t=0.8s
t_offset = t0 # The start time of PINN is 0. You can also set the start time to be t0.
end_t_in_physical = endt # The end time of the physical world that you want to model
end_t_in_pinn = end_t_in_physical - t_offset # The end time of the PINN world that you want to model

print(f"The size of the model is {grid_nz} km x {grid_nx} km")
print(f"The modeling time is {endt} s")

# %% Define the training and testing data
# Initial condition 1
u0 = load_wf(wf_files[int(t0/dt)], pmln, wf_scale)
# Initial condition 2
u1 = load_wf(wf_files[int(t1/dt)], pmln, wf_scale)
# Test data
u_test = load_wf(wf_files[int(t_test/dt)], pmln, wf_scale)

# show the initial condition and test data
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for d, ax, title in zip([u0, u1, u_test], 
                        axes, 
                        [f'I.C. u(x,z,t={t0})', f'I.C. u(x,z,t={t1})', f'Test data u(x,z,t={t_test})']):
    ax.imshow(d, cmap='seismic', aspect='auto')
    ax.set_title(title)
    ax.axis('off')
plt.show()

rec = load_rec(cfg['geom']['obsPath'])[0][::rec_scale, :]
plt.figure(figsize=(3, 4))
plt.imshow(rec, cmap='seismic', aspect='auto')
plt.title('Record')
plt.axis('off')
plt.show()


# %% Define the coordinates of the training and testing data
# The training data and testing data share the same spatial coordinates
# because the velocity model is the same. The only difference is the time.
shape_train = u0.shape
gridz_u, gridx_u = shape_train
size_u = u0.size

# coordinates of the wavefields
xx_u, zz_u = torch.meshgrid(torch.linspace(0,nx,gridz_u,device=device),
                            torch.linspace(0,nz,gridx_u,device=device), 
                            indexing='ij')
xx_u = xx_u.reshape((-1,1))
zz_u = zz_u.reshape((-1,1))

# time coordinates of the wavefields
tt0 = torch.ones(size_u, 1, device=device)*(t0-t_offset)
tt1 = torch.ones(size_u, 1, device=device)*(t1-t_offset)
tt_test = torch.ones(size_u, 1, device=device)*(t_test-t_offset)

print(f"The pinn will be trained at {tt0.mean():.1f}s, {tt1.mean():.1f}s and tested at {tt_test.mean():.1f}s")

# coordinates of the records
recs = load_pkl(cfg['geom']['receivers'])[0]
recs = np.array(recs)*dh
num_recs = recs.shape[1]
recs = torch.from_numpy(recs).float().to(device)
rec_x, rec_z = recs

xx_r = torch.tile(rec_x, (nt_pinn,)).reshape((-1,1))
zz_r = torch.tile(rec_z, (nt_pinn,)).reshape((-1,1))
tt_r = torch.linspace(0, endt, nt_pinn, device=device).unsqueeze(1).repeat(1, num_recs).reshape((-1,1))

# coordinates of the source
srcs = load_pkl(cfg['geom']['sources'])[0]
wavelet = ricker_wave(cfg['geom']['fm'], dt, cfg['geom']['nt'], delay=cfg['geom']['wavelet_delay'], dtype='tensor')
wavelet = wavelet.to(device)
xx_w = torch.tensor(srcs[0], device=device, dtype=torch.float32).repeat(cfg['geom']['nt']).reshape((-1,1))
zz_w = torch.tensor(srcs[1], device=device, dtype=torch.float32).repeat(cfg['geom']['nt']).reshape((-1,1))
tt_w = torch.linspace(0, endt, cfg['geom']['nt'], device=device).unsqueeze(1).repeat(1, 1).reshape((-1,1))

plt.plot(tt_w.cpu().numpy(), wavelet.cpu().numpy())
plt.title('Source wavelet')
plt.show()

# collocate points in the domain for pde loss
points = lhs(3, pde_samples)
x_data, z_data, t_data = [], [], []

if use_pde:
    x_data.append(to_tensor(points[:,0]*nx).unsqueeze(1))
    z_data.append(to_tensor(points[:,1]*nz).unsqueeze(1))
    t_data.append(to_tensor(points[:,2]*end_t_in_pinn).unsqueeze(1))

if use_u0:
    x_data.append(xx_u.clone())
    z_data.append(zz_u.clone())
    t_data.append(tt0.clone())

if use_u1:
    x_data.append(xx_u.clone())
    z_data.append(zz_u.clone())
    t_data.append(tt1.clone())

if use_rec:
    x_data.append(xx_r.clone())
    z_data.append(zz_r.clone())
    t_data.append(tt_r.clone())

if use_wavelet:
    x_data.append(xx_w.clone())
    z_data.append(zz_w.clone())
    t_data.append(tt_w.clone())

x_pde = torch.concatenate(x_data)
z_pde = torch.concatenate(z_data)
t_pde = torch.concatenate(t_data)
x_pde.requires_grad = True
z_pde.requires_grad = True
t_pde.requires_grad = True

# %% Transfer the data and model to the device
pinn = PINN(layers)
opt_nn = torch.optim.Adam(pinn.parameters(),lr=lr,amsgrad=False)
opt_vel = torch.optim.Adam([v], lr=0.01, amsgrad=False)
pinn.to('cuda')
u0 = torch.from_numpy(u0.flatten()).float().to('cuda')
u1 = torch.from_numpy(u1.flatten()).float().to('cuda')
u_test = torch.from_numpy(u_test).float().to('cuda')
rec = torch.from_numpy(rec.flatten()).float().to('cuda')
pde_zero = torch.zeros(x_pde.shape[0], 1).to('cuda')

# %% Train the PINN
for i in tqdm.trange(epochs):

  opt_nn.zero_grad()
  opt_vel.zero_grad()

  # Initial condition 1
  if use_u0:
    u0_pred = pinn(xx_u, zz_u, tt0)
    loss_u0 = F.mse_loss(u0_pred, u0.unsqueeze(1))
  else:
    u0_pred = torch.zeros_like(u0)
    loss_u0 = 0.

  # Initial condition 2
  if use_u1:
    u1_pred = pinn(xx_u, zz_u, tt1)
    loss_u1 = F.mse_loss(u1_pred, u1.unsqueeze(1))
  else:
    u1_pred = torch.zeros_like(u1)
    loss_u1 = 0.

  # loss of the initial condition
  loss_ini = loss_u0 + loss_u1

  # loss of record
  if use_rec:
    rec_pred = pinn(xx_r, zz_r, tt_r)
    loss_rec = F.mse_loss(rec_pred, rec.unsqueeze(1))
  else:
    loss_rec = 0.

  # loss of the wavelet
  if use_wavelet:
    wavelet_pred = pinn(xx_w, zz_w, tt_w)
    loss_wavelet = F.mse_loss(wavelet_pred, -wavelet.unsqueeze(1))
  else:
    loss_wavelet = 0.

  # loss of the pde
  if use_pde:
    u_pde = pde(x_pde, z_pde, t_pde, vel=v)
    loss_pde = F.mse_loss(u_pde, pde_zero)
  else:
    loss_pde = 0.
  
  # total loss
  loss = loss_ini + 1e-4*loss_pde + loss_rec + loss_wavelet

  loss.backward()
  opt_nn.step()
  opt_vel.step()

  if v<0:
    v.data = torch.zeros_like(v.data)

  if i%1000==0:
    print(f"Epoch {i} |v: {v} | Loss {loss:.6f} | Loss_ini {loss_ini:.6f} | Loss_pde {loss_pde:.6f}")
    with torch.no_grad():
      utest_pred = pinn(xx_u, zz_u, tt_test)
    utest_pred = utest_pred.detach().cpu().numpy().reshape(shape_train)
    fig, ax = plt.subplots(2,3, figsize=(8,5))
    show_d = [u0, u1, u_test, u0_pred, u1_pred, utest_pred]
    extent = [0, nx/1000, nz/1000, 0]
    for d, ax in zip(show_d, ax.ravel()):
        if isinstance(d, torch.Tensor):
            d = d.detach().cpu().numpy().reshape(shape_train)
        ax.imshow(d, extent=extent, cmap='seismic', aspect='auto')
        ax.set_xlabel('x (km)')
        ax.set_ylabel('z (km)')
        # disable the axis
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# %% Predict the wavefield
dh = 10
rec_pred = []
grid1m_x = int(nx/(dh/1000))
grid1m_z = int(nz/(dh/1000))
xx, zz = np.meshgrid(np.linspace(0,nx,grid1m_x),np.linspace(0,nz,grid1m_z))
xx0 = to_tensor(xx.reshape((-1,1)))
zz0 = to_tensor(zz.reshape((-1,1)))
shape = xx.shape
# predict all the time
for t in range(nt):
    if t%25==0:
        tt = t*dt*np.ones(shape)
        tt = to_tensor(tt.reshape((-1,1)))
        with torch.no_grad():
            u_pred = pinn(xx0, zz0, tt).detach().cpu().numpy().reshape(shape)
        u_true = load_wf(wf_files[t], pmln=pmln, down_scale=1)
        fig, axes = plt.subplots(1,3,figsize=(6,3))
        axes[0].imshow(u_pred, extent=extent, cmap='seismic', aspect='auto')
        axes[1].imshow(u_true, extent=extent, cmap='seismic', aspect='auto')
        axes[2].imshow(u_pred-u_true, extent=extent, cmap='seismic', aspect='auto')
        for ax in axes.ravel():
            ax.set_title(f"Time {t*dt:.2f} s")
            ax.set_xlabel('x (km)')
            ax.set_ylabel('z (km)')
            ax.axis('off')
        plt.tight_layout()
        plt.show()

rec_pred = np.array(rec_pred)
plt.figure(figsize=(3, 4))
plt.imshow(rec_pred, cmap='seismic', aspect='auto')
plt.title('Predicted record')
plt.show()
plt.plot(rec_pred[:, 50])
plt.plot(rec[:, 50])
plt.show()
# %%
