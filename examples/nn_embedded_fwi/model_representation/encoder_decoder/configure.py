# Configures for the model
nx = 128 # Number of grid points in x
nz = 64 # Number of grid points in z
radius = 10 # Radius of the anomaly
background_vp = 1500 # m/s
background_rho = 2000 #
anaomaly_vp = 2500 # m/s
anaomaly_rho = 3000 #
vp_vs_ratio = 1.73 # Vp/Vs ratio
src_x_step = 3 # Source spacing in x grid points
dh = 12.5 # m
std_vp = 200.
mean_vp = 2000.
std_rho = 50.
mean_rho = 2000.

# Configures for the wavelet
delay = 75 # ms
fm = 5 # Hz
dt = 0.002 # s
nt = 800 # timesteps

# Configures for the simulation
npml = 25 # grid points
srcz = 1+npml # grid point
recz = 1+npml # grid point

# Configures for the inversion
batch_size = 5
seed = 20240816
lr = 1e-3
lr_decay = 0.999
EPOCHS = 2001
show_every = 50