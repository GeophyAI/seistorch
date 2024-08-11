# Configures for the model
nx = 128 # Number of grid points in x
nz = 64 # Number of grid points in z
radius = 10 # Radius of the anomaly
background_vp = 1500 # m/s
anaomaly_vp = 2500 # m/s
vp_vs_ratio = 1.73 # Vp/Vs ratio
dh = 12.5 # m
std_vp = 200.
mean_vp = 2000.
std_rho = 50.
mean_rho = 2000.

# Configures for the wavelet
delay = 150 # ms
fm = 5 # Hz
dt = 0.002 # s
nt = 1000 # timesteps

# Configures for the simulation
npml = 50 # grid points
srcz = 1+npml # grid point
recz = 1+npml # grid point

# Configures for implicit neural network
in_features = 2 # x, z
out_features = 1 # vp, vs, rho
hidden_features = 128
hidden_layers = 8

seed = 20240809
lr = 1e-4
lr_decay = 0.999
EPOCHS = 2001
show_every = 100