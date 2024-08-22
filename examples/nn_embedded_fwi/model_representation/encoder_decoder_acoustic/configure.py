# Configures for the model
nx = 128 # Number of grid points in x
nz = 64 # Number of grid points in z
radius = 10 # Radius of the anomaly
background_vp = 1500 # m/s
anaomaly_vp = 2500 # m/s
dh = 12.5 # m
std_vp = 1000. # Standard deviation of vp  for denormalization
mean_vp = 3000. # Mean of vp for denormalization

# Configures for the wavelet
delay = 150 # ms
fm = 5 # Hz
dt = 0.002 # s
nt = 600 # timesteps

# Configures for the simulation
npml = 50 # grid points
srcz = 1+npml # grid point
recz = 1+npml # grid point

# Configures for implicit neural network
min_filters = 4
latent_length = 8

# Configures for the optimization
seed = 20240822
lr = 1e-1
lr_decay = 0.999
EPOCHS = 501
show_every = 50