# Configures for the model
seed = 20240828
nx = 128 # Number of grid points in x
nz = 64 # Number of grid points in z
radius = 10 # Radius of the anomaly
background_vp = 1500 # m/s
anaomaly_vp = 2500 # m/s
dh = 12.5 # m
std_vp = 1000. # Standard deviation of vp  for denormalization
mean_vp = 3000. # Mean of vp for denormalization

std_update = 5 # Standard deviation of the anomaly for denormalization
mean_update = 10 # Mean of the anomaly for denormalization

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
in_features = 2 # x, z
out_features = 1 # vp
hidden_features = 128
hidden_layers = 6

# Configures for the optimization
lr = 1e-4
lr2 = 1e-3
EPOCHS = 501
show_every = 50