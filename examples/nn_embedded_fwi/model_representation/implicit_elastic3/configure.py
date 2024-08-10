# Configures for the model
nx = 128 # Number of grid points in x
nz = 64 # Number of grid points in z
radius = 10 # Radius of the anomaly
background_vp = 1500 # m/s
anaomaly_vp = 2500 # m/s
vp_vs_ratio = 1.73 # Vp/Vs ratio
dh = 12.5 # m
std_vp = 1000. # Standard deviation of vp  for denormalization
mean_vp = 3000. # Mean of vp for denormalization

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
out_features = 3 # vp, vs, rho
hidden_features = 128
hidden_layers = 6

# Configures for the optimization
lr = 5e-5
EPOCHS = 10000
show_every = 100