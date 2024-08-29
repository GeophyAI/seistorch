# Configures for the model
seed = 20240829
seed2 = 12345678
marmousi_path = '../../models/marmousi_model/true_vp.npy'
nx = 231
nz = 87
dh = 20 # m
std_vp = 1000. # Standard deviation of vp  for denormalization
mean_vp = 3000. # Mean of vp for denormalization

# Configures for geometry
src_x_step = 2

# Configures for the wavelet
delay = 150 # ms
fm = 5 # Hz
dt = 0.002 # s
nt = 1250 # timesteps

# Configures for the simulation
npml = 20 # grid points
srcz = 1+npml # grid point
recz = 1+npml # grid point

# Configures for implicit neural network
in_features = 2 # x, z
out_features = 1 # vp
hidden_features = 128
hidden_layers = 4

# Configures for the optimization
batch_size = 5
lr = 1e-4
EPOCHS = 2001
show_every = 50