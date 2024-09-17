# Constants
model_scale = 2 # 1/2
expand = 50
expand = int(expand / model_scale)
delay = 150 # ms
fm = 5 # Hz
dt = 0.002 # s
nt = 1500 # timesteps
dh = 20 # m
pmln = 50
srcz = 5 + pmln # grid point
recz = 5 + pmln # grid point
lr = 0.025
std_vp = 1000. # Standard deviation of vp  for denormalization
mean_vp = 3000. # Mean of vp for denormalization
batch_size = 8
EPOCHS = 5001
show_every = 200
srcx_step = 1

true_path = r"../../../models/marmousi_model/true_vp.npy"
init_path = r"../../../models/marmousi_model/linear_vp.npy"