# Constants
model_scale = 1 # 1/2
expand = 50
expand = int(expand / model_scale)
delay = 150 # ms
fm = 3 # Hz
dt = 0.002 # s
nt = 2000 # timesteps
dh = 20 # m
pmln = 50
srcz = 5 + pmln # grid point
recz = 5 + pmln # grid point
lr = 20.
batch_size = 16
EPOCHS = 101
show_every = 50
srcx_step = 5

true_path = r"../../models/marmousi_model/true_vp.npy"
init_path = r"../../models/marmousi_model/linear_vp.npy"