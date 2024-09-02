# model path
# true_path = r"../../models/marmousi_model/true_vp.npy"
# smooth_path = r"../../models/marmousi_model/smooth_true_vp_for_rtm.npy"

nz, nx = 128, 256

seed = 20240901
model_scale = 2 # 1/2
expand = 50
delay = 150 # ms
fm = 20 # Hz
dt = 0.001 # s
nt = 1000 # timesteps
dh = 5 # m
bwidth = 50
srcz = 5+bwidth # grid point
recz = 5+bwidth # grid point
srcx_step = 5
batch_size = 8
# Training
lr = 0.01
epochs = 51
save_path = r'results/no_reg'
