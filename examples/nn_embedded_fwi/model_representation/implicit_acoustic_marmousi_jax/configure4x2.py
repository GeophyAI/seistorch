import time
RNG_KEY = 19491001
model_scale_x = 80
model_scale_z = 40
unit = 1.
expand = 0
expand = int(expand / model_scale_x)
delay = 150 # ms
multiple = False
fm = 5 # Hz
dt = 0.002 # s
nt = int(3./dt) # timesteps
dh = 25 # m
pmln = 50
spatial_order = 8
if multiple:
    srcz = 1 # grid point
    recz = 1 # grid point
else:
    srcz = 1 + pmln
    recz = 1 + pmln
lr_legacy = 25
lr = 0.0001
lr_decay = 0.9995
std_vp = 1000./unit # Standard deviation of vp  for denormalization
mean_vp = 2700./unit # Mean of vp for denormalization
batch_size = 8
EPOCHS = 2001
show_every = 200
srcx_step = 2

num_layers = 6
hidden_dim = 128
save_path = f'ifwi4x2_5hz/{time.strftime("%Y-%m-%d-%H-%M-%S")}'
# save_path = f'figures'
true_path = r"/ibex/user/wangs0j/models/elastic-marmousi-model/codes/MODEL_P-WAVE_VELOCITY_1_1.25m.npy"
rec_obs_path = r'rec_obs_4x2.npy'
# CKPT configures
# For legacy FWI
init_path = r"init_by_siren.npy"
use_pretrain = False
# ckptpath = '/ibex/user/wangs0j/seistorch/examples/nn_embedded_fwi/model_representation/implicit_vs_legacy/pretrain/ckpt'