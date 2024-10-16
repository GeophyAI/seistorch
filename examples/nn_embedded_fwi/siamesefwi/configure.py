import torch
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 20240724 # random seed
model_scale = 2 # down scale the model by this factor
expand = 50 # the model has been padded with this number of grid points
expand = int(expand/model_scale) # cut the left & right side of the model
delay = 150 # ms
water_depth = 480 # m
fm = 3 # Hz
dt = 0.0019 # s
batch_size = 5 # batch size
pmln = 50
nt = 1500 # timesteps
reset_water = True # reset the velocity of water to 1500 m/s
dh = 20 # m
water_grid = int(water_depth/dh/model_scale)+pmln
srcz = 5+pmln # grid point
srcx_step = 2 # grid point
recz = 5+pmln # grid point
vmin = 1000.
vmax = 5500.
lr_vel = 20.
lr_cnn = 0.000
EPOCHS = 200
show_every = 10