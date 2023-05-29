
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from wavetorch.model import build_model
from wavetorch.checkpoint import checkpoint as ckpt
from wavetorch.source import WaveSource
from wavetorch.utils import diff_using_roll, ricker_wave, to_tensor, save_boundaries, restore_boundaries
from wavetorch.utils import cpu_fft
from wavetorch.utils import cpu_fft, read_pkl
from wavetorch.setup_source_probe import setup_rec_coords
NPML = 50
N = 2

dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

pmld = np.load("../pmld.npy")
pmld = torch.from_numpy(pmld).to(dev)
nt = 601
pmln = 50
order = 2
fm = 5
dt = to_tensor(0.001)
dt2 = 0.001
dh = to_tensor(10).to(dev)
src_x = pmld.shape[1]//2
src_z = 50  # pmld.shape[1]//2
source_type = ["vz"]
source_func = WaveSource(src_z, src_x)
super_source = WaveSource([s.x for s in [source_func]],
                          [s.y for s in [source_func]]).to(dev)
# torch.ones_like(pmld).to(dev)*2500
vp = np.load("/mnt/data/wangsw/inversion/marmousi_10m/velocity/true_vp.npy")
# torch.ones_like(pmld).to(dev)*2500/1.73
vs = np.load("/mnt/data/wangsw/inversion/marmousi_10m/velocity/true_vs.npy")
src = ricker_wave(fm, dt2, nt, delay=256, dtype="numpy")
#src = cpu_fft(src, dt, 3, 5)
# src = src.astype(np.float32)
plt.plot(src)
plt.show()
src = to_tensor(src).to(dev)

vp = np.pad(vp, ((pmln, pmln), (pmln, pmln)), mode="edge")
# vp = np.expand_dims(vp, 0)
vp = to_tensor(vp).to(dev)
vp = vp.requires_grad_(True)
opt = torch.optim.Adam([vp], lr = 10.0)
vs = np.pad(vs, ((pmln, pmln), (pmln, pmln)), mode="edge")
# vs = np.expand_dims(vs, 0)
vs = to_tensor(vs).to(dev)

rho = torch.ones_like(vp).to(dev)*2000
rec_list = read_pkl("../geometry/marmousi_obn_10m/receivers.pkl")
probes = setup_rec_coords(rec_list[0], 50)

x = src
x = x.unsqueeze(0)

_, rnn = build_model("../config/coding.yml", mode="inversion")
rnn.to(dev)

opt.zero_grad()
rnn.reset_sources(super_source)
rnn.reset_probes(probes)
record = rnn(x)

target = torch.rand(record.size()).to(dev)
loss = torch.nn.MSELoss()(record, target)
loss.backward()
