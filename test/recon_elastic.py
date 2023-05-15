
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from wavetorch.model import build_model
from wavetorch.checkpoint import checkpoint as ckpt
from wavetorch.source import WaveSource
from wavetorch.cell_elastic import _time_step, _time_step_backward
from wavetorch.utils import diff_using_roll, ricker_wave, to_tensor, save_boundaries, restore_boundaries
from wavetorch.utils import cpu_fft
NPML = 50
N = 2

dev = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"

pmld = np.load("/home/wangsw/Desktop/wangsw/fwi/pmld.npy")
pmld = torch.from_numpy(pmld).to(dev)
nt = 2001
pmln = 50
order = 2
fm = 5
dt = to_tensor(0.001)
dt2 = 0.001
dh = to_tensor(10).to(dev)
src_x = pmld.shape[1]//2
src_z = 53  # pmld.shape[1]//2
source_type = ["vz"]
source_func = WaveSource(src_z, src_x)
super_source = WaveSource([s.x for s in [source_func]],
                          [s.y for s in [source_func]]).to(dev)
# torch.ones_like(pmld).to(dev)*2500
vp = np.load("/mnt/data/wangsw/inversion/marmousi_20m/velocity/true_vp.npy")
# torch.ones_like(pmld).to(dev)*2500/1.73
vs = np.load("/mnt/data/wangsw/inversion/marmousi_20m/velocity/true_vs.npy")
src = ricker_wave(fm, dt2, nt, dtype="numpy")
src = cpu_fft(src, dt, 3, 5)
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

"""Forward"""
boundarys = []
wavefield = [torch.zeros((1, pmld.size(0), pmld.size(1))).to(dev)
             for _ in range(5)]
model_vars = [vp, vs, rho]
save_for_backward = []
forwards = []
save_interval = 100
record = []
x = src
x = x.unsqueeze(0)

_, rnn = build_model("../config/coding.yml", mode="inversion")
rnn.to(dev)

opt.zero_grad()
rnn.reset_sources(super_source)
record = rnn(x)


# for i, xi in enumerate(x.chunk(x.size(1), dim=1)):
#     wavefield = cell(wavefield,
#                     model_vars,
#                     is_last_frame=(i == x.size(1)-1),
#                     omega=1.0,
#                     source=[["vz"], super_source, xi.view(xi.size(1), -1)])
#     # wavefield = [w.copy_(d) for w,d in zip(wavefield, temp)]
#     # wavefield = ckpt(_time_step, _time_step_backward, [["vz"], super_source, xi.view(
#     #     xi.size(1), -1)], i == nt-1, len(model_vars), *model_vars, *wavefield, *[dt, dh, pmld])
#     wavefield = list(wavefield)
#     wavefield[1] = super_source(wavefield[1], xi.view(xi.size(1), -1))

#     np.save(f"/mnt/data/wangsw/inversion/marmousi_20m/elastic_testcode/forward/forward{np.random.randint(0, 1e6, 1)[0]}.npy",
#             wavefield[0].cpu().detach().numpy())
#     if i % save_interval == 0:
#         forwards.append(wavefield[1][0][50:-50, 50:-50].cpu().detach().numpy())
#     record.append(wavefield[1][..., src_z, 50:-50])

# record = torch.cat(record)
target = torch.rand(record.size()).to(dev)
loss = torch.nn.MSELoss()(record, target)
loss.backward()

# wavefield = _time_step(vp, vs, rho, *wavefield, dt, dh, pmld)
# wavefield = list(wavefield)
# wavefield[1] = source_func(wavefield[1], src[i])

# boundarys.append([save_boundaries(_wavefield, pmln, order) for _wavefield in wavefield])

# if i %save_interval==0:
#     forwards.append(wavefield[1][0][50:-50,50:-50].cpu().detach().numpy())

# if i==nt-1:
#     save_for_backward.append(wavefield)

# # wavefield = [torch.zeros_like(pmld) for _ in range(5)]
# print("Backward")
# wavefield = list(save_for_backward[0])
# backwards = []
# "Backward"
# for i in range(nt-1, -1, -1):
#     last_frame = i == nt-1

#     if last_frame:
#         continue

#     wavefield = _time_step_backward(*[vp, vs, rho], *wavefield, *[dt, dh, pmld],
#                                      boundarys[i],
#                                      [source_type, source_func, src[i]])
#     # wavefield[1][...,src_z,src_x] -= src[i]

#     if i %save_interval==0:
#         backwards.append(wavefield[1][0][50:-50,50:-50].cpu().detach().numpy())


fig, axes = plt.subplots(1, 3, figsize=(8, 3))
no = 2
_forward = forwards[no]
_backward = backwards[len(forwards)-no]
shows = [_forward, _backward, _forward-_backward]
vmin, vmax = np.percentile(shows[0], [2, 98])

for d, ax in zip(shows, axes.ravel()):
    vmin, vmax = np.percentile(d, [2, 98])
    axx = ax.imshow(d, vmin=vmin, vmax=vmax, cmap=plt.cm.gray, aspect="auto")
    plt.colorbar(axx)
plt.show()
