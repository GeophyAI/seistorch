# Migrating from seistorch to SWEEP

[SWEEP](https://github.com/DeepWave-KAUST/sweep) is the successor to seistorch:
the same wave-equation-meets-autodiff idea, rebuilt as a clean Python API with a
compiled CUDA backend, 20+ equations, and PyTorch **and** JAX paths. This guide
maps every seistorch concept to its SWEEP equivalent.

> seistorch stays available (archived) so the results in
> *IEEE TGRS* 2023 — [10.1109/TGRS.2023.3317529](https://doi.org/10.1109/TGRS.2023.3317529)
> remain reproducible. New work should use SWEEP.

## Install

```bash
pip install .                                                     # Python-only
SWEEP_BUILD_CUDA=1 pip install -v ".[cuda]" --no-build-isolation  # compiled CUDA (recommended)
```

## The big change

seistorch is **YAML-config + shell driven**: you write model/geometry files to
disk, describe everything in a `.yml`, and run `forward.py` / `codingfwi.py`.
SWEEP is a **Python library**: models are arrays, geometry are arrays, and FWI
is an ordinary `torch` training loop. There is no config file — your script *is*
the config.

## Concept mapping

| seistorch (`template.yml` / CLI)              | SWEEP                                                        |
| --------------------------------------------- | ------------------------------------------------------------ |
| `equation: acoustic`                          | `from sweep.equations import Acoustic` (`Elastic`, `AcousticVTI`, `ElasticTTI`, …) |
| `backend: torch` / `jax`                      | `sweep.propagator.torch.PropTorch` / the JAX propagator      |
| `geom.h`, `geom.dt`, `geom.nt`                | `PropTorch(..., dh=, dt=)`; `nt` = wavelet length            |
| `geom.spatial_order`                          | `spatial_order=` on the propagator                           |
| `geom.boundary {type, width}`                 | `pml_type="cpmlr"` (type) + `abcn=` (ABC layers, default 50)  |
| **`geom.boundary_saving: true`**              | **`memory=MemoryOptions(strategy="boundary", boundary=BoundaryOptions())`** — see [Boundary saving](#boundary-saving) |
| `geom.truePath.{vp,vs,rho}` (`.npy`)          | plain arrays → `models=[vp, vs, rho]`                        |
| `geom.initPath` + `geom.invlist.{vp:true,…}`  | `requires_grad=True` only on the tensors you invert          |
| `geom.sources` / `geom.receivers` (`.pkl`)    | int arrays passed straight to `solver(...)`                  |
| `geom.wavelet` (`.npy`) + `wavelet_delay: 500`| `sweep.signal.ricker(t - 0.5, f=fm)` or load your own array  |
| `training.{lr, N_epochs, batch_size, minibatch}` | your `torch.optim` loop; `minibatch: true` = draw a random shot subset each epoch |
| `training.multiscale: [[1,3],5,all]`          | a staged band-pass loop (filter `obs`/`pred` per stage)     |
| loss `l2` / `ot` / `nim`                      | any `torch` loss / your own misfit on `(pred, obs)`         |
| `mpirun -f hosts python forward.py cfg.yml`   | a Python script; multi-GPU/MPI via `examples/multi-gpu`      |
| `python codingfwi.py inv.yml`                 | a ~10-line `torch` loop (see `examples/notebooks/00_hello_fwi.ipynb`) |

## Boundary saving

The memory-optimized **boundary-saving wavefield reconstruction** introduced in
the seistorch paper — *Memory Optimization in RNN-based FWI using Boundary Saving
Wavefield Reconstruction*, IEEE TGRS 2023
([10.1109/TGRS.2023.3317529](https://doi.org/10.1109/TGRS.2023.3317529)) — is a
first-class feature in SWEEP, and has been extended well beyond the original:

- seistorch's `geom.boundary_saving: true` becomes a `MemoryOptions` strategy.
- It works across **all backends** — PyTorch eager (`impl="eager"`), compiled
  C++/CUDA (`impl="c"`), and JAX — not just the RNN PyTorch path.
- Storage tiers: keep the saved boundaries on **GPU, CPU, or disk**, with
  optional **low-precision** storage (`fp16` / `bf16` / `int8`) to cut memory further.

```python
from sweep.propagator.options import MemoryOptions, BoundaryOptions, CkptOptions

# O(boundary) memory instead of O(nt) — reverse-time reconstruction,
# the method from the seistorch TGRS 2023 paper.
solver = PropTorch(
    Acoustic(device=dev), shape=shape, dh=dh, dt=dt, dev=dev, pml_type="cpmlr",
    memory=MemoryOptions(strategy="boundary", boundary=BoundaryOptions()),
)

# Offload to CPU and store in fp16 to save even more:
solver = PropTorch(
    Acoustic(device=dev), shape=shape, dh=dh, dt=dt, dev=dev, pml_type="cpmlr",
    memory=MemoryOptions(strategy="boundary",
                         boundary=BoundaryOptions(storage="cpu", storage_dtype="fp16")),
)
```

Gradient checkpointing is the other strategy:
`MemoryOptions(strategy="ckpt", ckpt=CkptOptions(...))`. For a memory-vs-accuracy
walkthrough see
[`examples/notebooks/07_memory_strategies.ipynb`](https://github.com/DeepWave-KAUST/sweep/blob/dev/examples/notebooks/07_memory_strategies.ipynb).

## Forward modeling — before & after

**seistorch**
```bash
python generate_model_geometry.py        # -> true_vp.npy, sources.pkl, receivers.pkl, wavelet.npy
# configs/acoustic.yml points at those files
mpirun -f hosts python forward.py configs/acoustic.yml --mode forward --num-batches 1 --use-cuda
```

**SWEEP**
```python
import numpy as np, torch
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

shape, dh, dt, nt = (96, 128), 10.0, 0.001, 7000
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
solver = PropTorch(Acoustic(device=dev), shape=shape, dh=dh, dt=dt, dev=dev,
                   pml_type="cpmlr")

t         = np.arange(nt) * dt
wavelet   = ricker(t - 0.5, f=15.0).astype(np.float32)            # wavelet.npy + wavelet_delay: 500
sources   = np.array([[64, 0]], dtype=np.int64)                   # sources.pkl   -> (nshots, ndim)
receivers = np.array([[[x, 0] for x in range(0, 128, 2)]], dtype=np.int64)  # receivers.pkl -> (nshots, nrec, ndim)
vp        = torch.tensor(np.load("true_vp.npy"), device=dev)      # truePath.vp

with torch.no_grad():
    obs = solver(wavelet, sources, receivers, models=[vp])
```

## FWI — before & after

**seistorch** — everything lives in `inversion.yml`, then:
```bash
python codingfwi.py inversion.yml
```

> Despite the name, `codingfwi.py` runs **standard (mini-batch) shot-based FWI**,
> not source encoding — `training.minibatch: true` just draws a random subset of
> shots per epoch (`np.random.choice`) and models them one at a time. (SWEEP has a
> source-encoding mode too, but it's separate and optional.)

**SWEEP** — the YAML `training:` block becomes an explicit loop:
```python
vp  = torch.tensor(np.load("init_vp.npy"), device=dev, requires_grad=True)  # initPath.vp + invlist.vp: true
opt = torch.optim.Adam([vp], lr=10.0)                                       # training.lr
for epoch in range(100):                                                    # training.N_epochs
    opt.zero_grad()
    pred = solver(wavelet, sources, receivers, models=[vp])
    loss = 0.5 * (pred - obs).pow(2).sum()                                  # L2; swap in your own misfit (OT, envelope, …)
    loss.backward()                                                         # opt into boundary saving via memory= (see above)
    opt.step()
```
Invert multiple parameters by giving each its own `requires_grad=True` tensor and
adding them to the optimizer; keep the rest as plain tensors (the `invlist` flags).

## Reproducing old seistorch results

The archived seistorch repo still runs as-is. Clone the tagged release and follow
the original example READMEs — nothing about SWEEP changes that.

## Citing

If you use SWEEP, please cite:

```bibtex
@misc{wang2026sweep,
  title  = {{SWEEP} ({S}eismic {W}ave {E}quation {E}xploration {P}latform):
            A Unified Solver Framework for Differentiable Wave Physics},
  author = {Wang, Shaowen and Alkhalifah, Tariq},
  year   = {2026},
  eprint = {2604.14189},
  archivePrefix = {arXiv},
  url    = {https://arxiv.org/abs/2604.14189},
}
```

The boundary-saving method itself originates from the seistorch paper —
IEEE TGRS 2023, [10.1109/TGRS.2023.3317529](https://doi.org/10.1109/TGRS.2023.3317529)
(see the seistorch README for its full citation).

## Help

Open an issue on the [SWEEP repo](https://github.com/DeepWave-KAUST/sweep/issues)
and mention which seistorch example you're porting.
