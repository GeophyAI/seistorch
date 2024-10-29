import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from siren import Siren, grid_init
# os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import jax, tqdm
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from utils_jax import *
import jax.random as random
from functools import partial
import optax, shutil
import flax.training.checkpoints as checkpoints
import sys, importlib

def main(key=None):

    if len(sys.argv) != 2:
        print("Usage: python main.py <config_file.py>")
        sys.exit(1)

    config_file = sys.argv[1].replace(".py", "")

    try:
        config = importlib.import_module(config_file)
    except ModuleNotFoundError:
        print(f"Config file {config_file}.py not found!")
        sys.exit(1)

    globals().update(vars(config))
    key = random.PRNGKey(RNG_KEY)

    # save_path = r'ifwi'
    os.makedirs(save_path, exist_ok=True)
    shutil.copy(sys.argv[1], f"{save_path}/configure.py")

    devices_count = len(jax.devices())
    devices = jnp.arange(devices_count)

    print('Running on {} devices'.format(devices_count))

    def get_sharding():
        devices = jax.devices()
        devices_count = len(devices)
        mesh = jax.sharding.Mesh(np.array(devices), ('devices',))
        input_spec = jax.sharding.PartitionSpec('devices', )
        dist_sharding = jax.sharding.NamedSharding(mesh, input_spec)
        replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        return dist_sharding, replicated_sharding

    dist_sharding, replicated_sharding = get_sharding()
    # Load velocity
    vel = np.load(true_path)/unit
    vel = vel[::model_scale_z, ::model_scale_x]
    print('Standard deviation of the velocity model:', vel.std())
    print('Mean of the velocity model:', vel.mean())
    print('Shape of the velocity model:', vel.shape)
    print(f'Size of the velocity model: {vel.shape[0]*dh}m x {vel.shape[1]*dh}m')

    fig,ax=plt.subplots(1, 1, figsize=(5, 3))
    plt.colorbar(ax.imshow(vel, vmin=vel.min(), vmax=vel.max(), cmap="seismic", aspect="auto"))
    plt.tight_layout()
    plt.savefig(f"{save_path}/true.png", dpi=300, bbox_inches="tight")
    plt.close()

    ori_domain = vel.shape
    ori_nz, ori_nx = ori_domain
    if not multiple:
        padding = ((pmln, pmln), (pmln, pmln))
        padding_domain = (ori_nz + 2*pmln, ori_nx + 2*pmln)
    else:
        padding = ((0, pmln), (pmln, pmln))
        padding_domain = (ori_nz + pmln, ori_nx + 2*pmln)

    pmlc = generate_pml_coefficients_2d(padding_domain, N=pmln, multiple=multiple).numpy()
    pmlc = jnp.array(pmlc)
    plt.imshow(pmlc, cmap="jet", aspect="auto")
    plt.colorbar()
    plt.savefig("pmlc.png", dpi=300, bbox_inches="tight")
    plt.close()

    wave = ricker(jnp.arange(nt) * dt - delay * dt, f=fm)
    tt = np.arange(nt) * dt
    plt.plot(tt, wave)
    plt.title("Wavelet")
    plt.close()

    # Frequency spectrum
    freqs = np.fft.fftfreq(nt, dt)[:nt//2]
    amp = np.abs(np.fft.fft(wave))[:nt//2]
    amp = amp[freqs <= 20] 
    freqs = freqs[freqs <= 20]
    plt.plot(freqs, amp)
    plt.title("Frequency spectrum")
    plt.close()

    # Geometry
    srcxs = np.arange(expand+pmln, padding_domain[1]-expand-pmln, srcx_step).tolist()
    srczs = (np.ones_like(srcxs) * srcz).tolist()
    src_loc = list(zip(srcxs, srczs))

    recxs = np.arange(expand+pmln, padding_domain[1]-expand-pmln, 1).tolist()
    reczs = (np.ones_like(recxs) * recz).tolist()
    rec_loc = list(zip(recxs, reczs))

    # Show geometry
    print(f"The number of sources: {len(src_loc)}")
    print(f"The number of receivers: {len(rec_loc)}")
    kwargs = dict(b=pmlc, domain=padding_domain, dt=dt, h=dh, recz=recz, pmln=pmln, spatial_order=spatial_order)
    rec_obs = jnp.load(rec_obs_path)
    show_gathers(rec_obs, figsize=(10, 6), save_path=save_path)

    """##############################################################"""
    """#######################INVERSION##############################"""
    """##############################################################"""

    # Training Loop
    SirenDef = Siren(num_layers=num_layers, hidden_dim=hidden_dim, final_activation='linear')

    lr_schedule = optax.exponential_decay(lr,1, lr_decay)
    opt = optax.adam(lr_schedule)

    grid = grid_init(ori_domain, jnp.float32)()

    @jax.pmap
    def init_step(key):
        grid = grid_init(ori_domain, jnp.float32)()
        variables = SirenDef.init(key, grid)
        if use_pretrain:
            variables = checkpoints.restore_checkpoint(ckpt_dir=ckptpath, target=variables)
        opt_state = opt.init(variables['params'])
        return variables, opt_state

    @jax.pmap
    def eval(variables):
        vp = SirenDef.apply(variables, grid).T
        vp = vp * std_vp + mean_vp
        return vp

    def loss(params, shot_nums):

        shot_nums = jnp.array(shot_nums)
        # Resample from the neural network
        vp = SirenDef.apply({'params': params}, grid).T
        vp = vp * std_vp + mean_vp
        vp = jnp.pad(vp, padding, mode='edge')

        # Forward modeling
        rec_syn = forward(wave, vp, src_list=jnp.array(src_loc)[shot_nums], **kwargs)
        return jnp.mean((rec_syn - rec_obs[shot_nums])**2)

    def compute_gradient(params, shot_nums=[1, 2, 3]):
        return jax.value_and_grad(loss)(params, jnp.array(shot_nums))

    @partial(jax.pmap, axis_name='devices', out_axes=(None, 0, 0, 0))
    def fwi_step(variables, opt_state, rand_shots):
        
        params = variables['params']

        _loss, gradient = compute_gradient(params, rand_shots)

        # sync gradient
        _loss = jax.lax.pmean(_loss, axis_name='devices')
        gradient = jax.lax.pmean(gradient, axis_name='devices')

        # update params
        updates, opt_state = opt.update(gradient, opt_state)
        params = optax.apply_updates(params, updates)
        variables['params'] = params
        return _loss, variables, opt_state, params

    LOSS = []
    key, init_key = jax.random.split(key)
    init_key = jnp.tile(init_key[None], (devices_count,1))
    variables, opt_state = init_step(init_key)

    # show the initial model
    initial = eval({'params': variables['params']})[0]
    # print(initial.max(), initial.min())
    np.save('init_by_siren.npy', (initial-mean_vp)/std_vp)
    fig,ax=plt.subplots(1, 1, figsize=(5, 3))
    plt.colorbar(ax.imshow(initial, vmin=vel.min(), vmax=vel.max(), cmap="seismic", aspect="auto"))
    plt.tight_layout()
    plt.savefig(f"{save_path}/initial.png", dpi=300, bbox_inches="tight")
    # exit()
    # variables = freeze(variables)
    for epoch in tqdm.trange(EPOCHS):

        key, subkey = random.split(key)

        rand_shots = random.randint(subkey, (batch_size,), 0, len(src_loc))
        rand_shots = rand_shots.reshape(devices_count, -1)
        rand_shots = jax.device_put(rand_shots, dist_sharding)

        _loss, variables, opt_state, params = fwi_step(variables, opt_state, rand_shots)
        LOSS.append(_loss)

        np.save(f"{save_path}/loss.npy", np.array(LOSS))

        if epoch % show_every == 0:
            # show vel
            # inverted = get_params(opt_state)#[pmln:-pmln, pmln:-pmln]
            inverted = eval({'params': params})[0]
            print(inverted.max(), inverted.min())
            extent = [0, inverted.shape[1]*dh, inverted.shape[0]*dh, 0]
            fig,ax=plt.subplots(1, 1, figsize=(5, 3))
            plt.colorbar(ax.imshow(inverted, vmin=vel.min(), vmax=vel.max(), extent=extent, cmap="seismic", aspect="auto"))
            plt.tight_layout()
            plt.savefig(f"{save_path}/invtered{epoch:05d}.png", dpi=300, bbox_inches="tight")
            plt.close()

            # show trace
            fig,ax=plt.subplots(1, 1, figsize=(5, 3))
            ax.plot(vel[:,100], label="True")
            ax.plot(inverted[:,100], label="Inverted")
            ax.legend()
            plt.tight_layout()
            plt.savefig(f"{save_path}/trace{epoch:05d}.png", dpi=300, bbox_inches="tight")
            plt.close()
            # show loss
            fig,ax=plt.subplots(1, 1, figsize=(5, 3))
            ax.plot(LOSS)
            ax.set_title("Loss")
            ax.set_xlabel("Epoch")
            ax.set_yscale("log")
            plt.tight_layout()
            plt.savefig(f"{save_path}/loss.png", dpi=300, bbox_inches="tight")
            plt.close()
    # show loss
    fig,ax=plt.subplots(1, 1, figsize=(5, 3))
    plt.plot(LOSS)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.yscale("log")
    plt.savefig(f"{save_path}/loss.png", dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main(None)