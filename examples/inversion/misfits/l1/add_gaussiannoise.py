import numpy as np
import matplotlib.pyplot as plt
shot_no = 40
obs = np.load('observed.npy', allow_pickle=True)
# show the data\
trace_clean=obs[shot_no][:,100].copy()
fig, axes = plt.subplots(1, 2, figsize=(6, 4))
vmin, vmax = np.percentile(obs[shot_no], [2, 98])
kwargs = {"cmap": "seismic", "aspect": "auto", "vmin": vmin, "vmax": vmax}
axes[0].imshow(obs[shot_no], **kwargs)
axes[0].set_title("Observed")

# add gaussian noise
for i in range(obs.shape[0]):
    obs[i] += np.random.normal(0, 0.5, obs[i].shape)

axes[1].imshow(obs[shot_no], **kwargs)
axes[1].set_title("Noise added")
plt.tight_layout()
plt.savefig('figures/gaussian/add_noise_profile.png')
plt.show()
trace_noisy=obs[shot_no][:,100]
# save the noisy data
# np.save('observed_gaussian.npy', obs)

fig,axes=plt.subplots(3,1,figsize=(5,8))
axes[0].plot(trace_clean, 'r', label='clean')
axes[0].legend()

axes[1].plot(trace_noisy, 'b', label='noisy')
axes[1].legend()

axes[2].plot(trace_noisy-trace_clean, 'g', label='noise')
axes[2].legend()

plt.tight_layout()
plt.savefig('figures/gaussian/add_noise_trace.png')

plt.show()
