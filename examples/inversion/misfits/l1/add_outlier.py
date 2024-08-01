import numpy as np
import matplotlib.pyplot as plt

missing_percentage = 0.3
obs = np.load('observed.npy', allow_pickle=True)
# show the data
shot_no = 40

fig, axes = plt.subplots(1, 2, figsize=(6, 4))
vmin, vmax = np.percentile(obs[shot_no], [2, 98])
kwargs = {"cmap": "seismic", "aspect": "auto", "vmin": vmin, "vmax": vmax}
axes[0].imshow(obs[shot_no], **kwargs)
axes[0].set_title("Observed")

# add gaussian noise
trace_length, trace_counts, _ = obs[0].shape
num_outliers_per_trace = 10
outlier_amplitude_min = 1.5
outlier_amplitude_max = 5.5
trace_have_outliers_percent = 0.3
for i in range(obs.shape[0]):
    trace_have_outliers = np.random.choice(trace_counts, int(trace_counts * trace_have_outliers_percent), replace=False)
    for j in trace_have_outliers:
        samples_have_outliers = np.random.choice(trace_length, num_outliers_per_trace, replace=False)

        outlier_amplitude = np.random.uniform(outlier_amplitude_min, outlier_amplitude_max)
        obs[i][samples_have_outliers, j] = outlier_amplitude


axes[1].imshow(obs[shot_no], **kwargs)
axes[1].set_title("Noise added")
plt.tight_layout()
plt.show()

# save the noisy data
np.save('observed_outlier.npy', obs)