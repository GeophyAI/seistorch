import numpy as np

import matplotlib.pyplot as plt

# with_top = np.load("with_top_d.npy")
# without_top = np.load("without_top.npy")

# plt.imshow(with_top)
# plt.show()

# plt.imshow(without_top)
# plt.show()
# plt.plot(without_top[:, 64])
# plt.show()

# wf = np.load("./wavefield2nd/wf0750.npy")
# plt.imshow(wf[0])
# plt.show()

# plt.plot(wf[0][:,64])
# plt.show()

wf = np.load("./wavefield/wf1050.npy")[0,:-50,50:-50]
plt.imshow(wf)
plt.show()

plt.plot(wf[:,64])
plt.show()
# rec = np.load("shot_gather.npy", allow_pickle=True)
# plt.imshow(rec[0], aspect="auto")
# plt.show()