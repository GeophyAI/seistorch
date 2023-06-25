import gc
import os

import matplotlib.pyplot as plt
import numpy as np


"""Objective loss function"""
# path = r"/public1/home/wangsw/FWI/EFWI/Marmousi/marmousi1_20m/compare_loss"
# path = r"/mnt/data/wangsw/inversion/marmousi_10m/elastic/compare_loss/"
path = r"/mnt/data/wangsw/inversion/marmousi_10m/elastic/good_init/compare_loss"
losses = ["l2", "l1"]
EXPAND = 100
# losses = ["lr3", "lr5", "lr7", "lr10"]

"Objective function"
# LOSS = []
# for loss_name in losses:
#     loss_path = os.path.join(path, loss_name, "loss.npy")
#     print(loss_path)
#     LOSS.append(np.load(loss_path))

# fig,ax=plt.subplots(1,1, figsize=(10,8))
# for _loss in LOSS:
#     for i in range(_loss.shape[0]):
#         _loss[i] /= _loss[i][0]
#     ax.plot(_loss.flatten())
# plt.legend(losses)
# plt.show()

"Model error"
true_vp = np.load("/mnt/data/wangsw/inversion/marmousi_10m/velocity/true_vp.npy")[:,EXPAND:-EXPAND]
true_vs = np.load("/mnt/data/wangsw/inversion/marmousi_10m/velocity/true_vs.npy")[:,EXPAND:-EXPAND]

FMAX = 6
EPOCHMAX = 49
PMLN = 50
VP_ERROR = []
VS_ERROR = []

for loss in losses:
    loss_dir = os.path.join(path, loss)
    temp_vp = []
    temp_vs = []
    # Calculate the model error in current folder
    model_error_vp_path = f"{loss_dir}/model_error_vp.npy"
    # if os.path.exists(model_error_vp_path):
    #     VP_ERROR.append(np.load(model_error_vp_path))
    #     continue
    model_error_vs_path = f"{loss_dir}/model_error_vs.npy"
    # if os.path.exists(model_error_vs_path):
    #     VS_ERROR.append(np.load(model_error_vs_path))
    for f in range(FMAX):
        for e in range(EPOCHMAX):
            vp = np.load(f"{loss_dir}/paravpF{f:02d}E{e:02d}.npy")[PMLN:-PMLN,PMLN+EXPAND:-PMLN-EXPAND]
            vs = np.load(f"{loss_dir}/paravsF{f:02d}E{e:02d}.npy")[PMLN:-PMLN,PMLN+EXPAND:-PMLN-EXPAND]
            temp_vp.append(np.sum((true_vp-vp)**2)/true_vp.size)
            temp_vs.append(np.sum((true_vs-vs)**2)/true_vp.size)
    VP_ERROR.append(temp_vp.copy())
    VS_ERROR.append(temp_vs.copy())
    np.save(model_error_vp_path, np.array(temp_vp))
    np.save(model_error_vs_path, np.array(temp_vs))
    
    del temp_vp, temp_vs
    gc.collect()

VP_ERROR = np.array(VP_ERROR)
VS_ERROR = np.array(VS_ERROR)

"Show the vp model error"
fig,ax=plt.subplots(1,1, figsize=(5,4))
for i, loss_name in enumerate(losses):
    ax.plot(VP_ERROR[i]/VP_ERROR[i][0])
plt.legend(losses)
plt.show()
fig,ax=plt.subplots(1,1, figsize=(5,4))
for i, loss_name in enumerate(losses):
    ax.plot(VS_ERROR[i]/VS_ERROR[i][0])
plt.legend(losses)
plt.show()




