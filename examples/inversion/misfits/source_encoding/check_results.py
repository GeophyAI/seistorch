import numpy as np
import matplotlib.pyplot as plt

losses = ["l2", "nim", "envelope", "integration", "w1d"]
pml = 50
expand = 50
true = np.load('../../../models/marmousi_model/true_vp.npy')[:,expand:-expand]
results=[]
for loss in losses:
    results.append(np.load(f'./{loss}/paravpF00E49.npy')[pml:-pml, pml+expand:-pml-expand])

fig, axes=plt.subplots(2,3,figsize=(12,6))
vmin,vmax=true.min(),true.max()
kwargs={"cmap":"seismic","aspect":"auto","vmin":vmin,"vmax":vmax}
for vel, ax, title in zip([true]+results, axes.ravel(), ["True"]+losses):
    ax.imshow(vel,**kwargs)
    ax.set_title(title)
plt.tight_layout() 