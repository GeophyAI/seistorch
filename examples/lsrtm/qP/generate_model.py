import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

# 
nz = 151
nx = 301
vp = np.zeros((nz, nx), dtype=np.float32)
_theta = np.zeros((nz, nx), dtype=np.float32)

# theta = 30
def calculate_y(x0, y0, x1, theta):
    # 将角度 theta 转换为弧度
    theta_rad = np.deg2rad(theta)
    
    # 计算直线的斜率
    slope = -np.tan(theta_rad)
    
    # 计算当 x = x1 时对应的 y 值
    y1 = y0 + slope * (x1 - x0)
    
    return y1

def fill_in(tensor, points, value):
    """将多边形内部的区域填充为固定值"""
    polygon_path = Path(points)
    y, x = np.meshgrid(np.arange(tensor.shape[1]), np.arange(tensor.shape[0]))
    points = np.vstack((y.flatten(), x.flatten())).T
    inside_polygon = polygon_path.contains_points(points)
    inside_polygon = inside_polygon.reshape(tensor.shape)
    #plot the polygonx
    tensor[inside_polygon] = value
    return tensor

# x, y
layer1 = [(-1, -1), (nx, -1), (nx, 25), (-1, 25)]
layer1_vel = 2000
vp = fill_in(vp, layer1, layer1_vel)

layer2 = [(-1, 125), (nx, 125), (nx, nz), (-1, nz)]
layer2_vel = 4000
vp = fill_in(vp, layer2, layer2_vel)

layer3 = [(-1, 100), (100, 100), (110, 126), (-1, 126)]
layer3_vel = 3500
vp = fill_in(vp, layer3, layer3_vel)

theta = 30
c1 = (100, 100) # x, y
c2 = (150, calculate_y(*c1, 150, theta))
c3 = (110, 126) # y, x
c4 = (160, calculate_y(*c3, 160, theta))
layer4 = [c1, c2, c4, c3]

# for coord, label in zip(layer4, ['c1', 'c2', 'c4', 'c3']):
#     plt.plot(*coord, 'o', label=label)
# plt.legend()
# # inverse y
# plt.gca().invert_yaxis()
# plt.show()

layer4_vel = 3750
vp = fill_in(vp, layer4, layer4_vel)
_theta = fill_in(_theta, layer4, theta)

theta = 45
c1 = c2
c3 = c4
c2 = (170, calculate_y(*c1, 170, theta))
c4 = (190, calculate_y(*c3, 190, theta))
layer5 = [c1, c2, c4, c3]
layer5_vel = 3000
vp = fill_in(vp, layer5, layer5_vel)
_theta = fill_in(_theta, layer5, theta)

theta = 60
c1 = c2
c3 = c4
c2 = (185, calculate_y(*c1, 185, theta))
c4 = (215, calculate_y(*c3, 215, theta))
layer6 = [c1, c2, c4, c3]
layer6_vel = 2750
vp = fill_in(vp, layer6, layer6_vel)
_theta = fill_in(_theta, layer6, theta)

vp[0:26] = 2000.

vp[vp==0.] = 2500

# epsilon
vels = [3500, 3750, 3000, 2750]
epsilon = np.zeros((nz, nx))
for vel in vels:
    epsilon[vp == vel] = 0.15

# delta
delta = np.zeros((nz, nx))
for vel in vels:
    delta[vp == vel] = 0.08

fig, axes = plt.subplots(2, 2, figsize=(9, 5))
dh = 10
extent = [0, nx*dh, nz*dh, 0]
for ax, data, title in zip(axes.flatten(), [vp, _theta, epsilon, delta], [r'$v_p$', r'$\theta$', r'$\epsilon$', r'$\delta$']):
    vmin, vmax = data.min(), data.max()
    kwargs = dict(cmap='jet', vmin=vmin, vmax=vmax, extent=extent, aspect='auto')
    plt.colorbar(ax.imshow(data, **kwargs), ax=ax)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_title(title)
plt.tight_layout()
plt.savefig('figures/model.png')
plt.show()

import os
os.makedirs('models', exist_ok=True)

np.save('models/vp.npy', vp)
np.save('models/theta.npy', _theta)
np.save('models/epsilon.npy', epsilon)
np.save('models/delta.npy', delta)
np.save('models/zero.npy', np.zeros_like(vp))

seabed = np.ones_like(vp)
seabed[:20,:] = 0
np.save('models/seabed.npy', seabed)

direct_vp = np.ones_like(vp)*2000
np.save('models/direct_vp.npy', direct_vp)

from scipy.ndimage import gaussian_filter
vp_smooth = gaussian_filter(vp.copy(), sigma=2)
fig, axes= plt.subplots(1, 2, figsize=(8, 3))
vmin, vmax=vp.min(), vp.max()
extent = [0, nx*dh, nz*dh, 0]
kwargs = dict(cmap='jet', vmin=vmin, vmax=vmax, extent=extent, aspect='auto')
plt.colorbar(axes[0].imshow(vp, **kwargs), ax=axes[0])
axes[0].set_title('Original $v_p$')
plt.colorbar(axes[1].imshow(vp_smooth, **kwargs), ax=axes[1])
axes[1].set_title('Smoothed $v_p$')
for ax in axes:
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
plt.tight_layout()
plt.savefig('figures/smoothed_vp.png')
plt.show()

fig,ax=plt.subplots(1,1,figsize=(5,3))
plt.plot(vp[:,50], 'r', label='Original')
plt.plot(vp_smooth[:,50], 'b', label='Smoothed')
plt.xlabel('Depth (m)')
plt.legend()
plt.tight_layout()

np.save('models/vp_smooth.npy', vp_smooth)

true_m = 2*(vp-vp_smooth)/vp_smooth
np.save('models/true_m.npy', true_m)
fig,ax=plt.subplots(1,1,figsize=(5,3))
plt.imshow(true_m, cmap='gray', aspect='auto', extent=extent)
plt.colorbar()
ax.set_xlabel('x (m)')
ax.set_ylabel('z (m)')
ax.set_title('True m')
plt.tight_layout()
plt.savefig('figures/true_m.png')
plt.show()