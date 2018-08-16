import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

N = 3000
theta = np.random.uniform(low=0, high=2*math.pi, size=N)
phi = np.random.uniform(low=-math.pi/2, high=math.pi/2, size=N)
xs = np.multiply( np.cos(theta), np.cos(phi) )
ys = np.multiply( np.sin(theta), np.cos(phi) )
zs = np.sin(phi)

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
ax.scatter(xs, ys, zs, s=3)

uxs = np.random.normal(size=N)
uys = np.random.normal(size=N)
uzs = np.random.normal(size=N)
norm22s = uxs * uxs + uys * uys + uzs * uzs
norm2s = np.sqrt(norm22s)
nxs = np.divide(uxs, norm2s)
nys = np.divide(uys, norm2s)
nzs = np.divide(uzs, norm2s)

ax = fig.add_subplot(122, projection='3d')
ax.scatter(nxs, nys, nzs, s=3)

plt.show()
