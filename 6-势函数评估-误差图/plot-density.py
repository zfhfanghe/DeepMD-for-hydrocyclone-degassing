import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from scipy.stats import gaussian_kde

f=plt.figure(figsize=[8,8])

x, y = np.loadtxt('detail_file.e.out', delimiter=' ', unpack=True)
xyE = np.vstack([x,y])
zE = gaussian_kde(xyE)(xyE)

scE = plt.scatter(x/179,y/179,c=zE,s=30)

plt.xlabel('real (eV/atom)',fontsize=20)
plt.ylabel('predict (eV/atom)',fontsize=20)

#cbar = f.colorbar(scE)
#cbar.set_label("Z",fontsize=15)
plt.grid()
x_major_locator=MultipleLocator(0.05)
y_major_locator=MultipleLocator(0.05)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
#plt.xlim(-369.23,-369.17)
#plt.ylim(-369.23,-369.17)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

plt.tick_params(axis='both',which='major',labelsize=12)
plt.savefig('e.png',dpi=1000)
'''
x1,y1,z1,x2,y2,z2 = np.loadtxt('detail_file.f.out', delimiter=' ', unpack=True)

xy = np.vstack([x1,x2])
z = gaussian_kde(xy)(xy)

sc = plt.scatter(x1,x2,c=z,s=15)
plt.xlabel('real (eV/A)',fontsize=30)
plt.ylabel('predict (eV/A)',fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)

cbar = f.colorbar(sc)
#cbar.set_label("Z",fontsize=15)
plt.grid()

plt.xlim(-8,8)
plt.ylim(-8,8)

plt.savefig('f.png',dpi=1000)
'''
