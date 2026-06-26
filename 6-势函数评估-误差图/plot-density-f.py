import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

f=plt.figure(figsize=[8,8])
'''
x, y = np.loadtxt('detail_file.e.out', delimiter=' ', unpack=True)
xyE = np.vstack([x,y])
zE = gaussian_kde(xyE)(xyE)

scE = plt.scatter(x/179,y/179,c=zE,s=50)

plt.xlabel('real (eV/atom)',fontsize=15)
plt.ylabel('predict (eV/atom)',fontsize=15)

cbar = f.colorbar(scE)
#cbar.set_label("Z",fontsize=15)

#plt.xlim(-615.7,-615.55)
#plt.ylim(-615.7,-615.55)
plt.tick_params(axis='both',which='major',labelsize=12)
plt.savefig('e.png',dpi=1000)
'''
x1,y1,z1,x2,y2,z2 = np.loadtxt('detail_file.f.out', delimiter=' ', unpack=True)

xy = np.vstack([x1,x2])
z = gaussian_kde(xy)(xy)

sc = plt.scatter(x1,x2,c=z,s=15)
plt.xlabel('real (eV/A)',fontsize=20)
plt.ylabel('predict (eV/A)',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

#cbar = f.colorbar(sc)
#cbar.set_label("Z",fontsize=15)
plt.grid()

plt.xlim(-8,8)
plt.ylim(-8,8)

plt.savefig('f.png',dpi=1000)
