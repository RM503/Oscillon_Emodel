import numpy as np
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)

phi_in = np.linspace(0.001, 0.12, 250)
k = np.linspace(0.1, 2.5, 250)

a = np.loadtxt('scale_factor.txt')
phi = 0.12/a**1.5
K = [0.5, 1.0, 1.5, 2.0, 2.5]


mu = np.loadtxt('floquet_Emodel_alpha_5x10^-4.txt')


X, Y = np.meshgrid(k, phi_in)

plt.contourf(X, Y, mu, 500, cmap='RdGy_r')
for i in range(5):
    plt.plot(K[i]/a, phi, c='white', linewidth=1)
plt.text(1.15, 0.075, r'$k(t)\propto a^{-1}$', fontsize=15, color='white', rotation=60)
plt.colorbar(format='%.2f').set_label(r'$\Re(\mu_k)/m$',fontsize=15)
plt.xlabel(r'$k[m]$', fontsize=15)
plt.ylabel(r'$\phi[M_{\rm{pl}}]$', fontsize=15)
plt.xlim(0.1, 2.5)
plt.savefig('floquet_chart_Emodel_alpha_5x10^-4.png', dpi=300, bbox_inches='tight')
plt.show()
