import numpy as np
import matplotlib.pyplot as plt
import h5py

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)

'''
Upload 3D energy snapshots
'''

EK_snapshot = h5py.File('kinetic_energy_snapshot_scalar.h5','r')
EG_snapshot = h5py.File('gradient_energy_snapshot_scalar.h5','r')
EV_snapshot = h5py.File('potential_energy_snapshot.h5','r')
E = np.loadtxt('average_energies.txt')

def rho_max(EK_array, EG_array, EV_array, rho):
    
    keys = [key for key in EK_snapshot['E_S_K_0'].keys()] # stores HDF5 keys in a list
    del_rho_max = []
    key_numeric = []
    
    for key in keys:
        
        EK = np.array(EK_array['E_S_K_0'][key])
        EG = np.array(EG_array['E_S_G_0'][key])
        EV = np.array(EV_array['E_V'][key])
        
        k = float(key)
        k = int(k)
        key_numeric.append(k)
        
        rho_ave = rho[10*k]
        rho_inhom = EK + EG + EV
        del_rho = (rho_inhom - rho_ave)/rho_ave
        
        '''
        Flatten the 3D del_rho array, sort it in descending order
        and compute the mean of the 10 most overdense points
        '''
        del_rho = del_rho.flatten()
        del_rho_sorted = -np.sort(-del_rho)
        del_rho_sorted_ave = np.average(del_rho_sorted[0:10])
        
        del_rho_max.append(np.max(del_rho_sorted_ave))
        
    return key_numeric, del_rho_max

time, del_rho_max = rho_max(EK_snapshot, EG_snapshot, EV_snapshot, E[:,4])

plt.scatter(time, del_rho_max, c='orange', edgecolors='k')
plt.xlabel(r'$\tilde{t}$', fontsize=15)
plt.ylabel(r'$\frac{\delta\rho}{\bar{\rho}}|_{\rm{max}}$', fontsize=15)
#plt.yscale('log')
plt.savefig('del_rho_max.jpeg', dpi=300, bbox_inches='tight')
plt.show()
