'''
This code computes the Floquet instability chart for inflationary preheating.
This code considers inflaton oscillating about an E-model potential
also neglecting the effects of expansion. 
'''


import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from scipy.optimize import root_scalar
from scipy import integrate
import matplotlib.pyplot as plt

'''
Define functions for the potential (V), its first derivative (dV) and its
second derivative (ddV)
'''
def V(phi, alpha):
    beta = np.sqrt(2/(3*alpha))
    return 0.75*alpha*(1 - np.exp(-beta*phi))**2


def V_for_root(phi, phi_in, alpha):
    beta = np.sqrt(2/(3*alpha))
    return 0.75*alpha*(1 - np.exp(-beta*phi_in))**2 - 0.75*alpha*(1 - np.exp(-beta*phi))**2

def dV(phi, alpha):
    beta = np.sqrt(2/(3*alpha))
    return np.sqrt(1.5*alpha)*np.exp(-beta*phi)*(1-np.exp(-beta*phi))

def ddV(phi, alpha):
    beta = np.sqrt(2/(3*alpha))
    return -np.exp(-2*beta*phi)*(-2 + np.exp(beta*phi))

def dT(phi, phi_in, alpha):
    return 1.0/np.sqrt(2*V(phi_in, alpha) - 2*V(phi, alpha))

'''
Computes the time period in the potential using energy conservation assuming
zero initial velocity; the formula is only applicable for both even/odd potentials
'''
#def TimePeriod(phi_in):
#    T = integrate.quad(dT, 0, phi_in, args=(phi_in,), epsabs=10e-10)[0]
#    return 2*T

def TimePeriod(phi_in, alpha):
    
    if V(phi_in, alpha) == V(-phi_in, alpha):
        
        T = integrate.quad(dT, 0, phi_in, args=(phi_in, alpha), epsabs=10e-10)[0]
        
        return 2*T
    
    elif V(phi_in, alpha) != V(-phi_in, alpha):
        
        sol = root_scalar(V_for_root, args=(phi_in, alpha), bracket=[0,-2], method='brentq')
        phi_min = sol.root 
        
        T1 = integrate.quad(dT, 0, phi_in, args=(phi_in, alpha), epsabs=10e-10)[0]
        T2 = integrate.quad(dT, phi_min, 0, args=(phi_min, alpha), epsabs=10e-10)[0]
        
        return T1 + T2

'''
Here the inflaton background is obtained by solving the following

phi''(t) + V'(phi) = 0

for some phi_in and phi'_in = 0. The field evolutions are obtained
using 4th order Runge-Kutta (RK4).
'''
def field_evolve(phi_in, T, alpha):
    
    dt = 0.01
    n = round(T/dt)
    phi = np.zeros(n)
    pi = np.zeros(n)
    
    phi[0] = phi_in
    pi[0] = 0
    
    for i in range(n-1):
        k0 = dt*pi[i]
        l0 = -dt*dV(phi[i], alpha)
        
        k1 = dt*(pi[i] + 0.5*l0)
        l1 = -dt*dV(phi[i] + 0.5*k0, alpha)
        
        k2 = dt*(pi[i] + 0.5*l1)
        l2 = -dt*dV(phi[i] + 0.5*k1, alpha)
        
        k3 = dt*(pi[i] + l2)
        l3 = -dt*dV(phi[i] + k2, alpha)
        
        phi[i+1] = phi[i] + (k0 + 2*k1 + 2*k2 + k3)/6
        pi[i+1] = pi[i] + (l0 + 2*l1 + 2*l2 + l3)/6
        
    return phi, pi

'''
Computes the Floquet exponents of the field perturbations; takes k, phi_array and 
T as inputs

The field perturbations dphi_k''(t) + (k^2 + V''(phi))dphi_k = 0 are solved using
orthogonal initial conditions using 4th order Runge-Kutta (RK4).
'''
def FloquetExponents(k, phi, T, alpha):
    
    dt = 0.01
    n = round(T/dt)
    
    #Define the orthogonal fields (dphi_1, dphi_2) and their derivatives
    
    dphi_1 = np.zeros(n)
    dpi_1 = np.zeros(n)
    dphi_2 = np.zeros(n)
    dpi_2 = np.zeros(n)
    
    dphi_1[0] = 1.0; dpi_1[0] = 0.0
    dphi_2[0] = 0.0; dpi_2[0] = 1.0
    
    for i in range(n-1):
        f0 = dt*dpi_1[i]
        F0 = - dt*( k**2 + ddV(phi[i], alpha) )*dphi_1[i]
        
        f1 = dt*(dpi_1[i] + 0.5*f0)
        F1 = - dt*( k**2 + ddV(phi[i], alpha) )*(dphi_1[i] + 0.5*F0)
        
        f2 = dt*(dpi_1[i] + 0.5*f1)
        F2 = - dt*( k**2 + ddV(phi[i], alpha) )*(dphi_1[i] + 0.5*F1)
        
        f3 = dt*(dpi_1[i] + f2)
        F3 = - dt*( k**2 + ddV(phi[i], alpha) )*(dphi_1[i] + F2)
        
        dphi_1[i+1] = dphi_1[i] + (f0 + 2*f1 + 2*f2 + f3)/6
        dpi_1[i+1] = dpi_1[i] + (F0 + 2*F1 + 2*F2 + F3)/6
        
        g0 = dt*dpi_2[i]
        G0 = - dt*( k**2 + ddV(phi[i], alpha) )*dphi_2[i]
        
        g1 = dt*(dpi_2[i] + 0.5*g0)
        G1 = - dt*( k**2 + ddV(phi[i], alpha) )*(dphi_2[i] + 0.5*G0)
        
        g2 = dt*(dpi_2[i] + 0.5*g1)
        G2 = - dt*( k**2 + ddV(phi[i], alpha) )*(dphi_2[i] + 0.5*G1)
        
        g3 = dt*(dpi_2[i] + g2)
        G3 = - dt*( k**2 + ddV(phi[i], alpha) )*(dphi_2[i] + G2)
        
        dphi_2[i+1] = dphi_2[i] + (g0 + 2*g1 + 2*g2 + g3)/6
        dpi_2[i+1] = dpi_2[i] + (G0 + 2*G1 + 2*G2 + G3)/6
    
    
    '''
    Eigenvalues of the time-evolved fundamental solution matrix
    '''
    LambdaPlus = 0.5*(dphi_1 + dpi_2) + 0.5*csqrt( (dphi_1 - dpi_2)**2 + 4*dphi_2*dpi_1 )
    LambdaMinus = 0.5*(dphi_1 + dpi_2) - 0.5*csqrt( (dphi_1 - dpi_2)**2 + 4*dphi_2*dpi_1 )
    
    '''
    muPlus and muMinus correspond to the positive and negative roots of the 
    eigenvalues; the function returns the greater of the two
    
    Re[mu^(+/-)] = ln(|Lambda^(+/-)|)/T
    '''
    muPlus = np.log(np.absolute(LambdaPlus))/T
    muMinus = np.log(np.absolute(LambdaMinus))/T
    
    muPlus = muPlus[-1]
    muMinus = muMinus[-1]
    
    return np.maximum(muPlus, muMinus)


if __name__ == "__main__":
   
    alpha = 0.0001
    ndim = 250
    phi_in = np.linspace(0.001, 0.06, ndim)
    k = np.linspace(0.1, 2.5, ndim)
    mu = np.zeros((ndim, ndim))
        
    for i in range(ndim):
        
        print("Current step: " + str(i+1))
        
        T = TimePeriod(phi_in[i], alpha)
        phi, pi = field_evolve(phi_in[i], T, alpha)
        
        for j in range(ndim):
            mu[i,j] = FloquetExponents(k[j], phi, T, alpha)

    #np.savetxt('floquet_Emodel.txt', mu)
