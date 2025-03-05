import numpy as np
from scipy.fft import fft2, ifft2

import os
import sys

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
        
def rk3(r, nc, K, rhs1, rhs2):
    y21 = r + dt*rhs1(r, nc, K)
    y22 = nc + dt*rhs2(r, nc)
    y31 = 0.75*r + 0.25*(y21 + dt*rhs1(y21, y22, K))
    y32 = 0.75*nc + 0.25*(y22 + dt*rhs2(y21, y22))
    
    r_new = 1./3 * r + 2./3 * (y31 + dt*rhs1(y31, y32, K))
    nc_new = 1./3 * nc + 2./3 * (y32 + dt*rhs2(y31, y32))
    return r_new, nc_new

def rhs1_m(rho, n, K):
    v = alpha*n - rho*zeta
    tempx = v/(2.0*Dr) * ifft2(1J*K[0]*fft2(rho*v) *dealias) #+ np.sqrt(2*Dn/dt)*np.random.standard_normal(rho.shape)
    tempy = v/(2.0*Dr) * ifft2(1J*K[1]*fft2(rho*v) *dealias) #+ np.sqrt(2*Dn/dt)*np.random.standard_normal(rho.shape)
    
    return ifft2(1J*(K[0]*fft2(tempx) + K[1]*fft2(tempy)) *dealias)

def rhs2_vanilla(r, nc):
    return 0

def rhs2_BMR(r, nc):
    return I - l * r * nc

def rhs2_AMR(r, nc):
    v = alpha*nc - r*zeta
    return I - l * r * v * nc

def substep(r, nc, K2):
    rhat = fft2(r)
    nhat = fft2(nc)
    return np.real(ifft2(np.exp(-D * K2 * dt -kappa * K2**2 * dt)*rhat)), np.real(ifft2(np.exp(-Dc * K2 * dt)*nhat))


alpha = 1.0
D = 1
Dr = 3
v0 = 20
Dc = float(sys.argv[1])
l = float(sys.argv[2])
zeta = 22
rho = 0.5
kappa = 10
Dn = 0.0 ### no noise

if sys.argv[3] == 'BMR':
    I = l*rho*v0
elif sys.argv[3] == 'AMR':
    I = l*rho*(alpha*v0 - zeta*rho)*v0
else:
    I = 0
    
folder_name = sys.argv[3] + '_rho_20_zeta_22_Dc_{:.2f}_lm_{:.4f}'.format(Dc, l)
createFolder(folder_name)

Nsteps = 1000000
dt = 0.001

N = 200 # size of the lattice

r = np.zeros((N, N), dtype=np.float32)
nc = np.zeros((N, N), dtype=np.float32)
noise = 0.01

dx = 1.0 # lattice size
L = N*dx
x = np.linspace(0, L, N+1)[:N]
x, y = np.meshgrid(x, x, indexing='ij')

kx = ky = np.fft.fftfreq(N, d=dx)*2*np.pi
K = np.array(np.meshgrid(kx , ky ,indexing ='ij'), dtype=np.float32)
K2 = np.sum(K*K,axis=0, dtype=np.float32)
K4 = np.sum(K*K*K*K,axis=0, dtype=np.float32)

# The anti-aliasing factor  
kmax_dealias = kx.max()*2.0/3.0 # The Nyquist mode
dealias = np.array((np.abs(K[0]) < kmax_dealias )*(np.abs(K[1]) < kmax_dealias ),dtype =bool)

if sys.argv[3] == 'BMR':
    rhs2 = rhs2_BMR
elif sys.argv[3] == 'AMR':
    rhs2 = rhs2_AMR
else:
    rhs2 = rhs2_vanilla

# initial condition for r and n
rng = np.random.default_rng(1345) 
r0_0 = 0.5 * np.ones((N,N))
r0 = r0_0 + ifft2(fft2(0.01*np.random.standard_normal(r0_0.shape)) * dealias).real
n0 = v0 + 0.0*np.sin(2*np.pi*((x+y)/L))

r = r0
nc = n0

nplt = 50000
fig_index = 0

np.savetxt(folder_name + '/rho_{:d}.txt'.format(fig_index), r.flatten(), fmt='%.5f', delimiter=' ')
np.savetxt(folder_name + '/n_{:d}.txt'.format(fig_index), nc.flatten(), fmt='%.5f', delimiter=' ')

fig_index = 1

# time evolution
for n in range(1, Nsteps+1):
    r_star, nc_star = rk3(r, nc, K, rhs1_m, rhs2)
    r_new, nc_new = substep(r_star, nc_star, K2)

    r = r_new
    nc = nc_new
    t = n*dt

    # Plotting
    if np.mod(n,nplt) == 0:
        np.savetxt(folder_name + '/rho_{:d}.txt'.format(fig_index), r.flatten(), fmt='%.5f', delimiter=' ')
        np.savetxt(folder_name + '/n_{:d}.txt'.format(fig_index), nc.flatten(), fmt='%.5f', delimiter=' ')
        
        fig_index = fig_index + 1