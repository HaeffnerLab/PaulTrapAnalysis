import numpy as np
from numpy.linalg import norm
from scipy.optimize import fsolve
from PaulTrapAnalysis.functions import gradients
from PaulTrapAnalysis.functions import potentials

import scipy.constants as const

e = const.e
pi = const.pi
epsilon0 = const.epsilon_0
#m = 40.078 * const.u
r0 = 1e-6 
m = const.m_e #for electron

def Coul(r1, r2, r0=1):
    dR01 = r1 - r2
    return e/(4*pi*epsilon0*r0)*(dR01)/(norm(dR01)**3)

def dCoul(r1, r2, r0=1):
    dR01 = r1 - r2
    return e/(4*pi*epsilon0*r0)*2*(dR01)/(norm(dR01)**4)

def get_rf_scale(l=1, o=2.5e9):
    return np.sqrt(e/m)/(2*l*o)

def func(zeta, N, Mj_DC, Mj_RF, rf_scale, order=4): # the equations
    zeta = np.reshape(zeta, (N,3))
    y=np.empty(shape=(N,3))
    for n in range(N):
        y[n] = -gradients.dPhi(zeta[n], Mj=Mj_DC, order=order) + \
               np.sum([Coul(zeta[n], zeta[j]) for j in range(N) if j != n], axis=0) + \
               get_pseudo_grad(zeta[n], Mj_RF, rf_scale, order=order)
    return y.flatten()

def EquilPos(N, Mj_DC, Mj_RF, order=4, l=1, o=2.5e9):
    rf_scale = get_rf_scale(l, o)
    return fsolve(lambda z: func(z, N, Mj_DC, Mj_RF, rf_scale, order=order), MultiTrialEqpos(N), xtol=1e-30).reshape((N,3))

def MultiTrialEqpos(ionNumber, center=0):
    sep = 1e-6
    inipos = np.linspace(-(ionNumber-1)/2*sep,(ionNumber-1)/2*sep,ionNumber)
    return [[0.,0.,i+center] for i in inipos]

def get_pseudo_grad(zeta, Mj, rf_scale, order=4):
    x0, y0, z0 = zeta
    x = tf.constant(x0, dtype=np.float64)
    y = tf.constant(y0, dtype=np.float64)
    z = tf.constant(z0, dtype=np.float64)
    with tf.GradientTape() as g:
        g.watch([x,y,z])
        V = potentials.get_pseudo_pot((x,y,z), Mj, rf_scale, order=order)

    grads = g.gradient(V, [x,y,z])
    return grads
