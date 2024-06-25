import numpy as np
import sympy
import tensorflow as tf

from PaulTrapAnalysis.functions.spherical_harmonics import harmonics
from PaulTrapAnalysis.functions import data
from PaulTrapAnalysis.functions import gradients
from PaulTrapAnalysis.functions.expansion_utils import *

def manual_harm_basis_single_shot(x0, y0, z0, order, scale=1):
    '''
    Computes spherical harmonics at a single given point or point array.
   
    Returns: Yxx, rnorm
    where Yxx is a 1D array of the spherical harmonic evaluated on the grid
    rnorm is a normalization factor for the spherical harmonics

    The function returns the coefficients in the order:[C00 C10 C11c C11s ]'
    These correspond to the multipoles in cartesian coordinates: 
    ['C','Ey', 'Ez', 'Ex', 'U=xy', 'U4=yz', r'U2=z^2-(x^2+y^2)/2', 'U5=zx', r'U1=x^2-y^2']
     1    2     3     4      5        6               7               8           9  ..
    The normalization factors are dropped up to 2nd order.
    higher order terms ordering: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
     '''
    #change variables
    x = x0 / scale
    y = y0 / scale
    z = z0 / scale
    Yj_func = ['lambda x, y, z: np.ones(np.shape(x))'] + \
              [f'lambda x, y, z: {func}' for func in np.concatenate([harmonics[j] for j in range(1, order+1)])]
    Q = [eval(Y)(x, y, z) for Y in Yj_func]
    Q = np.transpose(Q)
    
    return Q, scale

def manual_harm_basis(r0, X, Y, Z, order, scale=1, rotate=False):
    """
    Computes spherical harmonics, just re-written matlab code
   
    Returns: Yxx, rnorm
    where Yxx is a 1D array of the spherical harmonic evaluated on the grid
    rnorm is a normalization factor for the spherical harmonics

    The function returns the coefficients in the order:[C00 C10 C11c C11s ]'
    These correspond to the multipoles in cartesian coordinates: 
    ['C','Ey', 'Ez', 'Ex', 'U=xy', 'U4=yz', r'U2=z^2-(x^2+y^2)/2', 'U5=zx', r'U1=x^2-y^2']
     1    2     3     4      5        6               7               8           9  ..
    Reference: https://docs.sympy.org/latest/modules/functions/special.html
    """
    x, y, z = data.get_grid(X, Y, Z, r0, rotate=rotate)
    return manual_harm_basis_single_shot(x, y, z, order, scale)

def generate_potential(r0, X, Y, Z, Mj, order, scale, library='manual', rotate=False):
    """
    Generates potential values evaluated at given coordinates.
    
    Parameters
    ----------
    - X, Y, Z : axis ranges for the potential grid
    - order : int, order of the expansion
    - method : str, choice of method to fit the coefficient, options are *lstsq* (least square), *ridge*, and *lasso*
    
    Returns
    -------
    - Phi : Potential values evaluated
    """
    if library == 'sympy':
        print('>>> Using SymPy Harmonic Basis')
        Yj, scale = sympy_basis_eval(r0, X, Y, Z, order, scale, rotate=rotate)
    elif library == 'manual':
        #print('>>> Using manual Harmonic Basis')
        Yj, scale = manual_harm_basis(r0, X, Y, Z, order, scale, rotate=rotate)
    else:
        print('>>> Using default Harmonic Basis')
        Yj, scale = spher_harm_basis(r0, X, Y, Z, order, scale=scale, rotate=rotate)
    Phi = spher_harm_cmp(Mj, Yj, scale, order)
    return Phi

def generate_potential_single_shot(x, y, z, Mj, order, scale, library='manual'):
    """
    Generates potential values evaluated at given coordinates.
    
    Parameters
    ----------
    - x, y, z : the actual spacepoints
    - order : int, order of the expansion
    - method : str, choice of method to fit the coefficient, options are *lstsq* (least square), *ridge*, and *lasso*
    
    Returns
    -------
    - Phi : Potential values evaluated
    """
    if library == 'sympy':
        print('>>> Using SymPy Harmonic Basis')
        Yj, scale = sympy_basis_eval_single_shot(x, y, z, order, scale)
    elif library == 'manual':
        #print('>>> Using manual Harmonic Basis')
        Yj, scale = manual_harm_basis_single_shot(x, y, z, order, scale)
    else:
        print('>>> Using default Harmonic Basis')
        Yj, scale = spher_harm_basis_single_shot(x, y, z, order, scale=scale)
    Phi = spher_harm_cmp(Mj, Yj, scale, order)
    return Phi

def spher_harm_cmp(Mj,Yj,scale,order):
    '''
    Regenerates the potential (V) from the spherical harmonic coefficients. 
    Modified from the PaulTrap code to avoid destructively modifying M.
    THIS IS DANGEROUS!!!!!!
    '''
    V = []
    Mj = np.array(Mj) # Creating a copy of Mj to avoid modification
    #unnormalize
    i=0
    for n in range(1,order+1):
        for m in range(1,2*n+2):
            i += 1
            Mj[i] = Mj[i]*(scale**n)
    W = np.dot(Yj,Mj)
    return np.real(W)

def sympy_basis_eval(r0, X, Y, Z, order, scale=1, rotate=False):
    """
    Computes spherical harmonics, just re-written matlab code
   
    Returns: Yxx, rnorm
    where Yxx is a 1D array of the spherical harmonic evaluated on the grid
    rnorm is a normalization factor for the spherical harmonics

    The function returns the coefficients in the order:[C00 C10 C11c C11s ]'
    These correspond to the multipoles in cartesian coordinates: 
    ['C','Ey', 'Ez', 'Ex', 'U=xy', 'U4=yz', r'U2=z^2-(x^2+y^2)/2', 'U5=zx', r'U1=x^2-y^2']
     1    2     3     4      5        6               7               8           9  ..
    Reference: https://docs.sympy.org/latest/modules/functions/special.html
    """
    x, y, z = data.get_grid(X, Y, Z, r0, rotate=rotate)
    return sympy_basis_eval_single_shot(x, y, z, order, scale)   
    
def sympy_basis_eval_single_shot(x, y, z, order, scale=1):
    '''
    Computes spherical harmonics at a single given point or point array using Sympy.
   
    Returns: Yxx, rnorm
    where Yxx is a 1D array of the spherical harmonic evaluated on the grid
    rnorm is a normalization factor for the spherical harmonics

    The function returns the coefficients in the order:[C00 C10 C11c C11s ]'
    These correspond to the multipoles in cartesian coordinates: 
    ['C','Ey', 'Ez', 'Ex', 'U=xy', 'U4=yz', r'U2=z^2-(x^2+y^2)/2', 'U5=zx', r'U1=x^2-y^2']
     1    2     3     4      5        6               7               8           9  ..
    Reference: https://docs.sympy.org/latest/modules/functions/special.html
    '''
    #change variables
    r = np.sqrt(x*x+y*y+z*z)
    r_trans = np.sqrt(x*x+y*y)
    phi = np.arctan2(r_trans,z)
    theta = np.arctan2(y,x)
    # for now normalizing as in matlab code
    
    #scale = 1#np.sqrt(np.amax(r)*dl)
    rs = r/(scale)
    Znm_lambda = lambda n,m,theta,phi: np.complex64(sympy.Znm(n, m, theta, phi).evalf()).real
    Znm = np.vectorize(Znm_lambda)
    
    Q = []

    j = 1
    #real part of spherical harmonics
    for n in range(0,order+1):
        #p = harmonics_dict(n, theta, phi)
        for m in range(-n,n+1):
            #print(f'{j} & {n} & {m} & ${sympy.latex(simplify(Znm(n,m,theta,phi).expand(func=True)))}$', r'\\ \hline')
            #j += 1
            Q.append((rs**n) * Znm(n,m,theta,phi))
            '''
            if m == 0:
                c = (rs**n)*Znm(n, m, theta, phi)
                Q.append(c.real)
            elif m < 0:
                c = 1j/np.sqrt(2) * (rs**n) * (Znm(n, m, theta, phi) - (-1)**m * Znm(n, m, theta, phi))
                Q.append(c.real)
            elif m > 0:
                c = 1/np.sqrt(2) * (rs**n) * (Znm(n, m, theta, phi) + (-1)**m * Znm(n, m, theta, phi))
                Q.append(c.real)
            '''
        #print()
    Q = np.transpose(Q)

    return Q, scale

def spher_harm_basis_single_shot(x, y, z, order, scale=1):
    '''
    Computes spherical harmonics at a single given point or point array.
   
    Returns: Yxx, rnorm
    where Yxx is a 1D array of the spherical harmonic evaluated on the grid
    rnorm is a normalization factor for the spherical harmonics

    The function returns the coefficients in the order:[C00 C10 C11c C11s ]'
    These correspond to the multipoles in cartesian coordinates: 
    ['C','Ey', 'Ez', 'Ex', 'U=xy', 'U4=yz', r'U2=z^2-(x^2+y^2)/2', 'U5=zx', r'U1=x^2-y^2']
     1    2     3     4      5        6               7               8           9  ..
    The normalization factors are dropped up to 2nd order.
    higher order terms ordering: https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
     '''
    #change variables
    r = np.sqrt(x*x+y*y+z*z)
    r_trans = np.sqrt(x*x+y*y)
    phi = np.arctan2(r_trans,z)
    theta = np.arctan2(y,x)

    # for now normalizing as in matlab code
    #dl = z[1]-z[0]
    #scale = 1#np.sqrt(np.amax(r)*dl)
    rs = r/(scale)

    Q = []

    #real part of spherical harmonics
    for n in range(0,order+1):
        p = harmonics_dict(n, theta, phi)
        for m in range(-n,n+1):
            if m == 0:
                c = (rs**n)*p[0]
                Q.append(c.real)
            elif m < 0:
                c = 1j/np.sqrt(2) * (rs**n) * (p[m] - (-1)**m * p[-m])
                Q.append(c.real)
            elif m > 0:
                c = 1/np.sqrt(2) * (rs**n) * (p[-m] + (-1)**m * p[m])
                Q.append(c.real)

    Q = np.transpose(Q)
    #print(np.shape(Q))
    return Q, scale


def get_pseudo_pot(zeta, Mj, rf_scale, order=4):
    assert len(Mj) in np.cumsum([3, 5, 7, 9, 11, 13, 15]), 'Length of Mj seems wrong! Mj should be the one used for gradients'
    x, y, z = zeta
    V = tf.norm(tf.transpose(gradients.dPhi_tf((x,y,z), Mj=Mj, order=order)), axis=-1)**2 * rf_scale**2
    return V
