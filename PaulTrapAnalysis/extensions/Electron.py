## General
from time import time
import numpy as np
#import pyvista as pv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import ticker
import pandas as pd
import warnings
import sympy
import os 
import multiprocessing


## Constants and parameters
#mport run
from scipy.constants import m_e, e, k

## BEM 
from bem import Result


## PaulTrapAnalysis
from PaulTrapAnalysis.functions import plotting, data
from PaulTrapAnalysis.functions import optimsaddle
from PaulTrapAnalysis.components.Trap import Trap
from PaulTrapAnalysis.components.Electrode import SimulatedElectrode
from PaulTrapAnalysis.components.Multipoles import MultipoleControl
from PaulTrapAnalysis.functions.fitting import fit_potential as fit_anharm
from PaulTrapAnalysis.functions.spherical_harmonics import harmonics


def big_plt_font():
    """
    plt.rcParams.update({'font.size': 14,
                         'lines.markersize': 12,
                         'lines.linewidth': 2.5,
                         'xtick.labelsize': 15,
                         'ytick.labelsize': 15,
                         'errorbar.capsize': 2})
    """
    plt.rcParams.update({'font.size': 14,
                         'lines.markersize': 12,
                         'lines.linewidth': 2.5,
                         'xtick.labelsize': 15,
                         'ytick.labelsize': 15,
                         'errorbar.capsize': 2})

def small_plt_font():
    """
    plt.rcParams.update({'font.size': 6,
                         'lines.markersize': 12,
                         'lines.linewidth': 1.5,
                         'xtick.labelsize': 8,
                         'ytick.labelsize': 8,
                         'errorbar.capsize': 2})
    """
    plt.rcParams.update({'font.size': 6,
                         'lines.markersize': 12,
                         'lines.linewidth': 1.5,
                         'xtick.labelsize': 8,
                         'ytick.labelsize': 8,
                         'errorbar.capsize': 2})


### Anharmonicity fitting ###

## anharmonicity fitting parameters from (2.5GHz, 400MHz, 70MHz)
## configuration using pseudopotential
param_dict = {13: np.array([1.1657717 , 2.09979309, 2.0979524 ]),
              21: np.array([1.21347716, 2.23375099, 1.12998487]),
              31: np.array([2.30580395, 6.2917902 , 2.07483269]),
              41: np.array([588.73900582,   2.26673198,   1.08106801]),
              42: np.array([2.90021948, 7.9022565 , 1.9576991 ]),
              43: np.array([1.01755929, 4.10382322, 1.02398896]),
              44: np.array([4.15190102, 8.00865745, 1.99670645])}

def func(x, k, a, c): 
    """
    Fitting polynomial function
    """
    A, C = x
    return k * A**a * C**c

def get_Mj(Mj0, j): 
    """
    Get Mj in SI unit.
    """
    order = get_order(j)
    return Mj0*(1e3)**order

def get_Mj0(Mj, j): 
    """
    Get unitless Mj, assuming r0 = 1mm
    """
    order = get_order(j)
    return Mj*(1e-3)**order

def get_order(j): 
    """
    Get the order of the spherical harmonics term j
    """
    if j > 36: 
        return 6 
    elif j > 25: 
        return 5 
    elif j > 16: 
        return 4 
    elif j > 9: 
        return 3 
    elif j > 4: 
        return 2 
    elif j > 1: 
        return 1 
    else: 
        return 0
    

def find_A(fz, T=40): 
    """
    Find the motional amplitude of an electron given frequency and
    temperature.
    
    Parameters
    ------
    fz : [MHz]
        Frequency in MHz
    T : [K]
        Temperature
        
    Returns
    ------
    A : [um]
        Motional amplitude in um
    """
    return np.sqrt(2*k*T/m_e/(2*np.pi*fz*1e6)**2)*1e6

def compute_shift(A, C3, C4, C5, C6):
    """
    Calculate the approximated frequency shift using the
    equation from Landau perturbation methods, assuming a 
    1D potential of U(z) = sum( c_i z^i ). Equation adopted
    from Joshua Goldman's PhD thesis, P43
    """
    a2 = -15/16 * (C3**2) + 3/4 * C4 
    a3 = C3 * a2 
    a4 = -2565/1024*(C3**4) + 645/128 * (C3**2)*C4 + \
        -105/32*C3*C5 + 15/16*C6
    a5 = (C5-2*C3*C4)*a2 + 2*C3*a4 
    
    return a2*A**2 + a3*A**3 + a4*A**4 + a5*A**5

def get_shift_analytical(A, c3_c2=0, c4_c2=0, c5_c2=0, c6_c2=0): 
    """
    Calculate the approximated frequency shift using the
    equation from Landau perturbation methods, assuming a 
    1D potential of U(z) = sum( c_i z^i ). Equation adopted
    from Joshua Goldman's PhD thesis, P43

    Parameters
    ------
    A : [um]
        The amplitude of motion in um
    c3_c2 : [um^{-1}]
        The ratio c3/c2 in um^{-1}
    c4_c2 : [um^{-2}]
        The ratio c4/c2 in um^{-2}
    c5_c2 : [um^{-3}]
        The ratio c5/c2 in um^{-3}
    c6_c2 : [um^{-4}]
        The ratio c6/c2 in um^{-4}
    
    Returns
    ------
    The total relative frequency shift

    Plots
    -----
    Contribution of each c_i term to the frequency shift. Each
    bar in the plot represents the frequency shift with c_i and
    c_{j!=i} = 0. Note that the sum of all the bar values can be
    less than the total frequency shift, as the contribution from
    c_i and c_j may not be independent.
    """
    C3 = c3_c2 
    C4 = c4_c2
    C5 = c5_c2
    C6 = c6_c2

    shift_3 = compute_shift(A, C3, 0, 0, 0)
    shift_4 = compute_shift(A, 0, C4, 0, 0)
    shift_5 = compute_shift(A, 0, 0, C5, 0)
    shift_6 = compute_shift(A, 0, 0, 0, C6)

    big_plt_font()
    fig, ax = plt.subplots(figsize=(12, 4))
    all_shifts = [abs(shift_3), abs(shift_4), abs(shift_5), abs(shift_6)]
    ax.bar(range(len(all_shifts)), all_shifts)
    add_value_labels_log(ax, threshold=0)
    tick_name = [r'$z^3$', r'$z^4$', r'$z^5$', r'$z^6$']
    ax.set_xticks(range(len(all_shifts)), tick_name, rotation = 0)
    ax.set_ylabel(r'$\Delta \omega/\omega$')
    ax.set_yscale('log')
    ax.grid()
    ax.set_title(f'Relative frequency shift at {A:.0f}um motional amplitude')
    plt.tight_layout()
    plt.show()

    return compute_shift(A, C3, C4, C5, C6)
    

def get_shift(j, Mj, M2, A, param_dict=param_dict): 
    """
    Get the relative frequency shift of a spherical 
    harmonics term j based on the fitting coefficient Mj
    and the given coefficient for U2 (denoted as M2).
    
    Parameters
    ---
    j : int
        The spherical harmonics term
    Mj : [unitless]
        The unitless Mj with r0 = 1mm assumed
    M2 : [unitless]
        Coefficient for U2. For reference, 70MHz is M2=0.55
    A : [um]
        Electron motional amplitude in um!
    param_dict : dict
        Fitting coefficients for frequency shift at a given
        Mj and motional amplitude
    """
    order = get_order(j) 
    C = Mj/M2 * (1e-3)**(order-2)
    p = param_dict[j] 
    return func((A, C), *p)

def add_value_labels_log(ax, spacing=0.1, threshold=0.01):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        if abs(y_value) < threshold:
            continue
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1e}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.

def plot_frequency_shift(Mj, A=10,
            mutipole_names=['C', 'Ey', 'Ez', 'Ex', 'U3', 'U4', 'U2', 'U5', 'U1'], 
            shift_threshold=0):
    """
    Plots the frequency shift given a list of Mj values and motional amplitude A.
    
    Parameters
    ---
    Mj : list
        Unitless Mj coefficients
    A : [um] 
        Motional amplitudes in um!
    shift_threshold : [unitless]
        Relative frequency shift threshold value. Shifts below this threshold
        will not be denoted in the plot
    """
    fig, ax = plt.subplots(figsize=(0.3*len(Mj), 4))
    all_j = [i for i in param_dict if i < len(Mj)]
    M2 = abs(Mj.flatten()[6])
    all_shifts = [get_shift(j, abs(Mj.flatten()[j-1]), M2, A, param_dict) for j in all_j]
    ax.bar(range(len(all_j)), all_shifts)
    add_value_labels_log(ax, threshold=shift_threshold)
    tick_name = list(mutipole_names)
    tick_name += list(range(len(tick_name)+1, len(Mj)+1))
    tick_name = [tick_name[i-1] for i in all_j]
    ax.set_xticks(range(len(all_j)), tick_name, rotation = -90)
    ax.set_xlabel(r'$j$')
    ax.set_ylabel(r'$\Delta \omega/\omega$')
    ax.set_yscale('log')
    ax.grid()
    ax.set_title(f'Relative frequency shift at {A:.0f}um motional amplitude')
    plt.tight_layout()
    plt.show()





##################################################################################################################
################################### Needed for electron trap analysis pipeline ###################################
##################################################################################################################
 

################## Import trap data ##################

def import_trap_data(prefix, electrodes=[], elec_file_lookup={}, scale=10, 
                     file_type='vtk', debug=False):
    '''
    Import the data and create Electrode object, assemble 
    the Trap object from the Electrodes and parameters. Rename 
    the central Electrode from RF0 to DC0

    Parameters
    ----------
    prefix : str
        The prefix of the simulation output files (.vtk, .fld, .csv)
    electrodes : list 
        The list of electrodes to load the data from. Note that this 
        name should match the name used in file namings.
    elec_file_lookup : dict 
        The dict of electrode files if the file names do not agree with
        the electrode names. Note that this name can be any
        str as desired, as long as the key matches an existing elec.
        elec_file_lookup = {electrode_name: custom_file_name}
        e.g. elec_file_lookup = {'DC01': 'RF01'}
    scale : float
        Default is 10. scale = new/old = 1 mm / 100 um, 
        100um: a0 in bem simulation, 1mm: unit in multipole expansion

    Returns
    -------
    etrap : Trap object
    '''
    electrode_list = electrodes
    etrap = Trap()
    for elec in electrode_list:
        if debug:
            print(elec)
        if elec in elec_file_lookup: 
            elec_name = elec_file_lookup[elec]
        else: 
            elec_name = elec
        if file_type == 'vtk':
            etrap.update_electrodes(
                SimulatedElectrode.from_vtk(elec_name=elec, 
                                            scale=scale, 
                                            file=prefix+'_'+elec_name +'.vtk',
                                            maxderiv=2
                                        )
            )
        elif file_type == 'fld': 
            etrap.update_electrodes(
                SimulatedElectrode.from_fld(elec_name=elec, 
                                            scale=scale, 
                                            file=prefix+'_'+elec_name +'.fld',
                                            maxderiv=2
                                        )
            )
        else: 
            print(">>> Unknown file type, currently only supports .vtk and .fld. Please modify Electron extension for updates.")
    etrap.scale = scale
    etrap.electrode_file_lookup = elec_file_lookup
    etrap.prefix = prefix
    etrap.file_type = file_type
    return etrap

def import_rf_data(trap, prefix, suffix, file_type='vtk'):
    """
    Import the resulting RF data from simulated .vtk files.
    """
    t0 = time()
    if file_type == 'vtk':
        result_rf0 = Result.from_vtk(prefix+suffix, 'RF0')
        result_rf1 = Result.from_vtk(prefix+suffix, 'RF1')
        result_rf2 = Result.from_vtk(prefix+suffix, 'RF2')
    elif file_type == 'fld': 
        raise NotImplementedError
    print("Computing time: %f s"%(time()-t0))

    return result_rf0, result_rf1, result_rf2


# This block might need to import BEM
def view_result(prefix, suffix, electrode_name='RF1'):
    '''
    view the vtk file in pyvista, refer to result.view function in bem simulation notebook
    '''
    multiprocessing.Process(target=Result.view, args=(prefix+suffix, electrode_name)).start()



################# Pseudo-potential ##################
def PseudoPotential(field_square,m,Ω,V):
    ''' 
    Given the electric field square, 
    return the approximated pseudopotential in eV
    '''
    return field_square*e**2*V**2/(4*m*Ω**2)*6.242E18
    
def assign_trap_rf_single_pair(trap, prefix, suffix,  
                   V0_rf, electrodes='RF1',
                   l0=100e-6, Ω_e=2*np.pi*2.5e9):
    """
    Assign the imported RF data to the trap, and 
    calculate the pseudopoential. This assumes that the 
    trap only has one pair of RF electrodes.
    
    Parameters
    ----------
    trap : Trap object
        The trap to assign info to
    rf_data : [result_rf0, result_rf1, result_rf2]
        The imported RF result data from .vtk files
    V0_rf : float
        The amplitude for the RF drive
    l0 : [m]  #FIXME
        The length scale of the simulation grid, 
        default is 100e-6 (100 um)

    Returns
    -------
    trap : Trap object
        The updated trap 
    """
    scale = l0 
    trap.l0 = l0
    
    if trap.file_type == 'vtk':
        result_rf1 = Result.from_vtk(prefix+suffix, electrodes)
        rf1_field = result_rf1.field / scale
        rf_potential = V0_rf*result_rf1.potential 
        pp_rf = np.square(rf1_field).sum(axis=0).reshape(result_rf1.grid.shape)
    else:
        result_rf1 = trap.electrodes[electrodes].potential()
        rf1_field = trap.electrodes[electrodes].potential(derivative=1) / scale
        rf_potential = V0_rf*np.array(result_rf1)[:,:,:,0]
        pp_rf = np.array(np.square(rf1_field).sum(axis=-1)).reshape(np.shape(rf_potential))

    trap.result_rf1 = result_rf1
    trap.rf_field = rf1_field 
    
    trap.V0_rf = V0_rf
    trap.rf_potential_data = rf_potential
    
    trap.rf_field_square = pp_rf
    pp_rf = PseudoPotential(pp_rf,m_e,Ω_e,V0_rf)
    trap.pp_rf = pp_rf 

    return trap


def assign_trap_rf_two_pairs(trap, prefix, suffix,  
                   V0_rf, drive_method=[1,-1], 
                   electrodes=['RF1', 'RF2'],
                   l0=100e-6, Ω_e=2*np.pi*2.5e9):
    """
    Assign the imported RF data to the trap, and 
    calculate the pseudopoential. This assumes that the 
    trap only has one pair of RF electrodes.
    
    Parameters
    ----------
    trap : Trap object
        The trap to assign info to
    rf_data : [result_rf0, result_rf1, result_rf2]
        The imported RF result data from .vtk files
    V0_rf : float
        The amplitudes for the RF drive
    drive_method : list
        The drive method for the RF pairs. If out
        of phase, then this is [-1, 1]
    l0 : [m]  #FIXME
        The length scale of the simulation grid, 
        default is 100e-6 (100 um)

    Returns
    -------
    trap : Trap object
        The updated trap 
    """
    scale = l0 
    trap.l0 = l0
    
    if trap.file_type == 'vtk':
        result_rf1 = Result.from_vtk(prefix+suffix, electrodes[0])
        rf1_field = result_rf1.field / scale
        result_rf2 = Result.from_vtk(prefix+suffix, electrodes[1])
        rf2_field = result_rf2.field / scale
        rf1_potential = result_rf1.potential
        rf2_potential = result_rf2.potential
        shape = result_rf1.grid.shape
        axis = 0  # NOTE: Result field data is shaped as (dim, x, y, z)
    else: 
        result_rf1 = trap.electrodes[electrodes[0]].potential()
        rf1_field = trap.electrodes[electrodes[0]].potential(derivative=1) / scale / trap.scale
        rf1_potential = V0_rf*np.array(result_rf1)[:,:,:,0]
        result_rf2 = trap.electrodes[electrodes[1]].potential()
        rf2_field = trap.electrodes[electrodes[1]].potential(derivative=1) / scale / trap.scale
        rf2_potential = V0_rf*np.array(result_rf2)[:,:,:,0]
        shape = np.shape(rf1_potential)
        axis = -1  # NOTE: electrode field data is shaped as (x, y, z, dim)

    trap.result_rf1 = result_rf1
    trap.result_rf2 = result_rf2
    
    v1, v2 = drive_method
    rf_op = v1*rf1_field + v2*rf2_field
    trap.rf_field = rf_op
    
    rf_potential = V0_rf*(v1*rf1_potential + v2*rf2_potential)
    trap.V0_rf = V0_rf
    trap.rf_potential_data = rf_potential
    
    pp_rf = np.array(np.square(rf_op).sum(axis=axis)).reshape(shape)
    trap.rf_field_square = pp_rf
    pp_rf = PseudoPotential(pp_rf,m_e,Ω_e,V0_rf)
    trap.pp_rf = pp_rf 

    
    return trap



################# Trap height ##################
def analyze_trap_height(trap, find_saddle=True):
    '''
    Analyze the trap height
    
    Parameters
    ----------
    rf_data : list of rf result data
        The imported RF result data from .vtk files
    V_rf : ndarray
        Potential drive method: rf1, rf2, out-of-phase. 
        This is an rf voltage list of [V1,V2]. [1, -1]
        represents out of phase, [1, 0] represents only 
        using rf1.
    find_saddle : bool
        Default is True, if True: fit to find the real 
        saddle , if False: find the min value along all 
        the gridpoints

    Returns        
    -------
    rf_null_z : [um]
        Trap height, defined by the rf null point along the 
        z direction
    index : int
        Index of the trap height in the z direction (closet 
        to trap height if the height is not a grid point)
    '''
    if trap.file_type == 'vtk':
        coord_x, coord_y, coord_z = trap.result_rf1.grid.to_xyz()
        trap.x, trap.y, trap.z = coord_x, coord_y, coord_z
    else: 
        coords = trap.result_rf1.coords
        coord_x, coord_y, coord_z = np.array(coords['x']) * trap.scale, \
                                    np.array(coords['y']) * trap.scale, \
                                    np.array(coords['z']) * trap.scale
    
    
    pp_rf = trap.pp_rf
    scale = trap.l0
    a = int(np.floor(pp_rf.shape[0]/2))
    b = int(np.floor(pp_rf.shape[1]/2))
    index, rf_null = FindRFNull(pp_rf,coord_z,scale, a, b)
    rf_null_r0 = [coord_x[a], coord_y[b], coord_z[index]]

    if find_saddle:
        exact_rf_null = optimsaddle.exact_saddle(trap.rf_potential_data, coord_x, coord_y, coord_z, dim=3, scale=scale)
        rf_null_x, rf_null_y, rf_null_z = np.array(exact_rf_null)*scale/1e-6  # now rf_null_z and rf_null are both in [um]
        print(f"Approximated RF Null is at ({rf_null_r0[0]:.2f}, {rf_null_r0[1]:.2f}, {rf_null:.2f}) um, exact RF null is at ({rf_null_x:.2f}, {rf_null_y:.2f}, {rf_null_z:.2f}) um")
    else:
        rf_null_z = rf_null
        print(f"Approximated RF Null is at {rf_null:.2f} um")

    plt.plot(coord_z[:]*scale*1e6, pp_rf[a, b, :])
    plt.xlabel("Trap height (um)")
    plt.ylabel("Pseudo potential (eV)")
    plt.axvline(x=rf_null_z, color = 'grey', linestyle = '-.',label=f"h = {rf_null_z:.2f} um")
    plt.grid()
    plt.legend()
    
    print("z index of rf null is %s"%index)

    trap.rf_null_z = rf_null_z 
    trap.rf_null_index = index 
    
    return rf_null_z, index

def FindRFNull(pp_rf, z, scale, a=None, b=None):
    '''Given the pesodupential, return the approximate z axis coordinate of rf null in um
    A more accurate way needs to use the findsaddle method
    '''
    a = int(np.floor(pp_rf.shape[0]/2)) if a is None else a
    b = int(np.floor(pp_rf.shape[1]/2)) if b is None else b
    p = pp_rf[a, b, :]
    argmax = np.argmax(p)
    if argmax == len(p)-1:
        argmax = 0
    index = np.argmin(p[argmax:])
    return index+argmax, z[index+argmax]*scale*1e6


################# d_eff ##################
def analyze_d_eff(trap, dc_electrode='DC12', direction='y', 
                  force_center=False):
    '''
    Calculate the effective distance of the pickup electrode.

    Parameters
    ----------
    result_rf0 : rf data
        The BEM-simulated result for the RF pickup electrode
    z_null : [um]
        The RF null height
    
    Returns
    -------
    d_eff : [mm] 
        The effective distance
    '''
    rf_null = trap.rf_null_z
    rf_null_index = trap.rf_null_index
    scale = trap.l0 
    if trap.file_type == 'vtk':
        result_dc = Result.from_vtk(trap.prefix, trap.electrode_file_lookup[dc_electrode])
        coord_x, coord_y, coord_z = result_dc.grid.to_xyz()
        field_grid = result_dc.field
    else: 
        result_dc = trap.electrodes[dc_electrode].potential(derivative=1) 
        coords = result_dc.coords
        coord_x, coord_y, coord_z = np.array(coords['x']) * trap.scale, \
                                    np.array(coords['y']) * trap.scale, \
                                    np.array(coords['z']) * trap.scale
        field_grid = np.array(result_dc) / trap.scale
    trap.x, trap.y, trap.z = coord_x, coord_y, coord_z
    axis = ['x', 'y', 'z'].index(direction)
    a = int(np.floor(field_grid.shape[0]/2))
    b = int(np.floor(field_grid.shape[1]/2))
    if trap.file_type == 'vtk': 
        rf0_field = field_grid[axis] / scale # V/m
    else:
        rf0_field = field_grid[:,:,:,axis] / scale
    
    if force_center:  # force the rf null to be at the center of z
        rf_null_index = len(rf0_field[a,b,:])//2
        rf_null = trap.z[rf_null_index]
    big_plt_font()    
    plt.plot(trap.z[:]*scale*1e6, rf0_field[a, b, :])
    plt.xlabel("Trap height (um)")
    plt.ylabel(f"$E_{direction}$ (V/m)")
    plt.axvline(x = rf_null, color = 'grey', linestyle = '-.',label=f"E field @ rf null height = {rf0_field[a, b, rf_null_index]:.2f} V/m")
    plt.grid()
    plt.legend()
    Ez = rf0_field[a, b, rf_null_index]
    d_eff = np.abs(1/Ez*1e3)
    print("D_eff = {:.2f} mm".format(d_eff))
    
    return d_eff


################# radial frequency ##################
def TrapFrequency(d2Udx2, m):
    '''
    Given the second order gradient of the potential,
    return the trap frequency in Hz
    '''
    return np.sqrt(np.abs(d2Udx2/m)) / 2 / np.pi

def parabola(x, x0, amp):
    return amp * (x - x0)**2

def parabola0(x, amp):
    return amp * x**2

def poly_fit(xi, x0, a2, a3, a4, a5, a6): 
    x = xi - x0 
    return a2*x**2 + a3*x**3 + a4*x**4 + a5*x**5 + a6*x**6
        
def analyze_radial_frequency(trap, fit_range=10e-6, 
                             debug=False):
    '''
    Analyze the radial frequency and plot
    
    Parameters
    ----------
    trap : Trap object
        The trap being analyzed
    fit_range : [m]
        Fit range of the pseudo potential, default 10um.
        Note the unit, in SI!
    
    Returns
    -------
    omega_x : [MHz]
        Frequency in x direction
    omega_y : [MHz]
        Frequency in y direction

    Plots
    -----
    plot 1: x&y (um) line cut of pseudop, fit, and deviation
    plot 2: 2D color map (um) of pseudopotential (eV)
    '''
    c = trap.rf_null_index
    x, y = trap.x, trap.y 

    ### Setup axes ###
    scale = trap.l0
    pp_rf = trap.pp_rf 
    a = int(np.floor(pp_rf.shape[0]/2))
    b = int(np.floor(pp_rf.shape[1]/2))
    x_SI = x*scale
    y_SI = y*scale
    px = pp_rf[:, b, c]*e # in J, SI units
    py = pp_rf[a,:,c]*e # in J
    
    fit_xlim1 = x_SI[np.argmin(px)]-fit_range
    fit_xlim2 = x_SI[np.argmin(px)]+fit_range
    argx1 = np.argmin(np.abs(x_SI - fit_xlim1))
    argx2 = np.argmin(np.abs(x_SI - fit_xlim2))
    if debug: 
        print(argx1,argx2)
    fit_ylim1 = y_SI[np.argmin(py)]-fit_range
    fit_ylim2 = y_SI[np.argmin(py)]+fit_range
    argy1 = np.argmin(np.abs(y_SI - fit_ylim1))
    argy2 = np.argmin(np.abs(y_SI - fit_ylim2))
    if debug: 
        print(argy1,argy2)
    
    ### fit the data ###
    poptx, pcovx = curve_fit(parabola, x_SI[argx1:argx2], px[argx1:argx2], p0 = [x_SI[np.argmin(px)], 1E7], xtol=1e-10)
    x0 = poptx[0]
    x_amp = poptx[1]
    x_coordinates = x_SI #- x0
    
    popty, pcovy = curve_fit(parabola, y_SI[argy1:argy2],py[argy1:argy2], p0 = [y_SI[np.argmin(py)], 1E7], xtol=1e-10)
    y0 = popty[0]
    y_amp = popty[1]
    y_coordinates = y_SI #- y0
    trap.rf_null_x = x0 * 1e6
    trap.rf_null_y = y0 * 1e6  # in um
    
    ### Find trap frequencies ###
    k = 1
    x_range = np.linspace(k*x_coordinates[0],k*x_coordinates[-1], 1000)
    y_range = np.linspace(k*y_coordinates[0],k*y_coordinates[-1], 1000)
    fit_x = parabola(x_range, *poptx) #parabola0(x_range, x_amp)
    fit_y = parabola(y_range, *popty) #parabola0(y_range, y_amp)
    fit_x_match = parabola(x_coordinates, *poptx) #parabola0(x_coordinates, x_amp)
    fit_y_match = parabola(y_coordinates, *popty) #parabola0(y_coordinates, y_amp)
    
    d2Udx2 = 2*x_amp
    trapf_x = TrapFrequency(d2Udx2,m_e)
    print(f'x trap frequency is about {trapf_x/1E6:.3f} MHz')
    d2Udy2 = 2*y_amp
    trapf_y = TrapFrequency(d2Udy2,m_e)
    print(f'y trap frequency is about {trapf_y/1E6:.3f} MHz')
    
    ### Plot pseudo-potential fit ###
    small_plt_font()
    
    fig, (ax1, ax2) = plt.subplots(2,1,sharex = True, 
                                   gridspec_kw={'height_ratios': [3, 1]},
                                   figsize=(3.4,3.5), dpi = 200)
    ax1.plot(x*scale*1e6, pp_rf[:, b, c],label = "Pseudo potential along x")
    ax1.plot(x_range*1E6, fit_x/e, '--', label = f"Fit, $f_x$ = {trapf_x/1E6:.2f} MHz")
    ax1.plot(y*scale*1e6,pp_rf[a,:,c],label = "Pseudo potential along y")
    ax1.plot(y_range*1E6, fit_y/e, '--', label = f"Fit, $f_y$ = {trapf_y/1E6:.2f} MHz")
    ax1.grid()
    ax1.legend()
    ax1.set_ylim(-1E-3, k**2*np.max(px)*1.1/e)
    ax1.set_ylabel('Pseudopotential (eV)', size = 8)
    
    ax2.plot(x_coordinates*1E6,(px-fit_x_match)/e, '-', color = 'C1', label='$x$')
    ax2.plot(y_coordinates*1E6,(py-fit_y_match)/e, '-', color = 'C3', label='$y$')
    ax2.set_xlabel('Distance (um)', size = 8)
    ax2.set_ylabel('$\Delta$ (eV)', size = 8)
    ax2.set_ylim(-0.05, 0.05)
    ax2.set_xlim(k*x_coordinates[0]*1E6, k*x_coordinates[-1]*1E6)
    ax2.axhline(y = 0, color = 'grey', linestyle = '-.')
    ax2.grid()
    ax2.legend()
    fig.align_ylabels()
    fig.tight_layout()
    # plt.savefig('Electron_3layer_pseudo_y_1p6GHz_90V.png', bbox_inches = 'tight')
    
    plt.show()
    
    

    return trapf_x, trapf_y

def poly_fit_anharm_axial(trap, fit_range=10e-6, T=40): 
    """
    Fitting the axial potential along the line through x = 0 and y = 0, using a polynomial
    U(z) up to the 6th order. 
    
    Parameters
    ------
    trap : Trap object
        The trap being analyzed
    fit_range : [m]
        Fit range of the pseudo potential, default 10um.
        Note the unit, in SI!
    T : [K]
        Temperature of axial direction

    Returns
    -------
    omega_z : [MHz]
        Frequency in z direction only assuming RF confinement

    Plots
    -----
    x&y (um) line cut of pseudop, fit, and deviation
    '''
    """
    z = trap.z

    ### Setup axes ###
    scale = trap.l0
    pp_rf = trap.pp_rf 
    a = int(np.floor(pp_rf.shape[0]/2))
    b = int(np.floor(pp_rf.shape[1]/2))
    z_SI = z*scale
    pz = pp_rf[a, b, :]*e # in J, SI units

    fit_zlim1 = z_SI[np.argmin(pz)]-fit_range
    fit_zlim2 = z_SI[np.argmin(pz)]+fit_range
    argz1 = np.argmin(np.abs(z_SI - fit_zlim1))
    argz2 = np.argmin(np.abs(z_SI - fit_zlim2))


    ### fit the data ###
    poptx, pcovx = curve_fit(poly_fit, z_SI[argz1:argz2], pz[argz1:argz2], p0 = [z_SI[np.argmin(pz)], 1E7, 0, 0, 0, 0])
    z0 = poptx[0]
    c2, c3, c4, c5, c6 = poptx[1:]
    z_coordinates = z_SI - z0

    ### Find trap frequencies ###
    k = 1
    z_range = np.linspace(k*z_coordinates[0],k*z_coordinates[-1], 1000)
    fit_z = poly_fit(z_range, 0, *poptx[1:])
    fit_z_match = poly_fit(z_coordinates, 0, *poptx[1:])

    d2Udx2 = 2*c2
    trapf_z = TrapFrequency(d2Udx2,m_e)
    print(f'z trap frequency is about {trapf_z/1E6:.3f} MHz')


    ### Plot pseudo-potential fit ###
    small_plt_font()

    fig, (ax1, ax2) = plt.subplots(2,1,sharex = True, 
                                gridspec_kw={'height_ratios': [3, 1]},
                                figsize=(3.4,3.5), dpi = 200)
    ax1.plot(z*scale*1e6, pp_rf[a, b, :], label = "Pseudopotential along z")
    ax1.plot(z_range*1E6, fit_z/e, '--', label = r"Fit $U(z) = \sum_{i=2}^6 c_iz^i$," + \
                                                f"\n$f_z$ = {trapf_z/1E6:.2f} MHz" + \
                                                f"\n$c_3/c_2$ = {c3/c2*1e-6:.1e} {r'$um^{-1}$'}" + \
                                                f"\n$c_4/c_2$ = {c4/c2*1e-12:.1e} {r'$um^{-2}$'}" + \
                                                f"\n$c_5/c_2$ = {c5/c2*1e-18:.1e} {r'$um^{-3}$'}" + \
                                                f"\n$c_6/c_2$ = {c6/c2*1e-24:.1e} {r'$um^{-4}$'}")
    ax1.grid()
    ax1.legend()
    ax1.set_ylim(-1E-3, k**2*np.max(pz)*1.1/e)
    ax1.set_ylabel('Pseudopotential (eV)', size = 8)

    ax2.plot(z_coordinates*1E6,(pz-fit_z_match)/e, '-', color = 'C1', label='$z$')
    ax2.set_xlabel('Distance (um)', size = 8)
    ax2.set_ylabel('$\Delta$ (eV)', size = 8)
    ax2.set_ylim(-0.05, 0.05)
    # ax2.set_xlim(-0.0007, 0.0007)
    ax2.set_xlim(k*z_coordinates[0]*1E6, k*z_coordinates[-1]*1E6)
    ax2.axhline(y = 0, color = 'grey', linestyle = '-.')
    ax2.grid()
    ax2.legend()
    fig.align_ylabels()
    fig.tight_layout()
    # plt.savefig('Electron_3layer_pseudo_y_1p6GHz_90V.png', bbox_inches = 'tight')

    plt.show()

    A = find_A(trapf_z/1e6, T)
    df_f = get_shift_analytical(A, (c3/c2)*1e-6, (c4/c2)*1e-12, (c5/c2)*1e-18, (c6/c2)*1e-24)
    print(f">>> Relative frequency shift at {T}K is approximately {df_f:.3e}")


    return trapf_z


def plot_pseudo_contour(trap, rf_null_index=None, clim=(-4, 3), xlim=None, ylim=None, title=None): 
    pp_rf = trap.pp_rf 
    if rf_null_index is None: 
        c = trap.rf_null_index 
    else: 
        c = rf_null_index
    ### Plot pseudo-potential contour ###
    big_plt_font()
    #pp_plot = np.around(pp_rf, decimals=2)
    pp_plot = pp_rf
    x, y, z = trap.result_rf1.grid.to_xyz()

    fig, ax = plt.subplots(figsize = (8,6))
    for i in range(len(pp_plot[:, :, c])):
        for j in range(len(pp_plot[i, :, c])):
            if pp_plot[i, j, c] < 10**(clim[0]):
                pp_plot[i, j, c] = 10**(clim[0])

    pp_plot = pp_plot[:,:,c]
    x_plot, y_plot = x, y
    
    
    level = np.logspace(*clim, 1000, base = 10, endpoint = True) # min max level in the plot
    contp = ax.contourf(100 * x_plot[:], 100 * y_plot[:],
                        np.transpose(pp_plot),
                        locator=ticker.LogLocator(),
                        levels = level,
                        cmap = "seismic")
    cbar = fig.colorbar(contp,
                        ticks = [0.001, 0.01,0.1,1,10,100, 1000]
                       )
    ax.set_aspect('equal')
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    cbar.ax.set_ylabel("Pseudopotential (eV)")
    cbar.ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()


################# trap depth ##################
def analyze_trap_depth(trap):
    '''
    Find the trap depth and plot
    
    Parameters
    ----------
    trap: Trap object
    
    Returns
    -------
    trap_depth : [eV] 
        Max of pseudo potential

    Plots
    -----
    plot 1: x&y (um) line cut of pseudop (eV)
    plot 2: 2D color map (um) of pseudopotential (eV)
    '''
    c = trap.rf_null_index
    scale = trap.l0 
    pp_rf = trap.pp_rf 
    a = int(np.floor(pp_rf.shape[0]/2))
    b = int(np.floor(pp_rf.shape[1]/2))

    max_pp = np.log10(np.max(pp_rf))  # why is this log10? 
    max_pp_x = max(pp_rf[:,b,c])
    max_pp_y = max(pp_rf[a,:,c])
    max_pp_x_idx = np.argmax(pp_rf[:,b,c])
    max_pp_y_idx = np.argmax(pp_rf[a,:,c])

    big_plt_font()
    plt.plot(trap.x*scale*1e6, pp_rf[:, b, c],label = "Pseudo potential along x")
    plt.plot(trap.y*scale*1e6,pp_rf[a,:,c],label = "Pseudo potential along y")
    # Using errorbar function just as an illustration, this is NOT really an errorbar
    plt.errorbar(trap.x[max_pp_x_idx]*scale*1e6, max_pp_x/2, yerr=max_pp_x/2, alpha=0.4, capsize=10, capthick=3, label=f'Trap depth along x = {max_pp_x:.2f} eV')
    plt.errorbar(trap.y[max_pp_y_idx]*scale*1e6, max_pp_y/2, yerr=max_pp_y/2, alpha=0.4, capsize=10, capthick=3, label=f'Trap depth along y = {max_pp_y:.2f} eV')
    plt.xlabel("Distance (um)")
    plt.ylabel("Pseudopotential (eV)")
    plt.grid()
    plt.legend()
    plt.show()

    return max_pp
    

################# trap depth, drive frequency, RF amplitude ##################
def analyze_trap_frequencies(trap, beta=0.375, drive_range=(1,5), V0_rf=None, fit_range=10E-6, 
                             debug=False):
    '''
    For a given stability parameter, analyze the RF amplitude and trap depth for 
    a certain drive frequency.
    - trap frequency ~ linear with V
    - trap depth ~ linear with V square
    - a V that can give us same beta = 0.5*0.3/1.6 = 0.5*0.1875
    - for each drive frequency, simulate the trap depth (@ 200 um) and trap 
      frequency, -> the voltage -> the trap depth
    
    Parameters
    ----------
    trap : Trap object
    beta : float
        Stability parameter ~ 1/2 * trap freq/drive freq (default 0.375)
    drive_range : ([GHz], [GHz])
        The range of drive frequencies in a list as (min, max), in units of GHz.
    fit_range : [m]
        Fit range of the pseudo potential, default 10um.
        Note the unit, in SI!

    Returns
    -------
    V_rf : list
        The RF amplitude applied to each electrode (scale with the drive method 
        vector) with different drive frequencies
    Trap_depth : list
        The achieved trap depth with different drive frequencies

    Plots
    -----
    subplot 1: RF amplitude (V) vs, Drive frequency (GHz)
    subplot 2: Trap depth (eV) vs. Drive frequency (GHz)
    '''
    pp_rf_use = trap.rf_field_square
    
    omegas = np.linspace(*drive_range, 20)
    Ω_es = 2*np.pi*1e9*omegas
    V0 = trap.V0_rf if V0_rf is None else V0_rf 
    scale = trap.l0
    
    Vs = []
    trap_depths = []
    for Ω_e in Ω_es:
        pp_rf = PseudoPotential(pp_rf_use,m_e,Ω_e,V0)
        index, rf_null = FindRFNull(pp_rf,trap.z,scale)
        a = int(np.floor(pp_rf.shape[0]/2))
        b = int(np.floor(pp_rf.shape[1]/2))
        c = index
        # fit the trap frequency using V = 10V data
        fit_range = 10E-6 # in SI
        x_SI = trap.x*scale
        y_SI = trap.y*scale
        px = pp_rf[:, b, c]*e # in J, SI units
        py = pp_rf[a,:,c]*e # in J
    
    
        fit_xlim1 = x_SI[np.argmin(px)]-fit_range
        fit_xlim2 = x_SI[np.argmin(px)]+fit_range
        argx1 = np.argmin(np.abs(x_SI - fit_xlim1))
        argx2 = np.argmin(np.abs(x_SI - fit_xlim2))
        if debug:
            print(argx1,argx2)
        fit_ylim1 = y_SI[np.argmin(py)]-fit_range
        fit_ylim2 = y_SI[np.argmin(py)]+fit_range
        argy1 = np.argmin(np.abs(y_SI - fit_ylim1))
        argy2 = np.argmin(np.abs(y_SI - fit_ylim2))
        if debug:
            print(argy1,argy2)
    
        # fit the data
        poptx, pcovx = curve_fit(parabola, x_SI[argx1:argx2], px[argx1:argx2], p0 = [x_SI[np.argmin(px)], 1E7])
        x0 = poptx[0]
        x_amp = poptx[1]
        x_coordinates = x_SI - x0
        popty, pcovy = curve_fit(parabola, y_SI[argy1:argy2],py[argy1:argy2], p0 = [y_SI[np.argmin(py)], 1E7])
        y0 = popty[0]
        y_amp = popty[1]
        y_coordinates = y_SI - y0 
        d2Udx2 = 2*x_amp
        trapf_x = TrapFrequency(d2Udx2,m_e)
        d2Udy2 = 2*y_amp
        trapf_y = TrapFrequency(d2Udy2,m_e)

        ### find the right V by keeping beta the same ### 
        # desired_trapf_x = drive*0.1875 -> desired_voltage = V0 * desired_trapf_x/trapf_x
        # also note that by def. beta = 1/2 * fx/drive
        # translate: desired_fx = 2*drive*beta -> V = V0*(2*drive*beta)/fx 
        V = V0*(2*beta)*Ω_e/(2*np.pi*(trapf_x+trapf_y)/2)  
        # NOTE: this was hand-coded as 0.1875, when beta is 0.375???
        if debug:
            print(V)
        Vs.append(V)
        pp_rf = PseudoPotential(pp_rf_use,m_e,Ω_e,V)
        index, rf_null = FindRFNull(pp_rf,trap.z,scale)
        a = int(np.floor(pp_rf.shape[0]/2))
        b = int(np.floor(pp_rf.shape[1]/2))
        c = index 
        pp_rf_x = pp_rf[:, b, c]
        pp_rf_y = pp_rf[a,:,c]
        trap_depth_x = np.max(pp_rf_x)
        trap_depth_y = np.max(pp_rf_y)
        trap_depths.append((trap_depth_x+trap_depth_y)/2)
        
    fig,ax = plt.subplots(2,1,figsize = (8,6))
    ax1 = ax[0] 
    ax1.plot(omegas, Vs, label = f"$\\beta$ = {beta:.2f}")  # FIXME: this was hand-coded as 0.375
    # ax1.set_xlabel("Drive frequency (GHz)")
    ax1.set_ylabel("RF amplitude (V)")
    ax1.grid()
    ax1.legend()
    
    ax2 = ax[1]
    ax2.plot(omegas, trap_depths, label = f"$\\beta$ = {beta:.2f}")
    ax2.set_xlabel("Drive frequency (GHz)")
    ax2.set_ylabel("Trap depth (eV)")
    ax2.grid()
    ax2.legend()
    plt.show() 

    return Vs, trap_depths



################# multipole control ##################
def config_multipoles(trap, position=[0,0,2], nROI=0.005,
           controlled_electrodes=None, used_multipoles=None,
           order=2, plot_multipoles=True, debug=False):
    '''
    Define Multipole control parameters to create MultipoleControl object.

    Parameters
    ----------
    trap : Trap object
        A trap constructed from BEM simulated data
    nROI : float
        The size of ROI in mm, default is 5 um, meaning 
        the field is expanded in a 2*5 um cube
    position : arr
        Trap location, in mm unit
    controlled_electrodes : list
        The str names of the controlled electrodes
    used_multipoles : list
        The str names of the used multipoles
    order : int
        Expansion order
    plot_multipoles : bool
        Whether to plot the initial coefficients for the multipole expansions
    
    Returns
    -------
    s : MultipoleControl object
        An object that contains all the electrodes and potential data
    '''
    if type(nROI) is float:
        roi = [nROI,nROI,nROI]
    else:
        roi = nROI
    
    if debug:
        print(used_multipoles)
        
    s = MultipoleControl(trap, position, roi, controlled_electrodes, used_multipoles, order)
    v1 = pd.Series(np.zeros(len(controlled_electrodes)), index = controlled_electrodes)
    vs = []
    for ele in s.trap.names:
        v = v1.copy()
        v[ele] = 1
        vs.append(v)
    s = plotting.plot_multipole_vs_expansion_height(position[-1]*1e3, s, roi, vs, plot_multipoles=plot_multipoles)
    return s


################# display anharmonicity terms ##################
def display_harmonics_term(j):
    """ 
    Displays the spherical harmonic term j 

    Parameters
    ----------
    j : str or int 
        The spherical harmonics term name or 
        number
    """
    names = ['C', 'Ey', 'Ez', 'Ex', 'U3',
             'U4', 'U2', 'U5', 'U1']
    if type(j) is str:
        j = names.index(j) + 1 
    x, y, z = sympy.symbols('x, y, z')
    k = 1 
    for order in harmonics: 
        for term in harmonics[order]: 
            if k == j: 
                display(eval(term))
                return 
            k += 1 

def display_harmonics_terms(*js): 
    """ 
    Display multiple terms
    """
    for j in js: 
        print(f'{j}:')
        display_harmonics_term(j)


################# analyze DC and RF anharmonicities ##################
def analyze_anharm_DC(s, electrode_factors, position, nROI=0.005, order=6, Mj_threshold=0.05, **kwargs):
    '''
    Parameters
    ----------
    s : MultipoleControl object
    electrode_factors: arr 
        The electrode voltages used
    
    Returns
    -------
    Mj_DC : arr 
        The fitting coefficients

    Plots
    -----
    - The 3D fitting results 
    - The fitting coefficients
    '''
    roi = [nROI,nROI,nROI]
    s.update_origin_roi(position, roi)
    anharm_settings = dict(
        order = order,
        scale = 1,
        unit = 1,
        r0 = position,
        electrode = 'DC',
        rotate=False,
        electrode_factors = electrode_factors,
        fit_region = 'ROI',
        method = 'lstsq',
        n = 1,
        library = 'manual', 
        Mj_threshold = Mj_threshold, 
        **kwargs
    )
    Mj_DC0, err_DC = fit_anharm(s, **anharm_settings)
    return Mj_DC0 


def analyze_anharm_RF(s, electrode_factors, position, nROI=0.05, order=4, Mj_threshold=0.05, **kwargs):
    '''
    Parameters
    ----------
    s : MultipoleControl object
    electrode_factors: arr 
        The electrode voltages used.
        [V1,V2] for rf1 and rf2
    
    Returns
    -------
    Mj_RF : arr 
        The fitting coefficients

    Plots
    -----
    - The 3D fitting results 
    - The fitting coefficients
    '''
    roi = [nROI,nROI,nROI]
    s.update_origin_roi(position, roi)
    anharm_settings = dict(
        order = order,
        scale = 1,
        unit = 1,
        r0 = position,
        electrode = 'RF',
        rotate=False,
        electrode_factors = electrode_factors,
        fit_region = 'ROI',
        method = 'lstsq',
        n = 1,
        library = 'manual', 
        Mj_threshold = Mj_threshold, 
        **kwargs
    )
    Mj_RF0, err_RF = fit_anharm(s, **anharm_settings)

    return Mj_RF0