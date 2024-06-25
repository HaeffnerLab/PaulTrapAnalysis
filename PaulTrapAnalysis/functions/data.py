import numpy as np
import pandas as pd
from PaulTrapAnalysis.functions import plotting
from PaulTrapAnalysis.components.Trap import Trap
from PaulTrapAnalysis.components.Electrode import SimulatedElectrode
from PaulTrapAnalysis.components.Multipoles import MultipoleControl

def get_electrodes(prefix, nROI=0.005, position=[0,0,0.287], order=2, plot_multipoles=True):
    """
    Get the electrode data from .vtk files.
    
    Parameters
    ----------
    prefix : str
        The prefix of the .vtk files
    nROI : float
        The size of ROI in mm, default is 5 um, meaning 
        the field is expanded in a 2*5 um cube
    position : arr
        Trap location, in mm unit
    order : int
        Expansion order
    plot_multipoles : bool
        Whether to plot the initial coefficients for the multipole expansions
    
    Returns
    -------
    s : MultipoleControl object
        An object that contains all the electrodes and potential data
    """
    # path = './3D_design_v7_newformat.pkl'
    # f = open(path,'rb')
    # trap = pickle.load(f)
    #prefix = f"{esim.data_dir}/potential/Electron3dTrap_200um_v6_flipped"
    electrode_list = [
                              'DC0', 'DC1', 'DC2', 'DC3', 'DC4', 'DC5', 'DC6', 'DC7', 'DC8',
                             'RF1', 'RF2'
                            ]
    etrap_v4 = Trap()
    for elec in electrode_list:
        #print(elec)
        etrap_v4.update_electrodes(SimulatedElectrode.from_vtk(elec_name = elec, scale = 10, 
                                                               file = prefix + '_' + elec + '.vtk'))
    #scale = new/old = 1 mm / 100 um, 100um: a0 in bem simulation, 1mm: unit in multipole expansion
    
    #position = [0,0,0.287] # trap location, in mm unit
    #nROI = 0.005 # expand the field in a 2*5 um cube
    #nROI = 0.01
    if type(nROI) is float:
        roi = [nROI,nROI,nROI]
    else:
        roi = nROI
    #order = 2 # expansion order
    controlled_electrodes = [
                              'DC0', 'DC1', 'DC2', 'DC3', 'DC4', 'DC5', 'DC6', 'DC7', 'DC8'
        #'DC0',
                             # 'RF1', 'RF2'
                            ] # changed DC0 to be DC9 and changed the order
    used_order1multipoles = ['Ey', 'Ez', 'Ex']
    used_order2multipoles = ['U3', 'U4', 'U2', 'U5', 'U1']
    used_multipoles = used_order1multipoles + used_order2multipoles
    #print(used_multipoles)
    
    s = MultipoleControl(etrap_v4, position, roi, controlled_electrodes, used_multipoles, order)
    v1 = pd.Series(np.zeros(len(controlled_electrodes)), index = controlled_electrodes)
    vs = []
    for ele in s.trap.names:
        v = v1.copy()
        v[ele] = 1
        vs.append(v)
    s = plotting.plot_multipole_vs_expansion_height(position[-1]*1e3, s, roi, vs, plot_multipoles=plot_multipoles)
    return s

def rotate_45(x, y):
    norm_fact = 1/np.sqrt(2)
    return norm_fact * (x-y), norm_fact * (x+y)

def get_potential_data(s, electrode='DC', electrode_factors=[]):
    """
    Load the potential data from the electrodes.
    
    Parameters
    ----------
    s : MultipoleControl object 
        A MultipoleControl that contains all the updated electrode data and potential
    electrode : str
        Indicating whether the DC or RF electrode data is desired

    Returns
    -------
    coord_roi : arr
        (X_roi, Y_roi, Z_roi) coordinate bounds for the region of interests
    coord : arr
        (X, Y, Z) coordinate bounds for the entire region
    V_roi : arr
        The total potential values for ROI
    V : arr
        The total potential values for the entire region
    """
        
    if electrode == 'DC':
        V_roi = np.zeros(np.shape(s.electrode_potential_roi['DC1']))
        V = np.zeros(np.shape(s.electrode_potential['DC1']))
        if len(electrode_factors) == 0:
            DC_factors = [-0.364604, -0.550222, -0.019522, -0.758919, 0.078677,
                          -0.548974, -0.034351, -0.759486, 0.082480]
        else:
            DC_factors = electrode_factors
        
        for ele in range(9):
            V_roi += s.electrode_potential_roi[f'DC{ele}'] * DC_factors[ele] # V_
        
        for ele in range(9):
            V += s.electrode_potential[f'DC{ele}'] * DC_factors[ele]
        
            
        X_roi = V_roi.coords['x'].values
        Y_roi = V_roi.coords['y'].values
        Z_roi = V_roi.coords['z'].values
        
        X = V.coords['x'].values
        Y = V.coords['y'].values
        Z = V.coords['z'].values
        
    else:
        if len(electrode_factors) == 0:
            RF_factors = [0, 35]
        else:
            RF_factors = electrode_factors
            
        V_roi = np.zeros(np.shape(s.electrode_potential_roi['RF2']))
        V = np.zeros(np.shape(s.electrode_potential['RF2']))
        for ele in range(1,3):
            V_roi += s.electrode_potential_roi[f'RF{ele}'] * RF_factors[ele-1]
            V += s.electrode_potential[f'RF{ele}'] * RF_factors[ele-1]
        #V_roi = s.electrode_potential_roi['RF2'] * RF_factor # only RF2 since RF1 is grounded
        X_roi = V_roi.coords['x'].values
        Y_roi = V_roi.coords['y'].values
        Z_roi = V_roi.coords['z'].values
        
        #V = s.electrode_potential['RF2'] * RF_factor
        X = V.coords['x'].values
        Y = V.coords['y'].values
        Z = V.coords['z'].values
    
    return (X_roi, Y_roi, Z_roi), (X, Y, Z), np.array(V_roi), np.array(V)

def rescale_Mj(Mj, scale, order):
    Mj = np.array(Mj) # Creating a copy of Mj to avoid modification
    #unnormalize
    i=0
    for n in range(1,order+1):
        for m in range(1,2*n+2):
            i += 1
            Mj[i] = Mj[i]*(scale**n)
    return Mj

def get_grid(X, Y, Z, r0, rotate=False):
    x0,y0,z0 = r0
    nx = len(X)
    ny = len(Y)
    nz = len(Z)
    npts = nx*ny*nz

    y, x, z = np.meshgrid(Y-y0,X-x0,Z-z0)
    x, y, z = np.reshape(x,npts), np.reshape(y,npts), np.reshape(z,npts)
    if rotate:
        x, y = rotate_45(x, y)
    return x, y, z

def get_square(xlow=-200e-6, xhigh=200e-6, ylow=-200e-6, yhigh=200e-6, npoints=50):
    X = np.linspace(xlow, xhigh, npoints)
    Y = np.linspace(ylow, yhigh, npoints)
    x, y = np.meshgrid(X, Y)
    x = x.flatten()
    y = y.flatten()
    z = np.zeros(np.shape(x))
    return x, y, z