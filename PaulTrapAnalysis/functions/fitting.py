import numpy as np
from sklearn import linear_model

import warnings
warnings.simplefilter("ignore")

from PaulTrapAnalysis.functions import data
from PaulTrapAnalysis.functions import expansion
from PaulTrapAnalysis.functions import potentials
from PaulTrapAnalysis.functions import plotting
from PaulTrapAnalysis.functions import errors


def spher_harm_expansion(potential_grid, r0, X, Y, Z, order, 
                         method='lstsq', scale=1, rescale=True,
                         library='manual', norm=False, rotate=False):
    '''
    Compute the least-squares solution for the spherical harmonic expansion on potential_grid.
    Overwriting the expansion code in PaulTrap because added in the option for *method* and *scale*.
    
    Parameters
    ----------
    - potential_grid : 3D array of potential values
    - r0 : list [x0, y0, z0] of the expansion point
    - X, Y, Z : axis ranges for the potential grid
    - order : int, order of the expansion
    - method : str, choice of method to fit the coefficient, 
               options are **lstsq** (least square), **ridge*, and **lasso**
    - rescale: whether to rescale the coefficients in the end
    - norm : whether to add manual normalization factors
    
    Returns
    -------
    - Mj : expansion coefficients
    - Yj : harmonic expansions evaluated at the given coordinates
    - scale : the scaling factor used
    '''
    # Convert the 3D DC potential into 1D array.
    # Numerically invert, here the actual expansion takes place and we obtain the expansion coefficients M_{j}.

    nx = len(X)
    ny = len(Y)
    nz = len(Z)
    npts = nx*ny*nz

    W=np.reshape(potential_grid,npts) # 1D array of all potential values
    W=np.array([W]).T # make into column array
    if library == 'sympy':
        print('>>> Using SymPy Harmonic Basis')
        Yj, scale = potentials.sympy_basis_eval(r0, X, Y, Z, order, scale, rotate=rotate)
    elif library == 'manual':
        print('>>> Using manual Harmonic Basis')
        Yj, scale = potentials.manual_harm_basis(r0, X, Y, Z, order, scale, rotate=rotate)
    else:
        print('>>> Using default Harmonic Basis')
        Yj, scale = expansion.spher_harm_basis(r0,X,Y,Z,order, scale=scale, rotate=rotate)
        n_norm_rows = np.min([Yj.shape[1], len(expansion.NormsUptoOrder2)])
        if norm:
            for row_ind in range(n_norm_rows):
                Yj[:,row_ind] = Yj[:,row_ind] / expansion.NormsUptoOrder2[row_ind]
        #Yj, rnorm = spher_harm_basis_v2(r0, X, Y, Z, order)
    
    if method == 'lstsq':
        print(">>> Using Least Square")
        Mj=np.linalg.lstsq(Yj,W,rcond=None) 
        Mj=Mj[0] # array of coefficients
    elif method == 'lasso':
        print(">>> Using Lasso Regression")
        clf = linear_model.Lasso(alpha=0.1, fit_intercept=False)
        clf.fit(Yj, W)
        Mj = clf.coef_
    elif method == 'ridge':
        print(">>> Using Ridge Regression")
        clf = linear_model.Ridge(alpha=0.1, fit_intercept=False)
        clf.fit(Yj, W)
        Mj = clf.coef_
        Mj=Mj[0] # array of coefficients
    
    M0 = np.array(Mj)
    # rescale to original units
    if rescale:
        i = 0
        for n in range(1,order+1):
            for m in range(1,2*n+2):
                i += 1
                Mj[i] = Mj[i]/(scale**n)
    return Mj,Yj,scale, M0


def fit_potential(s, order, scale, r0, electrode='DC', electrode_factors=[],
                  validating_roi=None, fit_region='ROI', method='lstsq', 
                  n=1, library='default', plot_scale=1e3, unit=1, 
                  validate_entire=False, rotate=False,
                  plot_result=True, save_err=True, display_err=True):
    """
    Fit the potential data and plot corresponding plots.
    
    Parameters
    ----------
    s : MultipoleControl object
        An object that contains all the electrodes and potential data
    order : int
        Order of expansion
    scale : float
        To make the data points unitless
    r0 : arr
        Center of the trap
    electrode : str
        Indicating whether the DC or RF electrode data is desired
    validating_roi : arr
        The ROI to validate the fitting on, if None then use entire region
    fit_region : str
        The region used for fitting. Using ROI only if set to *ROI*, 
        otherwise use the entire region.
    method : str
        Method used to fit the coefficients
    n : int
        Plot every other *n* data points
    library : str
        The library of Harmoinc Basis to use. Can use either **default**, 
        i.e. the one used in *PaulTrapAnalysis*, **Sympy**, or **manual**,
        i.e. the manually typed version following Edith's thesis
    plot_scale : float
        The scale used to multiply the plotting axis by such that it's in
        unit of um
    unit : float
        The unit used to scale the coordinates, this should be 1 if scale != 1
        and vice versa. Essentially the idea that we should either scale the 
        coordinates for the data points or scale the coefficients
    plot_result : bool
        Whether to plot the results
    save_err : bool
        Whether to save the error values into a csv
    display_err : bool
        Whether to display the error table
        
    Returns
    -------
    Mj : arr
        The expansion coefficients
    err : (float, float)
        The mean square error evalued on ROI and entire region
    """
    #assert (unit == 1 and scale != 1) or (unit != 1 and scale == 1), 'Duplicated scaling!!'
    #rotate = (True if electrode=='RF' else False) or rotate
    coord_roi, coord0, V_roi, V0 = data.get_potential_data(s, electrode=electrode, electrode_factors=electrode_factors)
    X_roi, Y_roi, Z_roi = coord_roi
    X_roi *= unit
    Y_roi *= unit
    Z_roi *= unit
    r0 *= unit
    if validating_roi is not None:
        s.update_origin_roi(r0, validating_roi)
        coord0, _, V0, _ = data.get_potential_data(s, electrode=electrode, electrode_factors=electrode_factors)
    X0, Y0, Z0 = coord0
    X0 *= unit
    Y0 *= unit
    Z0 *= unit
    r0 = [i*unit for i in r0]
    
    if fit_region == 'ROI':
        print(">>> Fitting using ROI only")
        V = V_roi
        X, Y, Z = X_roi, Y_roi, Z_roi
        m = 1
    else:
        print(">>> Fitting using entire region")
        V = V0
        X, Y, Z = X0, Y0, Z0
        m = n
        
    Mj, Yj, scale, M0 = spher_harm_expansion(V, r0, X, Y, Z, order, 
                                             scale=scale, method=method,
                                             rescale=True, library=library, 
                                             rotate=rotate) # Obtain the harmonic expansions
    
    
    config_name = f'Fitting {electrode} on {fit_region} using {method} and {library} Harmonic Basis to order {order}, evaluated on '
    ## Eavluate and plot the fitting on ROI
    Phi_roi = potentials.generate_potential(r0, X_roi, Y_roi, Z_roi, Mj, order, scale=scale, library=library, rotate=rotate)
    if plot_result:
        plot_coord_roi = data.get_grid(X_roi, Y_roi, Z_roi, r0, rotate=rotate)
        plotting.plot_all_potentials(plot_coord_roi, V_roi, Phi_roi, region='ROI', m=1)
        plotting.plot_projection(plot_coord_roi, V_roi, Phi_roi, r0, Mj, order, scale=scale, plot_scale=plot_scale, n=m, library=library)
    err_roi = errors.make_err_table(V_roi, Phi_roi, config_name+'ROI', display_table=False, save_table=save_err)[1]

    ## Evaluate and plot the fitting on entire region
    if validate_entire:
        Phi0 = potentials.generate_potential(r0, X0, Y0, Z0, Mj, order, scale=scale, library=library, rotate=rotate)
        if plot_result:
            plot_coord0 = data.get_grid(X0, Y0, Z0, r0, rotate=rotate)
            plotting.plot_all_potentials(plot_coord0, V0, Phi0, region='Entire Region', m=n)
            plotting.plot_projection(plot_coord0, V0, Phi0, r0, Mj, order, scale=scale, plot_scale=plot_scale, n=m, library=library)
        err_entire = errors.make_err_table(V0, Phi0, config_name+'Entire Region', display_table=(True and display_err), save_table=save_err)[1]
    else:
        err_entire = None
    
    ## Plot expansion coefficients
    if plot_result:
        plotting.plot_Mj(Mj)
    
    
    return Mj, (err_roi, err_entire)