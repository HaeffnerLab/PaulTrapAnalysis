"""
multipoles.py

Container class for post-processing
resuls from BEM simulations.

"""
import numpy as np
import pandas as pd
import xarray as xr
from collections import OrderedDict
from .expansion import spher_harm_expansion, spher_harm_cmp, nullspace, NamesUptoOrder2, PrintNamesUptoOrder2, NormsUptoOrder2
from .optimsaddle import exact_saddle, find_saddle
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
import cvxpy as cvx
from .Trap import Trap

class MultipoleControl:

    trap = None

    # Below are all related to origin, roi

    origin = None
    roi = None # after initialization should be an array of 3 numbers for length
    
    electrode_potential = OrderedDict()
    electrode_potential_roi = OrderedDict() # update this only preserve function

    # Below are all related to expansion order

    multipole_names = NamesUptoOrder2
    multipole_print_names = pd.Series(PrintNamesUptoOrder2, index = multipole_names)
    normalization_factors = NormsUptoOrder2

    order = None

    multipole_expansion = pd.DataFrame()
    electrode_potential_regenerated = OrderedDict() # update this only preserve function

    # Below are related to controlled_electrodes, used_multipoles

    controlled_elecs = None
    used_multipoles = None

    expansion_matrix = pd.DataFrame()
    pinv_matrix = pd.DataFrame()

    def __init__(self, trap, origin, roi, controlled_electrodes, used_multipoles, order = 2):
        '''
        trap is an object created by the imported class Trap
        '''

        self.trap = trap
        self.electrode_potential = self.trap.individual_potential_contribution()

        # First read all basic values
        self.controlled_elecs = controlled_electrodes
        self.used_multipoles = used_multipoles
        self.order = order

        # Below setting up origin and region of interest and so on
        self.update_origin_roi(origin, roi)

        return

    def update_origin_roi(self, origin, roi):
        '''
        This function updates origin and roi, and also update everything related to them.
        Can be called to update orgin and roi from object.
        '''
        self.origin = np.array(origin)
        self.roi = roi

        x_slice = slice(self.origin[0] - self.roi[0], self.origin[0] + self.roi[0])
        y_slice = slice(self.origin[1] - self.roi[1], self.origin[1] + self.roi[1])
        z_slice = slice(self.origin[2] - self.roi[2], self.origin[2] + self.roi[2])

        self.electrode_potential_roi = self.trap.individual_potential_contribution(x = x_slice, y = y_slice, z = z_slice)

        self.update_expansion_order(self.order)
        return


    def update_expansion_order(self, order):
        '''
        This function updates expansion order, and also update everything related to them.
        Can be called from object to update all related staff.
        '''
        self.order = order
        self.multipole_expansion, self.electrode_potential_regenerated = self.expand_potentials_spherHarm(self.electrode_potential_roi, self.origin, order, self.multipole_names)
        self.update_control(self.controlled_elecs, self.used_multipoles)
        return

    def update_control(self, controlled_electrodes, used_multipoles):
        '''
        This function updates controlled electrodes and used multipoles, 
        and also the control matrix retrieved from min norm problem.
        '''
        self.controlled_elecs = controlled_electrodes
        self.used_multipoles = used_multipoles
        trim_elecs = self.multipole_expansion[self.controlled_elecs]
        self.expansion_matrix = trim_elecs.loc[self.used_multipoles]

        
        self.pinv_matrix = pd.DataFrame(np.linalg.pinv(self.expansion_matrix), self.expansion_matrix.columns, self.expansion_matrix.index)
        return self.expansion_matrix, self.pinv_matrix

    @staticmethod
    def expand_potentials_spherHarm(potential_roi, r0, order, multipole_names):
        '''
        This function expands potentials, and drop shperical harmonics normalization factors.
        It renames multipoles
        up to 2nd order: multipole_names = ['C','Ey','Ez', 'Ex', 'U3=xy', 'U4=yz', r'U2=z^2-(x^2+y^2)/2',
                                            'U5=zx', r'U1=x^2-y^2']
                         normalization_factors = [np.sqrt(1/4/np.pi), np.sqrt(3/4/np.pi), np.sqrt(3/4/np.pi), 
                                                  np.sqrt(3/4/np.pi), np.sqrt(15/4/np.pi), np.sqrt(15/4/np.pi), 
                                                  np.sqrt(20/16/np.pi), np.sqrt(15/4/np.pi), np.sqrt(15/16/np.pi)]
        '''

        N = (order + 1)**2
        assert N >= len(multipole_names)
        multipoles = pd.DataFrame()
        multipoles_index_names = list(np.arange(0, N, 1))
        multipoles_index_names[:len(multipole_names)] = multipole_names
        potential_regenerated = OrderedDict()
        for ele in potential_roi:
            X_roi = potential_roi[ele].x
            Y_roi = potential_roi[ele].y
            Z_roi = potential_roi[ele].z
            Mj,Yj,scale = spher_harm_expansion(potential_roi[ele].values, r0, X_roi, Y_roi, Z_roi, order)
            multipoles[ele] = pd.Series(Mj[0:N].T[0], index = multipoles_index_names)

            Vregen = spher_harm_cmp(Mj,Yj,scale,order)
            potential_regenerated[ele] = xr.zeros_like(potential_roi[ele])
            potential_regenerated[ele].loc[:] = Vregen.reshape([len(X_roi), len(Y_roi), len(Z_roi), 1])
        return multipoles, potential_regenerated

    def setVoltages(self, voltages):
        '''
        This function takes volteges you apply and returns multipole coefficients you get.
        input i.e. vs = {'DC1':1, 'DC2':2}
        '''
        M = (self.order + 1)**2
        coeffs = pd.Series(np.zeros(M), index = self.multipole_expansion.index)
        for key in voltages.keys():
            coeffs += self.multipole_expansion[key] * voltages[key]
        return coeffs

    def setMultipoles(self, coeffs):
        '''
        This function takes a set of desired multipole coefficients and returns the voltages needed to acheive that.
        Method: min norm
        input i.e. coeffs = {'Ex: 1', 'U2': 20}
        '''
        N = len(self.controlled_elecs)
        voltages = pd.Series(np.zeros(N), index = self.controlled_elecs)
        for key in coeffs.keys():
            voltages += self.pinv_matrix[key] * coeffs[key]
        return voltages

    def potentialControl_all(self, vs):
        '''
        This function takes voltages and returns the potential you get over the full space.
        input i.e. vs = {'DC1':1, 'DC2':2}
        '''
        for i, key in enumerate(vs.keys()):
            if i == 0:
                output = self.electrode_potential[key] * vs[key]
            else:
                output += self.electrode_potential[key] * vs[key]

        return output

    def potentialControl_roi(self, vs):
        '''
        This function takes voltages and returns the potential you get over the roi.
        i.e. vs = {'DC1':1, 'DC2':2}
        '''
        output_roi = []
        for key in vs.keys():
            output_roi.append(self.electrode_potential_roi[key] * vs[key])

        return sum(output_roi)

    def potentialControl_regen(self, vs):
        '''
        This function takes voltages and returns the potential regenerated from multipole coefficients over the roi.
        i.e. vs = {'DC1':1, 'DC2':2}
        '''
        output_roi = []
        for key in vs.keys():
            output_roi.append(self.electrode_potential_regenerated[key] * vs[key])

        return sum(output_roi)

    @staticmethod
    def min_linf(y, X):
        '''
        This function computes a constraint probelm: min(max(w)) s.t. X @ w = y.
        It returns w^{hat} that satisfy the above problem.
        '''
        X_mat = np.asarray(X)
        y_mat = np.asarray(y)
        w = cvx.Variable(X_mat.shape[1]) #b is dim x  
        objective = cvx.Minimize(cvx.norm(w,'inf')) #L_1 norm objective function
        constraints = [X_mat @ w == y_mat] #y is dim a and M is dim a by b
        prob = cvx.Problem(objective,constraints)
        result = prob.solve(verbose=False)
        w_hat = pd.Series(np.array(w.value), index = X.columns)
        return w_hat






