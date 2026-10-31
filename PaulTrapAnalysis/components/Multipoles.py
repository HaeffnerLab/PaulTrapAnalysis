"""
multipoles.py

Container class for post-processing
resuls from BEM simulations.

"""
import numpy as np
import pandas as pd
import xarray as xr
from collections import OrderedDict
from PaulTrapAnalysis.functions.expansion import spher_harm_expansion, spher_harm_cmp, nullspace, NamesUptoOrder2, PrintNamesUptoOrder2, NormsUptoOrder2
from PaulTrapAnalysis.functions.optimsaddle import exact_saddle, find_saddle
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
import cvxpy as cvx
from PaulTrapAnalysis.components.Trap import Trap

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
    shorted_groups = OrderedDict() # {representative: [representative, member1, ...]}

    expansion_matrix = pd.DataFrame()
    pinv_matrix = pd.DataFrame()

    def __init__(self, trap, origin, roi, controlled_electrodes, used_multipoles, order = 2, shorted_groups = None):
        '''
        trap is an object created by the imported class Trap

        shorted_groups : dict, optional
            Electrodes that are shorted to each other (but not to ground), given as
            {representative: [other members]}, e.g. {'DC1': ['DC13', 'DC15']}.
            Each group is treated as ONE electrode whose basis potential is the sum of
            its members' potentials, so the solver accounts for the field of every
            shorted electrode. Only the representative may be in controlled_electrodes;
            its voltage is applied to all members in setVoltages, potentialControl_*
            and write_txt.
        '''

        self.trap = trap
        self.electrode_potential = self.trap.individual_potential_contribution()

        # First read all basic values
        self.controlled_elecs = controlled_electrodes
        self.used_multipoles = used_multipoles
        self.order = order
        self.set_shorted_groups(shorted_groups, update = False)

        # Below setting up origin and region of interest and so on
        self.update_origin_roi(origin, roi)

        

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
        


    def update_expansion_order(self, order):
        '''
        This function updates expansion order, and also update everything related to them.
        Can be called from object to update all related stuff.
        '''
        self.order = order
        self.multipole_expansion, self.electrode_potential_regenerated = self.expand_potentials_spherHarm(self.electrode_potential_roi, self.origin, order, self.multipole_names)
        self.update_control(self.controlled_elecs, self.used_multipoles)
        

    def update_control(self, controlled_electrodes, used_multipoles):
        '''
        This function updates controlled electrodes and used multipoles, 
        and also the control matrix retrieved from min norm problem.
        '''
        self.controlled_elecs = controlled_electrodes
        self.used_multipoles = used_multipoles
        self._check_controlled()
        # each controlled electrode contributes the summed expansion of every electrode shorted to it
        trim_elecs = pd.DataFrame({e: self.multipole_expansion[self.group_members(e)].sum(axis = 1)
                                   for e in self.controlled_elecs})
        self.expansion_matrix = trim_elecs.loc[self.used_multipoles]

        
        self.pinv_matrix = pd.DataFrame(np.linalg.pinv(self.expansion_matrix), self.expansion_matrix.columns, self.expansion_matrix.index)
        return self.expansion_matrix, self.pinv_matrix

    def set_shorted_groups(self, shorted_groups, update = True):
        '''
        Define electrodes that are shorted together (not to ground). See __init__ for the format.
        Call with update = True (default) on an existing object to recompute the control matrix.
        '''
        groups = OrderedDict()
        owner = {}
        for rep, members in (shorted_groups or {}).items():
            if isinstance(members, str):
                members = [members]
            full = [rep] + [m for m in members if m != rep]
            for m in full:
                if m in owner:
                    raise ValueError(f"Electrode '{m}' is in more than one shorted group ('{owner[m]}' and '{rep}').")
                owner[m] = rep
            groups[rep] = full
        unknown = [m for m in owner if m not in self.trap.electrodes]
        if unknown:
            raise ValueError(f"Shorted-group electrodes {unknown} are not in the trap.")
        self.shorted_groups = groups
        self._group_of = owner
        if update:
            self.update_control(self.controlled_elecs, self.used_multipoles)

    def group_members(self, name):
        '''All electrodes that are shorted to `name` and share its voltage (just [name] if not in a group).'''
        rep = self._group_of.get(name, name)
        return self.shorted_groups.get(rep, [name])

    def _check_controlled(self):
        members = [e for e in self.controlled_elecs if self._group_of.get(e, e) != e]
        if members:
            raise ValueError(f"{members} are shorted to another electrode; put only the group representative "
                             f"({[self._group_of[e] for e in members]}) in controlled_electrodes.")

    def expand_group_voltages(self, voltages):
        '''
        Returns {electrode: voltage} where the voltage of each group representative is copied to
        every electrode shorted to it.
        '''
        out = OrderedDict()
        for key in voltages.keys():
            members = self.shorted_groups.get(key, [key])
            for m in members:
                if m != key and m in voltages.keys():
                    raise ValueError(f"'{m}' is shorted to '{key}'; give a voltage for '{key}' only.")
                out[m] = voltages[key]
        return out

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

    def setVoltages(self, voltages, expand_groups = True):
        '''
        This function takes volteges you apply and returns multipole coefficients you get.
        input i.e. vs = {'DC1':1, 'DC2':2}
        If expand_groups, a group representative's voltage is also applied to the electrodes shorted to it.
        Use expand_groups = False to look at individual electrodes on their own.
        '''
        if expand_groups:
            voltages = self.expand_group_voltages(voltages)
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
        vs = self.expand_group_voltages(vs)
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
        vs = self.expand_group_voltages(vs)
        output_roi = []
        for key in vs.keys():
            output_roi.append(self.electrode_potential_roi[key] * vs[key])

        return sum(output_roi)

    def potentialControl_regen(self, vs):
        '''
        This function takes voltages and returns the potential regenerated from multipole coefficients over the roi.
        i.e. vs = {'DC1':1, 'DC2':2}
        '''
        vs = self.expand_group_voltages(vs)
        output_roi = []
        for key in vs.keys():
            output_roi.append(self.electrode_potential_regenerated[key] * vs[key])

        return sum(output_roi)
    
    def write_txt(self, filename, strs, excl = None):
        '''
        Writes the pseudo-inverse column of every electrode in strs (in that order) for each multipole.
        Electrodes in a shorted group get their representative's values automatically.

        excl : dict, optional. {electrode: "gnd"} writes 0 for a grounded electrode.
            Tying one electrode to another ({'DC6': 'DC2'}) is only accepted when both are in the
            same shorted group. Otherwise the solution would have treated the tied electrode as
            grounded while the file drives it, so a ValueError is raised: use shorted_groups instead.
        '''
        excl = excl or {}
        source = {}
        for key in strs:
            rep = self._group_of.get(key, key)
            target = excl.get(key)
            if target == "gnd":
                if rep in self.controlled_elecs:
                    raise ValueError(f"'{key}' is set to gnd in excl but is controlled"
                                     + (f" through its group '{rep}'" if rep != key else "") + ".")
                source[key] = None
                continue
            if target is not None and self._group_of.get(target, target) != rep:
                raise ValueError(f"excl ties '{key}' to '{target}', but they are not in the same shorted group, "
                                 f"so the solution does not account for '{key}' at that voltage. "
                                 f"Add them to shorted_groups instead.")
            if rep not in self.controlled_elecs:
                raise ValueError(f"'{key}' is neither controlled (directly or through a shorted group) "
                                 f"nor marked 'gnd' in excl.")
            source[key] = rep

        outarray = []
        allmpl = ['Ex', 'Ey', 'Ez','U1', 'U2', 'U3', 'U4','U5']
        for multipole in allmpl:
            for key in strs:
                if multipole in self.pinv_matrix and source[key] is not None:
                    outarray.append(self.pinv_matrix[multipole][source[key]])
                else:
                    outarray.append(0.)
        print(np.shape(outarray))
        pd.DataFrame(outarray).to_csv(filename+'.txt', header=None, index=None, float_format='%.15f')

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






