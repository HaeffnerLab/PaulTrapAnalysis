import warnings, itertools
from contextlib import contextmanager
import logging
import numpy as np
import pandas as pd
import xarray as xr
from collections import OrderedDict
from scipy import optimize, constants as ct
from .Electrode import SimulatedElectrode
from .utils import expand_tensor


try:
	import cvxopt, cvxopt.modeling
except ImportError:
	warnings.warn("cvxopt not found, optimizations will fail", ImportWarning)
	cvxopt = None


logger = logging.getLogger("Electrode")

class Trap:
	'''A collection of Electrodes.


	Parameters
	--------
	electrodes: ordered dictionary of Electrode, eg. {'DC1': DC1, 'DC2': DC2}
	mass: mass of the trapped particle
	charge: charge of the trapped particle
	scale: length scale of the potential, 1 um from bem


	Data structure
	--------
	electrodes: ordered dictionary of trap electrodes
	config: ordered dictionary of trap configuration

	'''
	

	def __init__(self, electrodes = OrderedDict(), mass = 40*ct.atomic_mass, charge = ct.elementary_charge, scale = 1.e-6, Omega = 50.E6,  **kwargs):
		
		self.electrodes = electrodes
		self.config = OrderedDict([('mass', mass),
                      ('charge', charge),
                      ('scale', scale),
                      ('Omega', Omega)])

	def update_electrodes(self, elecs):
		if type(elecs) is not list:
			elecs = [elecs]
		for elec in elecs:
			self.electrodes.update({elec.name: elec})

	@property
	def names(self):
		'''List of names of the electrodes.
		'''
		return [el.name for key, el in self.electrodes.items()]

	@names.setter
	def names(self,names):
		'''names is in dictionary format, example: names = {'DC1': 'DC1.1', 'DC2': 'DC2.1'}
		'''
		for key, el in self.electrodes.items():
			el.name = names[key]
		# for ei, ni in zip(self,names):
		# 	ei.name = ni

	@property
	def V_dcs(self):
		'''Array of dc voltages of the electrodes
		'''
		return pd.Series({el.name: el.V_dc for key, el in self.electrodes.items()})

	@V_dcs.setter
	def V_dcs(self, voltages):
		'''Voltages is in dictionary format, example: voltages = {'DC1': 1, 'DC2': 0}
		'''
		for key, el in self.electrodes.items():
			el.V_dc = voltages[el.name]

	@property
	def V_rfs(self):
		'''Array of rf voltages of the electrodes.
		'''
		return pd.Series({el.name: el.V_rf for key, el in self.electrodes.items()})

	@V_rfs.setter
	def V_rfs(self, voltages):
		'''Voltages is in dictionary format, example: voltages = {'RF1': 100, 'RF2': -100}
		'''
		for key, el in self.electrodes.items():
			el.V_rf = voltages[el.name]

	# def __getitem__(self, name_or_index):
	# 	'''Electrode lookup.


	# 	Returns
	# 	------
	# 	Electrode
	# 		The electrode given by its name or index.
	# 		None if not found by name

	# 	Raises
	# 	------
	# 	IndexError
	# 		If electrode index does not exist
	# 	'''

	# 	try:
	# 		return list.__getitem__(self,name_or_index)
	# 	except TypeError:
	# 		for ei in self:
	# 			if ei.name == name_or_index:
	# 				return ei

	# Electrode = __getitem__

	@contextmanager
	def with_voltages(self, V_dcs=None, V_rfs=None):
		'''Returns a contextmanager with temporary voltage setting.

		This is a convenient way to temporarily change the voltages
		and they are reset to their old values.

		Parameters
		------
		V_dcs : dictionary format
			dc voltages for specific electrodes, or don't include in the dictionary/or None to keep the same
		V_rfs : dictionary format
			dc voltages for specific electrodes, or don't include in the dictionary/or None to keep the same

		Returns
		------
		contextmanager


		Example
		------
		>>> t = Trap()
		>>> with t.with_voltages(V_dcs = {'DC1': 1, 'DC2': 0}, V_rfs = {'RF1': 100, 'RF2': -100}):
				print(t.potential([0,0,1]))
		'''

		try:
			if V_dcs is not None:
				V_dcs, self.V_dcs = self.V_dcs, V_dcs
			if V_rfs is not None:
				V_rfs, self.V_rfs = self.V_rfs, V_rfs
			yield
		finally:
			if V_dcs is not None:
				self.V_dcs = V_dcs
			if V_rfs is not None:
				self.V_rfs = V_rfs

	@contextmanager
	def with_config(self, new_config=None):
		'''Returns a contextmanager with temporary config setting.

		This is a convenient way to temporarily change the configs
		and they are reset to their old values.

		Parameters
		------
		config : dictionary format
		
		Returns
		------
		contextmanager


		Example
		------
		>>> t = Trap()
		>>> with t.with_config({'scale' = 1.e-3}):
				print(t.potential([0,0,1]))
		'''
		try:
			if new_config is not None:
				old_config = self.config.copy()
				for key, value in new_config.items():
					self.config[key] = value
			
			yield
		finally:
			if new_config is not None:
				self.config = old_config

			


	def dc_potential(self, x = None, y = None, z = None, derivative=0, expand=False):
		'''Electrical potential derivative from the DC voltages contribution.

		Parameters
		-------
		x, y, z: array_like, shape (n,1) for each
			Positions to evaluate the potential at.
		derivative: int
			Derivative order
		expand: bool
			If True, return the fully expanded tensor, else return the reduced form.

		Returns
		------
		potential: xarray
			Potential at (x, y, z)
			If expand == False, shape (n, l) and l = 2*derivative+1 is the derivative index
			Else, shape (n, 3, ..., 3) and returns the fully expanded tensorial form


		See Also
		------
		system.electrical_potential
		utils.expand_tensor

		Note
		-----
		Haven't implement the higher order derivative method yet
		'''

		pot = []
		for key, ei in self.electrodes.items():
			vi = getattr(ei, 'V_dc', None)
			if vi:
				pot.append(ei.potential(x, y, z, derivative,voltage=vi))
		pot = sum(pot)
		if expand:
			pot = expand_tensor(pot) # when derivative >= 2, return numpy.array instead of xarray.DataArray
			pass
		return pot

	def rf_potential(self, x = None, y = None, z = None, derivative=0, expand=False):
		'''Electrical potential derivative from the RF voltages contribution.

		Parameters
		-------
		x, y, z: array_like, shape (n,1) for each
			Positions to evaluate the potential at.
		derivative: int
			Derivative order
		expand: bool
			If True, return the fully expanded tensor, else return the reduced form.

		Returns
		------
		See `dc_potential`
		
		See Also
		------
		system.electrical_potential
		utils.expand_tensor

		Note
		-----
		Haven't implement the higher order derivative method yet
		'''
		pot = []
		for key, ei in self.electrodes.items():
			vi = getattr(ei, 'V_rf', None)
			if vi:
				pot.append(ei.potential(x, y, z, derivative,voltage=vi))
		pot = sum(pot)
		if expand:
			pot = expand_tensor(pot)
			pass
		return pot


	def time_dependent_potential(self, x = None, y = None, z = None, derivative=0, t=0., omega=2*np.pi*40.E6,expand=False):
		'''Electric potential at an instant. No pseudopotential averaging.

			V_dc + cos(omega*t)*V_rf

			Parameters
			-------
			x, y, z: array_like, shape (n,1) for each
				Positions to evaluate the potential at.
			derivative: int
				Derivative order
			t: float
				Time instant
			omega: float
				RF frequency
			expand: bool
				Expand to full tensor form if True

			Returns
			-------
			See `dc_potential`

			See Also
			-------
			system.time_potential

			Note
			-----
			Haven't implement the higher order derivative method yet
			Include the frequency of the rf potential as well
		'''
		dc = self.dc_potential(x, y, z, derivative, expand)
		rf = self.rf_potential(x, y, z, derivative, expand)
		return dc + np.cos(omega*t)*rf


	def pseudo_potential(self, x = None, y = None, z = None, derivative = 0):
		'''The pseudopotential/ ponderomotive potential

		Parameters
		-------
		x, y, z: array_like, shape (n,1) for each
			Positions to evaluate the potential at.
		derivative: int <= 2
			Derivative order. Currently only implemented up to 2nd order

		Returns
		------
		potential, array, shape (n, 3, ..., 3)
			Pseudopotential derivative. Fully expanded since this is not generally harmonic
		'''
		q = self.config['charge']
		m = self.config['mass']
		l = self.config['scale']
		o = self.config['Omega']
		rf_scale = np.sqrt(q/m)/(2*l*o)
		p = [self.rf_potential(x, y, z, derivative=i, expand=True) for i in range(1, derivative+2)] # pseudopotential is proportional to field (derivative = 1) squared
		if derivative == 0:
			return rf_scale**2 * xr.dot(p[0], p[0], dims = 'l')

		# below return numpy array
		elif derivative == 1:
			return rf_scale**2 * 2 * np.einsum("ijkl, ijklm->ijkm",p[0],p[1])
		elif derivative == 2:
			return rf_scale**2 * 2 * (np.einsum("ijklm,ijkln -> ijkmn",p[1],p[1]) + np.einsum("ijkl,ijklmn->ijkmn",p[0],p[2]))
		else:
			raise ValueError("only know how to generate pseupotentials up to 2nd order")
		

		return

	def total_potential(self, x = None, y = None, z = None, derivative=0):
		'''Combined electrical and pseudo potential.

		Parameters
		------
		x, y, z: array_like, shape (n,1) for each
			Positions to evaluate the potential at.
		derivative : 
			Order of derivative

		Returns
		------
		potential: array

		'''
		dc = self.dc_potential(x, y, z, derivative, expand=True)
		rf = self.pseudo_potential(x, y, z, derivative)
		return dc + rf

	def individual_potential_contribution(self, x = None, y = None, z = None, derivative=0):
		'''Individual contributions to the electrical potential from all the electrodes.
		Returns an array of the contributions by each electrode in the trap to the potential at points x
		Each electrode is taken to have unit voltage while grounding all other electrodes

		Parameters
		-------
		x, y, z: array_like, shape (n,1) for each
			Positions to evaluate the potential at.
		derivative: int
			Derivative order

		Returns
		-------
		potential_matrix : dictionary, m keys, each value is a shape (n,l) array
			`m` is the electrode index (index into `self`). `n` is the point index,
			`l = 2*derivative + 1` is the derivative index

		See Also
		-------
		system.individual_potential
		'''
		potential_matrix = OrderedDict()
		for key, ei in self.electrodes.items():
			potential_matrix.update({key: ei.potential(x, y, z, derivative)})
		return potential_matrix




