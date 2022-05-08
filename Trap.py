import warnings, itertools
from contextlib import contextmanager
import logging
import numpy as np
from scipy import optimize, constants as ct
from .Electrode import SimulatedElectrode


try:
	import cvxopt, cvxopt.modeling
except ImportError:
	warnings.warn("cvxopt not found, optimizations will fail", ImportWarning)
	cvxopt = None


logger = logging.getLogger("Electrode")

class Trap(list):
	'''A collection of Electrodes.


	Parameters
	--------
	electrodes: list of Electrode
	'''

	def __init__(self, electrodes = [],**kwargs):
		super(Trap, self).__init__(**kwargs)
		self.extend(electrodes)

	@property
	def names(self):
		'''List of names of the electrodes.

		'''
		return [el.name for el in self]

	@names.setter
	def names(self,names):
		for ei, ni in zip(self,names):
			ei.name = ni

	@property
	def V_dcs(self):
		'''Array of dc voltages of the electrodes
		'''
		return np.array([el.V_dc for el in self])

	@V_dcs.setter
	def V_dcs(self, voltages):
		for ei, vi in zip(self,voltages):
			ei.V_dc = vi

	@property
	def V_rfs(self):
		'''Array of rf voltages of the electrodes.
		'''
		return np.array([el.V_rf for el in self])

	@V_rfs.setter
	def V_rfs(self, voltages):
		for ei, vi in zip(self, voltages):
			ei.V_rf = vi

	def __getitem__(self, name_or_index):
		'''Electrode lookup.


		Returns
		------
		Electrode
			The electrode given by its name or index.
			None if not found by name

		Raises
		------
		IndexError
			If electrode index does not exist
		'''

		try:
			return list.__getitem__(self,name_or_index)
		except TypeError:
			for ei in self:
				if ei.name == name_or_index:
					return ei

	Electrode = __getitem__

	@contextmanager
	def with_voltages(self, V_dcs=None, V_rfs=None):
		'''Returns a contextmanager with temporary voltage setting.

		This is a convenient way to temporarily change the voltages
		and they are reset to their old values.

		Parameters
		------
		V_dcs : array_like
			dc voltages for all electrodes, or None to keep the same
		V_rfs : array_like
			dc voltages for all electrodes, or None to keep the same

		Returns
		------
		contextmanager


		Example
		------
		>>> t = Trap()
		>>> with t.with_voltages(V_dcs = 0.5*s.V_dcs, V_rfs = [0,1]):
				print(t.potential([0,0,1]))
		'''

		try:
			if V_dcs is not None:
				V_dcs, self.V_dcs = self.V_dcs, V_dcs
			if rfs is not None:
				V_rfs, self.V_rfs = self.V_rfs, V_rfs
			yield
		finally:
			if V_dcs is not None:
				self.V_dcs = V_dcs
			if V_rfs is not None:
				self.V_rfs = V_rfs


	def dc_potential(self, x, derivative=0,expand=False):
		'''Electrical potential derivative from the DC voltages contribution.

		Parameters
		-------
		x: array_like, shape (n,3)
			Positions to evaluate the potential at.
		derivative: int
			Derivative order
		expand: bool
			If True, return the fully expanded tensor, else return the reduced form.

		Returns
		------
		potential: array
			Potential at 'x'
			If expand == False, shape (n, l) and l = 2*derivative+1 is the derivative index
			Else, shape (n, 3, ..., 3) and returns the fully expanded tensorial form


		See Also
		------
		system.electrical_potential

		Note
		-----
		Haven't implement the higher order derivative method yet
		'''

		x = np.asanyarray(x,dtype=np.double).reshape(-1,3)
		pot = np.zeros((x.shape[0],2*derivative+1),np.double)
		for ei in self:
			vi = getattr(ei,V_dc,None)
			if vi:
				ei.potential(x,derivative,voltage=vi,output=pot)
			if expand:
				# pot = expand_tensor(pot)
				pass
		return pot

	def rf_potential(self, x, derivative=0,expand=False):
		'''Electrical potential derivative from the RF voltages contribution.

		Parameters
		-------
		x: array_like, shape (n,3)
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

		Note
		-----
		Haven't implement the higher order derivative method yet
		'''
		x = np.asanyarray(x,dtype=np.double).reshape(-1,3)
		pot = np.zeros((x.shape[0],2*derivative+1),np.double)
		for ei in self:
			vi = getattr(ei,V_rf,None)
			if vi:
				ei.potential(x,derivative,voltage=vi,output=pot)
			if expand:
				# pot = expand_tensor(pot)
				pass
		return pot


	def time_dependent_potential(self, x, derivative=0, t=0., omega=2*np.pi*40.E6,expand=False):
		'''Electric potential at an instant. No pseudopotential averaging.

			V_dc + cos(omega*t)*V_rf

			Parameters
			-------
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
		dc = self.dc_potential(x, derivative, expand)
		rf = self.rf_potential(x, derivative, expand)
		return dc + np.cos(omega*t)*rf

	def pseudo_potential(self, x, derivative = 0):
		'''The pseudopotential/ ponderomotive potential

		Parameters
		'''
		return

	def potential(self, x, derivative=0):
		'''Combined electrical and pseudo potential.

		Parameters
		------
		x: array, shape (n,3)
			Points to evaluate at.
		derivative : 
			Order of derivative

		Returns
		------
		potential: array

		'''
		dc = self.dc_potential(x,derivative,expand=True)
		rf = self.pseudo_potential(x, derivative)
		return dc + rf



