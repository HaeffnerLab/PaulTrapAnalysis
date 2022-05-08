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
	def dcs(self):
		'''Array of dc voltages of the electrodes
		'''
		return np.array([el.dc for el in self])

	@dcs.setter
	def dcs(self, voltages):
		for ei, vi in zip(self,voltages):
			ei.dc = vi

	@property
	def rfs(self):
		'''Array of rf voltages of the electrodes.
		'''
		return np.array([el.rf for el in self])

	@rfs.setter
	def rfs(self, voltages):
		for ei, vi in zip(self, voltages):
			ei.rf = vi

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
	def with_voltages(self, dcs=None, rfs=None):
		'''Returns a contextmanager with temporary voltage setting.

		This is a convenient way to temporarily change the voltages
		and they are reset to their old values.

		Parameters
		------
		dcs : array_like
			dc voltages for all electrodes, or None to keep the same
		rfs : array_like
			dc voltages for all electrodes, or None to keep the same

		Returns
		------
		contextmanager


		Example
		------
		>>> t = Trap()
		>>> with t.with_voltages(dcs = 0.5*s.dcs, rfs = [0,1]):
				print(t.potential([0,0,1]))
		'''

		try:
			if dcs is not None:
				dcs, self.dcs = self.dcs, dcs
			if rfs is not None:
				rfs, self.rfs = self.rfs, rfs
			yield
		finally:
			if dcs is not None:
				self.dcs = dcs
			if rfs is not None:
				self.rfs = rfs

	
	
	