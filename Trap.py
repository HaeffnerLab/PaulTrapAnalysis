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
	
	