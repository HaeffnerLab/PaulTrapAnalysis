
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from .utils import construct_derivative


class Electrode:
	"""An electrode of a Paul trap.

	Encapsulates the name, position, dc and rf voltages, 
	and the electrical potential distribution of the electrode.


	Parameters
	-----
	name : str
	dc: float
		DC voltage associated with this electrode. The electrode's electrical potential
		is proportional to the DC voltage.
	rf: float
		RF voltage of this electrode. The pseudopotential contribution is proportional to
		the square of its RF voltage.

	"""
	__slots__ = "name V_dc V_rf".split()

	def __init__(self, name='', V_dc=0., V_rf=0.):
		self.name = name
		self.V_dc = V_dc
		self.V_rf = V_rf

	def potential(self, x, derivative=0,voltage=1.,output=None):
		'''Electrical potential contribution of this Electrode.

		Return the specified derivative of the electrical contribution of this
		electrode assuming all other electrodes in the system are grounded.

		Parameters
		------
		x : array_like, shape (n,3)
			Position to evaluate the electrical potential at. n is the number of the points,
			each of which contains (x,y,z) in Cartesian coordinates
		derivative : int
			Order of the derivative of the potential. derivative = 0 returns the potential itself,
			`derivative=1` the field/force, `derivative=2` the curvature/hessian.
		voltage : float
			Scaling factor of the potential. 
		output: None or array_like, shape (n,2*derivative+1), double
			Array to add the potential contribution to. Needs to be zeroed before.
			If None, an array is created and returned.

		Returns
		-------
		potential : array, shape(n, 2*derivative + 1), double
            Output potential or `output` if given. The first dimension is
            the point index (same as `x`) the second is the derivative
            index. There are only `2*derivative + 1` values as the
            others are lineraly dependent. 


        See Also
        -------
        utils.cartesian_to_spherical_harmonics
        	Convert the tensor from Cartesian coordinates to spherical harmonics

		'''

		pass

	def plot(self, ax, label=None, color=None,**kw):
		'''Visualize and plot this electrode in the supplied axes.

		'''
		pass




class SimulatedElectrode(Electrode):
	'''Electrode based on a precalculated grid of electrical potentials. 
	(From either BEM or FEM simulation methods)
		The scaling is in um.

	Parameters
	---------
	data : list of array_like, shape (n, m, k, l)
		List of potential derivatives. The ith data entry is of
		order (l-1)/2. 'l=1' is the simulated electrical potential. 
		`derivative=1 (l=3)` the field/force, `derivative=2 (l=5)` the curvature/hessian.
		Each entry is shaped as a (n, m, k) grid.
	origin : array_like, shape (3,)
		Position of the (n, m, k) = (0,0,0) voxel
	step : array_like, shape (3,)
		Voxel pitch. Step of the calculated grid


	See Also
	---------
	Electrode
		'name','V_dc','V_rf' attributes/parameters

'''
	__slots__ = "data origin step".split()

	def __init__(self, data = [], origin = (0,0,0), step=(1, 1, 1),**kwargs):
		super(SimulatedElectrode, self).__init__(**kwargs)
		self.data = [np.asanyarray(i, np.double) for i in data]
		self.origin = np.asanyarray(origin, np.double)
		self.step = np.asanyarray(step,np.double)


	@classmethod
	def from_bem():
		'''Create a 'SimulatedElectrode' object from a 'bem.result.Result' instance

		Parameters
		-----

		Returns
		-----
		SimulatedElectrode
		'''
	@classmethod
	def from_ansys():
		'''Load grid potential data from fld file (exported from ansys) and create a 'GridLayout' object
		fld file: potential data file exported from ansys without grid points

		Parameters
		-------

		Returns
		-----
		GridLayout
		'''

	@classmethod
	def from_vtk(cls, elec_name, file, maxderiv=4):
		'''Load grid potential data from vtk StructurePoints and create a 'GridLayout' object

		Parameters
		-------

		Returns
		-----
		GridLayout
		'''
		from tvtk.api import tvtk
		sgr = tvtk.StructuredPointsReader(file_name=file)
		sgr.update()
		sg = sgr.output
		pot = [None, None]
		for i in range(sg.point_data.number_of_arrays):
			name = sg.point_data.get_array_name(i)
			if "_pondpot" in name:
				continue # not harmonic, do not use it
			elif name not in ("potential", "field"):
				continue
			sp = sg.point_data.get_array(i)
			data = sp.to_array()
			step = sg.spacing
			origin = sg.origin
			dimensions = tuple(sg.dimensions)
			dim = sp.number_of_components
			data = data.reshape(dimensions[::-1]+(dim,)).transpose(2, 1, 0, 3)
			pot[int((dim-1)/2)] = data
		obj = cls(name=elec_name, origin=origin, step=step, data=pot)
		obj.generate(maxderiv)
		return obj

	def generate(self, maxderiv=4):
		"""Generate missing derivative orders by successive finite
		differences from the already present derivative orders.
		.. note:: Finite differences amplify noise and discontinuities
			in the original data.
		Parameters
		----------
		maxderiv : int
			Maximum derivative order to precompute if not already
			present.
		"""
		for deriv in range(maxderiv):
			if len(self.data) < deriv+1:
				self.data.append(self.derive(deriv))
			ddata = self.data[deriv]
			assert ddata.ndim == 4, ddata.ndim
			assert ddata.shape[-1] == 2*deriv+1, ddata.shape
			if deriv > 0:
				assert ddata.shape[:-1] == self.data[deriv-1].shape[:-1]

	def derive(self, deriv):
		"""Take finite differences along each axis.
		Parameters
		----------
		deriv : derivative order to generate
		Returns
		-------
		data : array, shape (n, m, k, l)
			New derivative data, l = 2*deriv + 1
		"""
		odata = self.data[deriv-1]
		ddata = np.empty(odata.shape[:-1] + (2*deriv+1,), np.double)
		for i in range(2*deriv+1):
			(e, j), k = construct_derivative(deriv, i)
			# TODO triple work
			grad = np.gradient(odata[..., j], *self.step)[k]
			ddata[..., i] = grad
		return ddata

	def potential(self, x, derivative =0, voltage=1.,output=None):
		x_index = (x - self.origin[None, :])/self.step[None, :]
		if output is None:
			output = np.zeros((x_index.shape[0], 2*derivative+1), np.double)
		dat = self.data[derivative]
		for i in range(2*derivative+1):
			output[:, i] += voltage*map_coordinates(dat[..., i], x_index.T,
				order=1, mode="nearest")
		return output

	




