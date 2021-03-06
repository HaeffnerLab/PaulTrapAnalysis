
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from .utils import construct_derivative, _derivative_names
import xarray as xr


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

	def __init__(self, data = [xr.DataArray()], **kwargs):
		super(SimulatedElectrode, self).__init__(**kwargs)
		self.data = data
		self.step = data[0].attrs['step']

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
		'''Load grid potential data from fld file (exported from ansys) and create a 'SimulatedElectrode' object
		fld file: potential data file exported from ansys without grid points

		Parameters
		-------

		Returns
		-----
		GridLayout
		'''

	@classmethod
	def from_vtk(cls, elec_name, file, maxderiv=4, scale = 1, decimals = 10):
		'''Load grid potential data from vtk StructurePoints and create a 'SimulatedElectrode' object

		Parameters
		elec_name: file name string
		maxderiv: max derivative order to generate
		scale: the ratio between the new unit to the old unit, 
				used to rescale the field and higher order derivative
		decimals: the coordinates accuracy will be truncated to the designated decimal
		-------

		Returns
		-----
		GridLayout
		'''
		from tvtk.api import tvtk
		sgr = tvtk.StructuredPointsReader(file_name=file)
		sgr.update()
		sg = sgr.output
		x = np.linspace(sg.origin[0], sg.origin[0] + sg.spacing[0] * (sg.dimensions[0] - 1), sg.dimensions[0]) / scale
		x = np.around(x, decimals = decimals)
		y = np.linspace(sg.origin[1], sg.origin[1] + sg.spacing[1] * (sg.dimensions[1] - 1), sg.dimensions[1]) / scale
		y = np.around(y, decimals = decimals)
		z = np.linspace(sg.origin[2], sg.origin[2] + sg.spacing[2] * (sg.dimensions[2] - 1), sg.dimensions[2]) / scale
		z = np.around(z, decimals = decimals)
		pot = [None] * maxderiv
		for i in range(sg.point_data.number_of_arrays):
			name = sg.point_data.get_array_name(i)
			if "_pondpot" in name:
				continue # not harmonic, do not use it
			elif name not in ("potential", "field"):
				continue
			sp = sg.point_data.get_array(i)
			data = sp.to_array()
			dimensions = tuple(sg.dimensions)
			dim = sp.number_of_components
			data = data.reshape(dimensions[::-1]+(dim,)).transpose(2, 1, 0, 3)
			order = int((dim-1)/2)
			pot[order] = xr.DataArray(data * scale**order, 
												dims = ('x', 'y', 'z', 'l'), 
												coords = {'x': x, 'y': y, 'z': z, 'l': _derivative_names[order]},
												attrs = dict(derivative_order = order,
															step = sg.spacing / scale,
															origin = sg.origin / scale)
												)
		obj = cls(name=elec_name, data=pot)
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
			if self.data[deriv] is None:
				self.data[deriv] = self.derive(deriv)
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
		data : xarray, shape (n, m, k, l)
			New derivative data, l = 2*deriv + 1
		"""
		odata = self.data[deriv-1]
		ddata = np.empty(odata.shape[:-1] + (2*deriv+1,), np.double)
		for i in range(2*deriv+1):
			(e, j), k = construct_derivative(deriv, i)
			# TODO triple work
			grad = np.gradient(odata[..., j], *self.step)[k]
			ddata[..., i] = grad
		output = xr.DataArray(ddata, 
								dims = ('x', 'y', 'z', 'l'), 
								coords = {'x': odata.x, 'y': odata.y, 'z': odata.z, 'l': _derivative_names[deriv]},
								attrs = dict(derivative_order = int(deriv),
											step = odata.attrs['step'],
											origin = odata.attrs['origin'])
							)
		return output

	def potential(self, x = None, y = None, z = None, derivative =0, voltage=1., method = None):
		dat = self.data[derivative]
		if x is not None:
			dat = dat.sel(x = x, method = method)
		if y is not None:
			dat = dat.sel(y = y, method = method)
		if z is not None:
			dat = dat.sel(z = z, method = method)
		output = voltage * dat
		return output

	




