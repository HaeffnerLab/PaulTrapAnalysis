
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from .utils import construct_derivative, _derivative_names,demesh
import xarray as xr
import pyvista as pv
import pandas as pd


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
	def from_fld(cls, elec_name, file, stepSize, min_grid, max_grid, mirror = False, maxderiv=4, scale = 1, decimals = 10, skiprows=2,sep=' ',names=['potential'],perm=[2,0,1]):
		'''Load grid potential data from fld file (exported from ansys) and create a 'SimulatedElectrode' object
		fld file: potential data file exported from ansys without grid points

		Parameters
		-------
		stepSize = [0.005,0.005,0.005] #step size in mm
		min_grid = [-0.25,-0.25,-0.25] #x,y,z_min of the grid, in mm
		max_grid = [0.25,0.25,0.25] #x,y,z_max of the grid, in mm
		mirror: boolean, mirror around x axis for three layer trap br from bl

		Returns
		-----
		GridLayout
		'''
		# x,y,z demesh, numpy function
		
		ug = pd.read_csv(file, skiprows=skiprows, sep=sep,names=names)
		nXYZ = [int((max_grid[i]-min_grid[i])/stepSize[i]+1) for i in range(3)]
		roi = np.array([np.array([0,nXYZ[i]-1])*stepSize[i]+min_grid[i] for i in range(3)])
		gridXYZ = [np.linspace(roi[l,0],roi[l,1],nXYZ[l]) for l in range(3)]
		x = gridXYZ[0]
		y = gridXYZ[1]
		z = gridXYZ[2]
		#label voltage data: in Ansys coordinate, change z first, then y, at last change x
		Z = np.tile(z,len(x)*len(y))
		Y = np.tile(y,(len(z),len(x)))
		Y = np.ndarray.flatten(Y.T)
		X = np.tile(x,(len(z)*len(y),1))
		X = np.ndarray.flatten(X.T)
		if mirror:
			X = - X
		axis = [X,Y,Z]
		#rename the coordinates
		Xi = axis[perm[0]]
		Yi = axis[perm[1]]
		Zi = axis[perm[2]]
		ug['y'] = Yi
		ug['z'] = Zi
		ug['x'] = Xi
		ug = ug.sort_values(by=['z', 'y', 'x'])
		# sorted_df = sorted_df.reset_index(drop=True)
		pot = [None] * maxderiv
		x = np.asarray(demesh(ug['x']))/scale
		x = np.around(x, decimals = decimals)
		y = np.asarray(demesh(ug['y']))/scale
		y = np.around(y, decimals = decimals)
		z = np.asarray(demesh(ug['z']))/scale
		z = np.around(z, decimals = decimals)
		step = np.array((x[1]-x[0],y[1]-y[0],z[1]-z[0]))
		origin = np.array((x[0],y[0],z[0]))
		
		name = 'potential'

		data = np.array(ug[name])
		shape = (len(x),len(y),len(z))
		dim = shape[::-1]
		if data.ndim > 1:
			m_dim = data.shape[-1]
		elif data.ndim == 1:
			m_dim = 1
		dim += (m_dim,)
		data = np.asarray(data.reshape(dim).transpose(2, 1, 0, 3))
		order = int((m_dim-1)/2)
		pot[order] = xr.DataArray(data * scale**order, 
											dims = ('x', 'y', 'z', 'l'), 
											coords = {'x': x, 'y': y, 'z': z, 'l': _derivative_names[order]},
											attrs = dict(derivative_order = order,
														step = step,
														origin = origin)
											)

		obj = cls(name=elec_name, data=pot)
		obj.generate(maxderiv)
		return obj

	@classmethod
	def from_vtk(cls, elec_name, file, maxderiv=4, scale = 1, decimals = 10):
		'''Load grid potential data from vtk StructurePoints and create a 'SimulatedElectrode' object

		Parameters
		elec_name: file name string
		maxderiv: max derivative order to generate
		scale: the ratio between the new unit to the old unit, (scale = new / old, e.g. scale = 1mm/100µm = 10) 
				used to rescale the field and higher order derivative
		decimals: the coordinates accuracy will be truncated to the designated decimal
		-------

		Returns
		-----
		GridLayout
		'''
		ug = pv.UniformGrid(file)
		x = np.linspace(ug.origin[0], ug.origin[0] + ug.spacing[0] * (ug.dimensions[0] - 1), ug.dimensions[0]) / scale
		x = np.around(x, decimals = decimals)
		y = np.linspace(ug.origin[1], ug.origin[1] + ug.spacing[1] * (ug.dimensions[1] - 1), ug.dimensions[1]) / scale
		y = np.around(y, decimals = decimals)
		z = np.linspace(ug.origin[2], ug.origin[2] + ug.spacing[2] * (ug.dimensions[2] - 1), ug.dimensions[2]) / scale
		z = np.around(z, decimals = decimals)
		pot = [None] * maxderiv
		for name in ug.array_names:
			if name not in ("potential", "field"):
				continue
			data = ug.point_data[name]
			shape = ug.dimensions
			dim = shape[::-1]
			if data.ndim > 1:
				m_dim = data.shape[-1]
			elif data.ndim == 1:
				m_dim = 1
			dim += (m_dim,)
			data = np.asarray(data.reshape(dim).transpose(2, 1, 0, 3))
			order = int((m_dim-1)/2)
			pot[order] = xr.DataArray(data * scale**order, 
												dims = ('x', 'y', 'z', 'l'), 
												coords = {'x': x, 'y': y, 'z': z, 'l': _derivative_names[order]},
												attrs = dict(derivative_order = order,
															step = np.asarray(ug.spacing) / scale,
															origin = np.asarray(ug.origin) / scale)
												)
		obj = cls(name=elec_name, data=pot)
		obj.generate(maxderiv)
		return obj
	

	@classmethod
	def from_csv(cls, elec_name, file, maxderiv=4, scale = 0.001, decimals = 10, skiprows=9,sep=',',names=['x','y','z','potential','normE','Ex','Ey','Ez']):
		'''Load grid potential data from .csv file and create a 'SimulatedElectrode' object

		Parameters
		elec_name: electrode name
		file: file name string
		maxderiv: max derivative order to generate
		scale: the ratio between the new unit to the old unit, (scale = new / old, e.g. scale = 1mm/1m = 0.001) 
				used to rescale the field and higher order derivative
		decimals: the coordinates accuracy will be truncated to the designated decimal
		sep: default is matching the output from comsol
		skiprows: number of rows skipped in the csv file, default is matching the output from comsol
		names: dataframe column name, need to match the order in the output file
		perm: permutation of the coordinate. in multipole code, coordinates are [radial, height, axial], in ansys coordinates are [radial,axial,height] so perm would be [0,2,1]
		-------

		Returns
		-----
		GridLayout
		'''
		# x,y,z demesh, numpy function
		ug = pd.read_csv(file, skiprows=skiprows, sep=sep,names=names)
		ug = ug.sort_values(by = ['z', 'y','x'], ascending = [True, True,True])
		pot = [None] * maxderiv
		x = np.asarray(demesh(ug['x']))/scale
		x = np.around(x, decimals = decimals)
		y = np.asarray(demesh(ug['y']))/scale
		y = np.around(y, decimals = decimals)
		z = np.asarray(demesh(ug['z']))/scale
		z = np.around(z, decimals = decimals)
		step = np.array((x[1]-x[0],y[1]-y[0],z[1]-z[0]))
		origin = np.array((x[0],y[0],z[0]))
		for name in ('potential',['Ex','Ey','Ez']):
			data = np.array(ug[name])
			shape = (len(x),len(y),len(z))
			dim = shape[::-1]
			if data.ndim > 1:
				m_dim = data.shape[-1]
			elif data.ndim == 1:
				m_dim = 1
			dim += (m_dim,)
			data = np.asarray(data.reshape(dim).transpose(2, 1, 0, 3))
			order = int((m_dim-1)/2)
			pot[order] = xr.DataArray(data * scale**order, 
												dims = ('x', 'y', 'z', 'l'), 
												coords = {'x': x, 'y': y, 'z': z, 'l': _derivative_names[order]},
												attrs = dict(derivative_order = order,
															step = step,
															origin = origin)
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

	




