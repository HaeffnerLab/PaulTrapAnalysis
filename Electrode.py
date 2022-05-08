
import numpy as np
from scipy.ndimage.interpolation import map_coordinates

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

	def __init__(self,name='',dc=0.,rf=0.):
		self.name = name
		self.dc = dc
		self.rf = rf

	def potential(self, x, derivative=0,r0=1.,output=None):
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
		r0 : float
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

	def plot(self ax, label=None, color=None,**kw):
		'''Visualize and plot this electrode in the supplied axes.

		'''
		pass




class GridElectrode(Electrode):
	'''Electrode based on a precalculated grid of electrical potentials. 
	(From either BEM or FEM methods)

	Parameters
	---------
	data : 
	origin :
	spacing : 

'''

@classmethod
def from_result():
	'''Create a 'GridLayout' object from a 'bem.result.Result' instance

	Parameters
	-----

	Returns
	-----
	GridLayout
	'''

@classmethod
def from_vtk():
	'''Load grid potential data from vtk StructurePoints and create a 'GridLayout' object

	Parameters
	-------

	Returns
	-----
	GridLayout
	'''


	

@classmethod
def from_fld():
	'''Load grid potential data from fld file (exported from ansys) and create a 'GridLayout' object
	fld file: potential data file exported from ansys without grid points

	Parameters
	-------

	Returns
	-----
	GridLayout
	'''


