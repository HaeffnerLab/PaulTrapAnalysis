"""
trajectory.py

Container class for analyzing and plot the particle trajectories in a given Trap.

"""
import numpy as np
import pandas as pd
import xarray as xr
from collections import OrderedDict
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import scipy.signal
import time
import math
from matplotlib import cm, gridspec
from mpl_toolkits import mplot3d
import mpl_toolkits.mplot3d.axes3d as p3
import cvxpy as cvx
from .Trap import Trap

class Trajectory:

    def __init__(self, trap, initstate = [0,0,0,0,0,0], tstop = 1.E-6, tstep = 1.E-10, phase = 0, bound = [200.E-6, 200.E-6,200.E-6]):
        '''
        

        Parameters
        --------
        trap: an object created by the imported class Trap
        initstate: initial state, size-6 array of x-coordinate, y-coordinate, z-coordinate, x-velocity, y-velocity, and z-velocity
        tstop: simulation duration
        tstep: simulation time step
        phase: initial phase of rf drive
        bound: size-3 array of [xbound, ybound, zbound]. bound for x/y/z coordinate (usually related to the trap size) 

        '''

        self.trap = trap
        self.initstate = initstate
        self.tstop = tstop
        self.tstep = tstep
        self.phase = phase
        self.bound = bound

        return


    def eom_3d(self, state, t, phase, bound):
        '''Equation of motion for a single charged particle in electric field. SI units
        
        Parameters
        -------
        state: size-6 array of x-coordinate, y-coordinate, z-coordinate, x-velocity, y-velocity, and z-velocity
        t: array of time samples, measured in seconds
        phase: initial phase of rf drive
        bound: size-3 array of [xbound, ybound, zbound]. bound for x/y/z coordinate (usually related to the trap size) 
        
        Returns
        -------
        d(state)/dt = [dx/dt, dy/dt, dv/dt, dvx/dt, dvy/dt, dvz/dt]


        '''
        q = self.trap.config['charge']
        m = self.trap.config['mass']
        o = self.trap.config['Omega']
        x,y,z = state[0], state[1], state[2]
        dx, dy, dz = state[3], state[4], state[5]
        xbound,ybound,zbound = bound[0], bound[1], bound[2]
        E_dc = self.Trap.dc_potential(x=x,y=y,z=z,derivative=1,expand=True)
        E_rf = self.Trap.rf_potential(x=x,y=y,z=z,derivative=1,expand=True)
        E = E_dc + E_rf*np.cos(o*t + phase)
        Ex,Ey,Ez = E.sel(l='x'), E.sel(l='y'), E.sel(l='z')
        if np.abs(x) >= xbound or np.abs(y) >= ybound or np.abs(z) >= zbound: 
            return xbound, ybound, zbound, 0, 0, 0 # particle escape the trap
        else:
            dvx = q/m*Ex
            dvy = q/m*Ey
            dvz = q/m*Ez
            return dx, dy, dz, dvx, dvy, dvz

    
    def trajectory_3d(self):
        '''Simulating trajectories of a single charged particle in a Paul trap. 
        Solve the ODE

        Returns
        -------
        
        '''

        
        t = np.arange(0,self.tstop,self.tstep)
        with np.errstate(over='ignore'):
            psoln, errmsg = odeint(self.eom_3d, self.initstate, t, args=(self.phase,self.bound), full_output=1)

        return psoln




    # def eom_2d(self, state, t, E, bound):
    #     '''Equation of motion for a single charged particle in electric field. SI units
        
    #     Parameters
    #     -------
    #     state: size-4 array of x-coordinate, y-coordinate, x-velocity, and y-velocity
    #     t: array of time samples, measured in seconds
    #     E: size-2 array of [Ex, Ey]. Electric field in x/y direction (may contain time dependent part)
    #     E: size-3 array of [Ex, Ey, Ez]. Electric field in x/y/z direction (in this code, x/y is time dependent, may contain static term if rf biased; z is the axial direction, don't pick up time dependence)
    #     bound: size-2 array of [xbound, ybound]. bound for x/y coordinate (usually related to the trap size) 


    #     Returns
    #     -------
    #     d(state)/dt = [dx/dt, dy/dt, dvx/dt, dvy/dt]


    #     '''
    #     x,y = state[0], state[1]
    #     dx,dy = state[2], state[3]
    #     Ex,Ey = E[0],E[1]
    #     timedep = np.cos(Omega*t + phase)

    #     xbound,ybound = bound[0],bound[1]
    #     if np.abs(x) >= xbound or np.abs(y) >= ybound: 
    #         return xbound, ybound, 0, 0 # particle escape the trap
    #     else:
    #         dvx = q/m*Ex
    #         dvy = q/m*Ey
    #         return dx, dy, dvx, dvy






