import scipy.constants as ct

class Particle: 
    """
    A general base class for the particls to be trapped.

    Attributes
    ------
    mass : [kg]
        Particle mass 
    charge : [c]
        Particle charge
    trajectory : dict
        A dictionary for particle trajectories. 

    Functions
    ------
    - setup_trajectory(parameters, trajectory_model)
    - simualte_trajectory(parameters, trajectory_model)
    """
    def __init__(self, mass, charge):
        """       
        Parameters
        ------
        mass : [kg]
            The mass of the particle in SI units (kg)
        charge : [c]
            The charge of the particle in SI units
        """
        self.mass = mass 
        self.charge = charge 
    
    def setup_trajectory(self, parameters, trajecotry_model):
        """
        Sets up the particle trajectory model based on the 
        given parameters and a trajectory_model. This function
        only establishes the setup or the initial conditions,
        and it does not run the simulation.
        
        Parameters
        ------
        parameters : dict 
            The input parameters to set the initial conditions 
            and simulation constants.
        trajectory_model : Trajectory
            The equation of motion model used to conduct the integration
            for finding the trajectories.
        
        Returns
        ------
        trajectory : Trajectory
            A Trajectory object with initial conditions. To
            obtain the simulated trajectories, continue with 
            Particle.simulate_trajectory or trajectory.simulate 
        """
        parameters['m'] = self.mass 
        parameters['q'] = self.charge 
        self.trajectory = trajecotry_model(parameters)
        return self.trajectory
    
    def simulate_trajectory(self, simulation_time, log_per_cycle=5):
        """
        Simulate the trajectories and log the resulting values.
        
        Parameters
        ------
        simulation_time : [s]
            The total simulation time in SI units.
        log_per_cycle : int
            The number of points recorded per RF cycle.
        
        Returns
        ------
        trajectory : Trajectory
            The trajectory object with results. To obtain the results, 
            continue with trajectory.get_results() or directly get the
            desired attributes from trajectory, such as trajectory.t, 
            trajectory.x, etc.

        See Also
        ------
        Trajectory.simulate
        """
        return self.trajectory.simulate(simulation_time, log_per_cycle)


class Electron(Particle): 
    def __init__(self):
        super().__init__(ct.m_e, -ct.e)

class Ca40(Particle): 
    def __init__(self):
        super().__init__(40*ct.atomic_mass, ct.elementary_charge)