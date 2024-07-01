class Trajectory:
    """
    A base class for particle trajectories.

    Functions
    ------
    - save(output_dir)
    - load_from_dir(path)
    """
    def __init__(self, parameters) -> None:
        """
        Initializes a trajectory model, either by
        passing a parameter dictionary to setup the
        basic parameters for simulation, or load an
        existing results from a path.
        
        Parameters
        ------
        parameters : dict or str
            If a parameter dictionary is passed, then
            sets up an equation of motion model with the
            parameters given. If a str is passed, then
            assumes the str is a path and loads existing 
            results from the given file path.
        """
        if type(parameters) is None: 
            return self.load_from_dir(parameters)
    
    def simulate(self, simulation_time, log_per_cycle=5):
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
            continue with Trajectory.get_results() or directly get the
            desired attributes from trajectory, such as trajectory.t, 
            trajectory.x, etc. 

        See Also
        ------
        Particle.simulate_trajectory
        """
        raise NotImplementedError

    def load_from_dir(self, path):
        raise NotImplementedError
    
    def save(self, output_dir):
        raise NotImplementedError
