# PaulTrapAnalysis

## Description
PaulTrapAnalysis is a toolkit for developing and analyzing RF Paul traps (still under development). It can analyze the electric propoerties, trapping parameters, and multipole control of a given trap design.

The Electrode class contains information about the electrical potential of individual electrodes and uses xarray to store the potential data with 3D grid points. The Trap class consists of two dictionaries: a dictionary of the Electrodes in the trap, and a dictionary of the basic trap parameters (mass, charge, drive frequency etc.). It takes advantage of the pandas dataframe and dictionary to support queries in a user friendly manner.

Other container calsses will takes a Trap object as an argument and analyze the different properties of this specific trap design (multipole control analysis, particle trajectory analysis, anharmonicity analysis etc.). For example, multipoles calss expands the trapping potential in multipoles (spherical harmonic basis) and allows one find the efficient electrode voltage solution to generate a specific potential shape.