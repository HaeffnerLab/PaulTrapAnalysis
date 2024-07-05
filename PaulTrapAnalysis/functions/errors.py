import pandas as pd
import numpy as np
import os

def calc_err(V0, Phi0, err='lstsq_norm'):
    V = V0.flatten()
    Phi = Phi0.flatten()
    
    if err == 'lstsq_norm':
        return np.linalg.norm(V-Phi)**2/np.shape(V)[0]
    elif err == 'abe_norm':
        return np.sum(np.abs(V-Phi))/np.shape(V)[0]
    elif err == 'lstsq':
        return np.linalg.norm(V-Phi)**2
    elif err == 'abe':
        return np.sum(np.abs(V-Phi))
    elif err == 'avg_pct':
        return np.sum(np.abs(V-Phi)/V)/np.shape(V)[0]

def make_err_table(V, Phi, config_name, display_table=False, save_table=True):
    """
    Calculate different error values based on given error metrics.
    
    Parameters
    ----------
    V : arr
        Original potential values
    Phi : arr
        Predicted potential values
    config_name : str
        Configuration name used as table titles
    display_table : bool
        Whether to print the error table
    save_table : bool
        Whether to save the new table
    
    Returns
    -------
    df : pd.DataFrame
        The error table
    err : float
        The mean square error value
    """
    err_names = ['lstsq_norm', 'lstsq', 'abe_norm', 'abe', 'avg_pct']
    err_names_display = ['Mean Square Error', 'Total Squared Error', 
                         'Mean Absolute Error', 'Total Absolute Error',
                         'Mean Percentage Error']
    df = {"Error Names": err_names_display}
    err = [calc_err(V, Phi, i) for i in err_names]
    file_name = 'all_errors.csv'
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        df[config_name] = err
    else:
        df = pd.DataFrame({"Error Names": err_names_display, config_name: err})
    if display_table:
        try:
            display(df)
        except:
            print(df)
    if save_table:
        df.to_csv(file_name, index=False)
    return df, err[0]