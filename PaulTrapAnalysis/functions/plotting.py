import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm
import pandas as pd
from PaulTrapAnalysis.functions import potentials
from PaulTrapAnalysis.functions import data

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def add_value_labels(ax, spacing=0.1, threshold=0.01):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        if abs(y_value) < threshold:
            continue
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.2f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.
            

def plot_multipole_vs_expansion_height(height, s, roi, vs, plot_multipoles=True):
    position1 = [0, 0, height*1e-3]
    s.update_origin_roi(position1, roi)
#     print np.dot(s.multipole_expansions,vs[0])

    Nmulti = s.multipole_expansion.shape[0]
    if plot_multipoles:
        fig,ax = plt.subplots(len(vs),1,figsize = (10, 24))
    for i,v in enumerate(vs):
        coeffs = s.setVoltages(v)
        if plot_multipoles:
            ax[i].bar(range(Nmulti),coeffs)
        max_coeff = np.max(coeffs)
        min_coeff = np.min(coeffs)
        margin = (max_coeff - min_coeff)*0.5
        ymax = max_coeff + margin
        ymin = min_coeff - margin
        if plot_multipoles:
            ax[i].set_ylim(ymin, ymax)
            ax[i].set_title(s.trap.names[i])
            fig.canvas.draw()
            add_value_labels(ax[i])
    if plot_multipoles:
        tick_name = list(s.multipole_print_names)
        tick_name += list(range(len(tick_name), Nmulti))
        plt.xticks(range(Nmulti), tick_name, rotation = -90)
        fig.tight_layout(pad=1)
        plt.show()
    return s
    

def plot_potential(coord, Phi, size=1, n=100, scale=1,
                   cmap='viridis', title=r'$\Phi(x,y,z)$',
                   ax=None):
    x, y, z = coord
    if ax is None:
        fig = plt.figure(figsize=(6,6))
        ax = plt.axes(projection='3d')
    im = ax.scatter(x[::n]*scale, y[::n]*scale, z[::n]*scale, 
                    s=size, c=Phi[::n], 
                    marker='.', cmap=cmap)
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_zlabel('z (um)')
    ax.set_title(title)
    if ax is None:
        fig.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.show()
    else:
        return im
    

def get_color(Phi, ref=None):
    if ref is None:
        ref = Phi
    minima = min(ref)
    maxima = max(ref)

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima)#, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap='viridis')

    color = mapper.to_rgba(Phi)
    return color


def plot_Mj(Mj, mutipole_names=['C', 'Ey', 'Ez', 'Ex', 'U3', 'U4', 'U2', 'U5', 'U1'], 
            Mj_threshold=0.01,
            title='', save_fig=False):
    
    fig, ax = plt.subplots(figsize=(0.3*len(Mj), 4))
    ax.bar(list(range(1,len(Mj)+1)), Mj.flatten())
    add_value_labels(ax, threshold=Mj_threshold)
    #ax.axvline(np.argmax(abs(Mj))+1, label='j ='+str(np.argmax(abs(Mj))+1), 
    #            linestyle='--', color='r', alpha=0.7)
    tick_name = list(mutipole_names)
    tick_name += list(range(len(tick_name)+1, len(Mj)+1))
    df = pd.DataFrame({'Mj': [f'{float(i):.3e}' for i in Mj]}, index=tick_name)
    try:
        display(df)
    except:
        print(df)
    ax.set_xticks(range(1,len(Mj)+1), tick_name, rotation = -90)
    ax.set_xlabel(r'$j$')
    ax.set_ylabel(r'$M_j$')
    #ax.legend()
    ax.grid()
    ax.set_title(title)
    plt.tight_layout()
    if save_fig:
        plt.savefig(f'{title}.pdf')
    plt.show()
    

def plot_projection(coord, V, Phi, r0, Mj, order, scale=1e-3, plot_scale=1e3, n=1, figsize=(15,5), library='other'):
    fig, ax = plt.subplots(figsize=figsize, ncols=3)
    axis = ['x', 'y', 'z']
    axis_ind = [0, 1, 2]
    for i in range(3):
        axis_zeroed = [j for j in axis_ind if j != i]
        j, k = axis_zeroed
        ind = np.where((abs(coord[j]) < 5e-4) & (abs(coord[k]) < 5e-4))
        ax[i].plot(coord[i][ind][::n]*plot_scale, V.flatten()[ind][::n], 'bo', label='Data')
        ax[i].plot(coord[i][ind][::n]*plot_scale, Phi.flatten()[ind][::n], 'rx', label='Fit')
        
        fit_i = np.linspace(min(coord[i])-abs(min(coord[i]))*0.2, max(coord[i])+abs(max(coord[i]))*0.2, 20)
        temp_coord = [0,0,0]
        temp_coord[i] = fit_i
        temp_coord[j] = temp_coord[k] = np.zeros(np.shape(fit_i))
        Phi_curve = potentials.generate_potential_single_shot(*temp_coord, Mj, order, scale=scale, library=library)
        ax[i].plot(fit_i*plot_scale, Phi_curve.flatten(), 'r--')
        ax[i].set_xlabel(axis[i] + r' (um)')
        ax[i].set_ylabel('Potential (V)')
        ax[i].set_title(f'{axis[j]}, {axis[k]} = {r0[j]*plot_scale:.2f}, {r0[k]*plot_scale:.2f} (um)')
        ax[i].grid(True)
        ax[i].legend()
        
    plt.tight_layout()
    plt.show()


def plot_all_potentials(plot_coord_fit, V, Phi, region, m, plot_scale=1e3):
    """
    Plot the original, fitted, and residual potential.
    """
    fig1 = plt.figure(figsize=plt.figaspect(1/3))
    ax1 = fig1.add_subplot(1, 3, 1, projection='3d')
    ax2 = fig1.add_subplot(1, 3, 2, projection='3d')
    ax3 = fig1.add_subplot(1, 3, 3, projection='3d')
    
    im1 = plot_potential(plot_coord_fit, V.flatten(), size=10, n=m, scale=plot_scale, 
                   title=r'Original $\Phi(x,y,z)$ on ' + region, ax=ax1)
    im2 = plot_potential(plot_coord_fit, Phi.flatten(), size=10, n=m, scale=plot_scale, 
                   title=r'Fit $\Phi(x,y,z)$ on ' + region, ax=ax2)
    im3 = plot_potential(plot_coord_fit, abs(Phi.flatten()-V.flatten()), size=10, n=m, scale=plot_scale, 
                   title=r'Residual $\Delta\Phi(x,y,z)$ on ' + region, cmap='Reds', ax=ax3)
    fig1.colorbar(im1, ax=ax1, shrink=0.8)
    fig1.colorbar(im2, ax=ax2, shrink=0.8)
    fig1.colorbar(im3, ax=ax3, shrink=0.8)
    plt.tight_layout()
    plt.show()


def plot_gradient(get_grad, Mj, s, r0, electrode, order,length=10, 
                  scale=1e-3, plot_scale=1e3, library='manual', n=1, 
                  rotate=False):
    """
    Plot the gradient of the potential.
    
    Parameters
    ----------
    get_grad : func
        A function to compute the gradients, in the form of f(x, y, z)
    s : MultipoleControl object
        An object that contains all the electrodes and potential data
    """
    fig = plt.figure(figsize=(6,6))
    ax = plt.axes(projection='3d')
    
    coord_roi, coord0, V_roi, V0 = data.get_potential_data(s, electrode=electrode)
    X_roi, Y_roi, Z_roi = coord_roi
    
    x, y, z = data.get_grid(X_roi, Y_roi, Z_roi, r0, rotate=rotate)
    if n != 1:
        indices = np.random.randint(0, len(x), len(x)//n)
        x, y, z = x[indices], y[indices], z[indices]
    print('>>> Evaluating', len(x), 'data')
    Phi = potentials.generate_potential_single_shot(x, y, z, Mj, order, scale=scale, library=library)
    color = get_color(Phi)[::n]
    grad = get_grad(x, y, z)
    
    u, v, w = grad.T
    im = ax.quiver(x*plot_scale, y*plot_scale, z*plot_scale, 
                   u*plot_scale, v*plot_scale, w*plot_scale, 
                   length=length, color=color, 
                   arrow_length_ratio=0.3, normalize=1)
    ax.set_xlabel('x (um)')
    ax.set_ylabel('y (um)')
    ax.set_zlabel('z (um)')
    ax.set_title(r'$\bf{E}$ (x,y,z)')
    plt.tight_layout()
    plt.show()
    
def plot_contour(phi, x, y, scale=1e6, nspacing=1):
    
    fig = plt.figure(figsize=(7.5,6))
    im = plt.scatter(x[::nspacing]*scale, y[::nspacing]*scale, c=phi[::nspacing], cmap='seismic', norm='log')
    fig.colorbar(im)
    plt.xlabel('x (um)')
    plt.ylabel('y (um)')
    plt.show()