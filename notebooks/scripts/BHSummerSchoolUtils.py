import h5py as h5
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ipywidgets import AppLayout, FloatSlider, Play, IntSlider, widgets, HBox, VBox, interact, interactive
from scipy.optimize import fsolve




########################################################################
### 
### Function to print the data from a given COMPAS HDF5 group 
### in a readable pandas template
### 
########################################################################

def printCompasDetails(data, *seeds, mask=()):
    """
    Function to print the full Compas output for given seeds, optionally with an additional mask
    """
    list_of_keys = list(data.keys())

    # Check if seed parameter exists - if not, just print without (e.g RunDetails)
    if ('SEED' in list_of_keys) | ('SEED>MT' in list_of_keys): # Most output files 
        #SEED>MT is a relic from older versions, but we leave this in for backwards compatibility

        # Set the seed name parameter, mask on seeds as needed, and set the index
        seedVariableName='SEED' if ('SEED' in list_of_keys) else 'SEED>MT'
        list_of_keys.remove(seedVariableName) # this is the index above, don't want to include it
    
        allSeeds = data[seedVariableName][()]
        seedsMask = np.in1d(allSeeds, seeds)
        if len(seeds) == 0: # if any seed is included, do not reset the mask
            seedsMask = np.ones_like(allSeeds).astype(bool)
        if mask == ():
            mask = np.ones_like(allSeeds).astype(bool)
        mask &= seedsMask

        df = pd.DataFrame.from_dict({param: data[param][()][mask] for param in list(data.keys())}).set_index(seedVariableName).T

    else: # No seed parameter, so do custom print for Run Details

        # Get just the keys without the -Derivation suffix - those will be a second column
        keys_not_derivations = []
        for key in list_of_keys:
            if '-Derivation' not in key:
                keys_not_derivations.append(key)
        
        # Some parameter values are string types, formatted as np.bytes_, need to convert back
        def convert_strings(param_array):
            if isinstance(param_array[0], np.bytes_):
                return param_array.astype(str)
            else:
                return param_array

        df_keys = pd.DataFrame.from_dict({param: convert_strings(data[param][()]) for param in keys_not_derivations }).T
        nCols = df_keys.shape[1] # Required only because if we combine RDs, we get many columns (should fix later)
        df_keys.columns = ['Parameter']*nCols
        df_drvs = pd.DataFrame.from_dict({param: convert_strings(data[param+'-Derivation'][()]) for param in keys_not_derivations }).T
        df_drvs.columns = ['Derivation']*nCols
        df = pd.concat([df_keys, df_drvs], axis=1)

    # Add units as first col
    units_dict = {key:data[key].attrs['units'].astype(str) for key in list_of_keys}
    df.insert(loc=0, column='(units)', value=pd.Series(units_dict))
    return df


def get_data(index, isSingles=True, isCoarse=True):
    first = 'singles' if isSingles else 'binaries'
    second = 'coarse' if isCoarse else 'fine'
    return h5.File('data/{}/{}/Detailed_Output/BSE_Detailed_Output_{}.h5'.format(first, second, index))


def make_interactive_widget(fig, update_func, value=0, smin=0, smax=100, step=1, speed=1):
    
    plt.ioff()
    
    slider = FloatSlider(
        orientation='horizontal',
        description='Factor:',
        value=value,
        min=smin,
        max=smax
    )
    slider.layout.margin = '0px 0% 0px 0%'
    slider.layout.width = '40%'
    interval = np.power(10.0, 3.0-speed)
    play = Play( value=value, step=step, max=smax, interval=interval, description="Press play", disabled=False, repeat=True )
    play.layout.margin = '0px 0% 0px 15%'
    widgets.jslink((play, 'value'), (slider, 'value'))
    widgets.VBox([play, slider])
    
    slider.observe(update_func, names='value')
    
    return AppLayout(
        left_sidebar=None,
        right_sidebar=None,
        center=fig.canvas,
        footer=widgets.VBox([play, slider]),
        pane_heights=[1, 6, 1],
        pane_widths=[0,7,0]
    )



## Constants
#G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
#
## Function to calculate gravitational potential at a point (x, y)
#def gravitational_potential(x, y, m1, m2, d):
#    # Positions of the two stars
#    x1, y1 = -d / 2, 0
#    x2, y2 = d / 2, 0
#
#    # Distances from the point to each star
#    r1 = np.sqrt((x - x1)**2 + (y - y1)**2)
#    r2 = np.sqrt((x - x2)**2 + (y - y2)**2)
#
#    # Gravitational potential at the point (x, y)
#    phi = -G * (m1 / r1 + m2 / r2)
#
#    return phi
#
## Function to find the Lagrangian points L1, L2, and L3
#def find_lagrangian_points(m1, m2, d):
#    # Defining the equation to solve for L1, L2, and L3
#    def equation_L1_L2_L3(x, m1, m2, d):
#        return G * m1 / (d/2 + x)**2 - G * m2 / (d/2 - x)**2 - G * (m1 + m2) * x / d**3
#
#    # Find L1
#    L1 = fsolve(equation_L1_L2_L3, 0.5 * d, args=(m1, m2, d))[0]
#
#    # Find L2
#    L2 = fsolve(equation_L1_L2_L3, -0.5 * d, args=(m1, m2, d))[0]
#
#    # Find L3
#    def equation_L3(x, m1, m2, d):
#        return G * m1 / (d/2 + x)**2 - G * m2 / (d/2 - x)**2 + G * (m1 + m2) * x / d**3
#    L3 = fsolve(equation_L3, -1.5 * d, args=(m1, m2, d))[0]
#
#    return L1, L2, L3
#
## Function to calculate and plot equipotential surfaces
#def plot_equipotential_surfaces(m1, m2, d):
#    # Define the grid
#    x_min, x_max = -1.5 * d, 1.5 * d
#    y_min, y_max = -1.5 * d, 1.5 * d
#    n_points = 500
#
#    x = np.linspace(x_min, x_max, n_points)
#    y = np.linspace(y_min, y_max, n_points)
#    X, Y = np.meshgrid(x, y)
#
#    # Calculate the potential at each point on the grid
#    Z = gravitational_potential(X, Y, m1, m2, d)
#
#    # Find the Lagrangian points
#    L1, L2, L3 = find_lagrangian_points(m1, m2, d)
#
#    # Calculate potential at Lagrangian points
#    phi_L1 = gravitational_potential(L1, 0, m1, m2, d)
#    phi_L2 = gravitational_potential(L2, 0, m1, m2, d)
#    phi_L3 = gravitational_potential(L3, 0, m1, m2, d)
#
#    # Plot the equipotential surfaces
#    plt.figure(figsize=(10, 8))
#    contour_levels = np.linspace(Z.min(), Z.max(), 50)
#    plt.contour(X, Y, Z, levels=contour_levels, cmap='inferno')
#    plt.colorbar(label='Gravitational Potential (J/kg)')
#
#    # Highlight the equipotential lines passing through Lagrangian points
#    plt.contour(X, Y, Z, levels=[phi_L1], colors='blue', linestyles='dashed', linewidths=2)
#    plt.contour(X, Y, Z, levels=[phi_L2], colors='green', linestyles='dashed', linewidths=2)
#    plt.contour(X, Y, Z, levels=[phi_L3], colors='red', linestyles='dashed', linewidths=2)
#
#    # Plot the positions of the stars and the Lagrangian points
#    plt.scatter([-d/2, d/2], [0, 0], color='white', s=100, edgecolor='black', label='Stars')
#    plt.scatter([L1, L2, L3], [0, 0, 0], color='black', s=50, marker='x', label='Lagrangian Points')
#    plt.legend()
#    plt.title('Equipotential Surfaces around a Binary Star System with Lagrangian Points')
#    plt.xlabel('x (m)')
#    plt.ylabel('y (m)')
#    plt.grid(True)
#    plt.show()
#
## Example usage
## Masses of the stars (in kg)
#m1 = 1.989e30  # Mass of star 1 (e.g., the mass of the Sun)
#m2 = 1.989e30  # Mass of star 2 (e.g., another Sun-like star)
#
## Separation between the stars (in meters)
#d = 1.496e11  # Approximately 1 AU
#
## Plot the equipotential surfaces
#plot_equipotential_surfaces(m1, m2, d)




####
#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.optimize import fsolve
#
## Constants
#G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
#
## Function to calculate gravitational potential at a point (x, y)
#def gravitational_potential(x, y, m1, m2, d):
#    # Positions of the two stars
#    x1, y1 = -d * m2 / (m1 + m2), 0
#    x2, y2 = d * m1 / (m1 + m2), 0
#
#    # Distances from the point to each star
#    r1 = np.sqrt((x - x1)**2 + (y - y1)**2)
#    r2 = np.sqrt((x - x2)**2 + (y - y2)**2)
#
#    # Gravitational potential at the point (x, y)
#    phi = -G * (m1 / r1 + m2 / r2)
#
#    return phi
#
## Function to find the Lagrangian points L1, L2, and L3
#def find_lagrangian_points(m1, m2, d):
#    # Positions of the two stars
#    x1 = -d * m2 / (m1 + m2)
#    x2 = d * m1 / (m1 + m2)
#
#    # Equation for L1 and L2
#    def equation_L1_L2(x, m1, m2, d):
#        r1 = x - x1
#        r2 = x - x2
#        return G * m1 / r1**2 - G * m2 / r2**2 - G * (m1 + m2) * x / d**3
#
#    # Equation for L3
#    def equation_L3(x, m1, m2, d):
#        r1 = x - x1
#        r2 = x - x2
#        return G * m1 / r1**2 - G * m2 / r2**2 + G * (m1 + m2) * x / d**3
#
#    # Solve for L1, L2, and L3
#    L1 = fsolve(equation_L1_L2, x1 + d * 0.5, args=(m1, m2, d))[0]
#    L2 = fsolve(equation_L1_L2, x2 + d * 0.5, args=(m1, m2, d))[0]
#    L3 = fsolve(equation_L3, -d * 1.5, args=(m1, m2, d))[0]
#
#    return L1, L2, L3
#
## Function to calculate and plot equipotential surfaces
#def plot_equipotential_surfaces(m1, m2, d):
#    # Define the grid
#    x_min, x_max = -1.5 * d, 1.5 * d
#    y_min, y_max = -1.5 * d, 1.5 * d
#    n_points = 500
#
#    x = np.linspace(x_min, x_max, n_points)
#    y = np.linspace(y_min, y_max, n_points)
#    X, Y = np.meshgrid(x, y)
#
#    # Calculate the potential at each point on the grid
#    Z = gravitational_potential(X, Y, m1, m2, d)
#
#    # Find the Lagrangian points
#    L1, L2, L3 = find_lagrangian_points(m1, m2, d)
#
#    # Calculate potential at Lagrangian points
#    phi_L1 = gravitational_potential(L1, 0, m1, m2, d)
#    phi_L2 = gravitational_potential(L2, 0, m1, m2, d)
#    phi_L3 = gravitational_potential(L3, 0, m1, m2, d)
#
#    # Plot the equipotential surfaces
#    plt.figure(figsize=(10, 8))
#    contour_levels = np.linspace(Z.min(), Z.max(), 50)
#    plt.contour(X, Y, Z, levels=contour_levels, cmap='inferno')
#    plt.colorbar(label='Gravitational Potential (J/kg)')
#
#    # Highlight the equipotential lines passing through Lagrangian points
#    plt.contour(X, Y, Z, levels=[phi_L1], colors='blue', linestyles='dashed', linewidths=2)
#    plt.contour(X, Y, Z, levels=[phi_L2], colors='green', linestyles='dashed', linewidths=2)
#    plt.contour(X, Y, Z, levels=[phi_L3], colors='red', linestyles='dashed', linewidths=2)
#
#    # Plot the positions of the stars and the Lagrangian points
#    plt.scatter([-d * m2 / (m1 + m2), d * m1 / (m1 + m2)], [0, 0], color='white', s=100, edgecolor='black', label='Stars')
#    plt.scatter([L1, L2, L3], [0, 0, 0], color='black', s=50, marker='x', label='Lagrangian Points')
#    plt.legend()
#    plt.title('Equipotential Surfaces around a Binary Star System with Lagrangian Points')
#    plt.xlabel('x (m)')
#    plt.ylabel('y (m)')
#    plt.grid(True)
#    plt.show()
#
## Example usage
## Masses of the stars (in kg)
#m1 = 1.989e30  # Mass of star 1 (e.g., the mass of the Sun)
#m2 = 0.5 * 1.989e30  # Mass of star 2 (e.g., half the mass of the Sun)
#
## Separation between the stars (in meters)
#d = 1.496e11  # Approximately 1 AU
#
## Plot the equipotential surfaces
#plot_equipotential_surfaces(m1, m2, d)
#
