import h5py as h5
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ipywidgets import AppLayout, FloatSlider, Play, IntSlider, widgets, HBox, VBox, interact, interactive



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

