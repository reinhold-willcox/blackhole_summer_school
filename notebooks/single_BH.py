# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Black Hole formation

# ### This is a jupyter notebook. It allows you to run python code interactively, in separate cells each with their own output.
#
# ### This particular notebook is run on a remote cloud server, in this case a Binder environment, so that you don't need to worry about installing anything or configuring your software environment. The only requirment is access to the web.



# %matplotlib inline

# +
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt

import ipywidgets as widgets
from ipywidgets import HBox, VBox, interact

from compasUtils import printCompasDetails


# -

def get_data(index, isSingles=True, isCoarse=True):
    first = 'singles' if isSingles else 'binaries'
    second = 'coarse' if isCoarse else 'fine'
    return h5.File('../data/{}/{}/Detailed_Output/BSE_Detailed_Output_{}.h5'.format(first, second, index))


# +
# TODO give an introduction to single star evolution:
# Talk about how stars have several intrinsic properties: Mass, Radius, Luminosity, Effective Temperature, and Core Mass
# Talk about how stars evolve by fusing elements in their cores, growing the core mass
# As the stars age, all of these intrinsic properties will "evolve", similarly to a person growing from a baby to a child to an adult to an old person
# We break up the stars into different phases 
# -

data = get_data(0)
print(data.keys())

# +
# Make plot of the evolution of the mass and radius with time

fig, axes = plt.subplots(ncols=2, figsize=(20, 8))

colormap = mpl.cm.rainbow

for index in range(9):

    # Pick out the correct data file
    data = get_data(index)

    # Extract the data from this file
    mass = data['Mass(1)'][()]
    radius = data['Radius(1)'][()]
    time = data['Time'][()]
    stellar_type = data['Stellar_Type(1)'][()]
    record_type = data['Record_Type'][()]

    # Create a mask, to show only the interesting parts of the data
    mask = record_type == 4 
    mask &= time < 1e3 #& (time > .01) 
    mask &= stellar_type < 10

    # Set the color of the line
    color=colormap(index/9)
    
    # First plot: Mass vs Time
    ax = axes[0]
    ax.plot(time[mask], mass[mask], color=color, lw=3)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Mass [$M_\odot$]')
    ax.set_xscale('log')
    ax.set_xlim(left=.1)
    ax.grid()
    
    # Second plot: Radius vs Time
    ax = axes[1]
    ax.plot(time[mask], radius[mask])
    ax.plot(time[mask], radius[mask], color=color, lw=3)
    ax.set_xlabel('Time [Myr]')
    ax.set_ylabel('Radius [$R_\odot$]')
    ax.set_xscale('log')
    ax.set_xlim(left=1)
    ax.set_yscale('log')
    ax.grid()
# -



# +
#printCompasDetails(data)

# Want to extract arrays for time, Lum, and Teff, and make an HR based on this
time = np.zeros((9, get_data(0)['Time'][()].shape[0]))
Lum =  np.zeros((9, get_data(0)['Time'][()].shape[0]))
Teff = np.zeros((9, get_data(0)['Time'][()].shape[0]))

for index in range(100):

    # Pick out the correct data file
    data = get_data(index, isCoarse=False)

    # Extract the data from this file
    record_type = data['Record_Type'][()]
    stellar_type = data['Stellar_Type(1)'][()]
    mask = (record_type == 4) & (stellar_type < 7)
    t = data['Time'][()][mask]
    time[index] = t[-1] # fill to last value
    time[index][:t.shape[0]] = t
    l = data['Luminosity(1)'][()][mask]
    Lum[index] = l[-1] # fill to last value
    Lum[index][:l.shape[0]] = l
    te = data['Teff(1)'][()][mask]
    Teff[index]= te[-1]
    Teff[index][:te.shape[0]] = te


def make_interactive_HR(logTcut=-2):

    fig, ax = plt.subplots(figsize=(6, 4))

    time_pre = np.where(time < np.power(10.0, logTcut), time, 0)
    max_t = np.max(time_pre, axis=1)
    mask = np.where( time == max_t[:,None], True, False)

    ax.plot(Teff[mask], Lum[mask], 'bo')

    xlim = (1e3, 1e5) #ax.get_xlim()
    ylim = (1e3, 1e8) #ax.get_ylim()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.invert_xaxis()
    ax.set_xlabel(r'Effective Temperature [T/K]')
    ax.set_ylabel(r'Luminosity [$L/L_\odot$]')

   # Add lines of const radii
    for R in np.logspace(-1, 5, 13):
        #print(R)
        exp = "{:.2e}".format(R)
        #print(exp)
        #exp = exp[-3] + exp[-1]
        #if ((int(exp) % 2) == 1):  # skip odd ones to remove clutter
        #    continue
        T_K = np.logspace(3, 7, 41)  # in K
        T = T_K / 6e3  # Tsol=6e3K

        def get_L(t):  # assumes K
            return R * R * t * t * t * t

        L = get_L(T)
        ax.plot(T_K, L, '--k', alpha=0.2)
        # Plot the Rsol text at the bottom and right
        Lbot = ylim[0] * 8  # Lsun  -2
        Trgt = xlim[0] * 2  # 3e3
        Tbot = np.sqrt(np.sqrt(Lbot / (R * R))) * 6e3  # K
        Lrgt = get_L(Trgt / 6e3)
        alpha = 0.4
        s = "$R_\odot^{{{exp}}}$".format(exp=exp)
        if (Tbot > Trgt) and (Tbot < xlim[1]):
            ax.text(x=Tbot, y=Lbot, s=s, alpha=alpha)
        elif (Lrgt > Lbot) and (Lrgt < ylim[1]):
            ax.text(x=Trgt, y=Lrgt, s=s, alpha=alpha)
            
    ax.set_xlim(xlim[::-1])
    ax.set_ylim(ylim)
#colormap = mpl.cm.rainbow


# -

interact(make_interactive_HR, logTcut=widgets.FloatSlider(min=-2.0, max=2.0, step=0.01, value=-2.0))













from astroUtils import PtoA
mbh = 33
p = 11.6 # yr
m1 = 0.8
a = PtoA(Mtot=mbh+m1, P=11.6*365.25) #AU

a*215 # Rsol

# !ls

# !./COMPAS


