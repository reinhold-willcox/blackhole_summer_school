# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

import numpy as np
import h5py as h5


# +
def generate_outdir(str_length=10):
    # Need a random string to hold the data
    np.random.seed(np.datetime64('now').astype(int))
    ascii_lowercase = list('abcdefghijklmnopqrstuvwxyz')
    random_str = ''.join(np.random.choice(ascii_lowercase) for i in range(str_length))
    return random_str
    
def run_compas(*args, **kwargs):
    outdir=generate_outdir()
    # !././scripts/COMPAS -n 1 -o 'data/on_the_fly_data/' -c {outdir}
    return outdir
    
def get_myf(outdir):
    return h5.File('data/on_the_fly_data/{}/{}.h5'.format(outdir, outdir), 'r')
    
# +

outdir = run_compas()
myf = get_myf(outdir)
myf.keys()
# -



