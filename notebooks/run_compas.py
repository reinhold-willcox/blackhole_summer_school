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

# +
import numpy as np

np.random.seed(np.datetime64('now').astype(int))


# +
def run_compas(*args, **kwargs):

    # Need a random string to hold the data
    str_length = 10
    ascii_lowercase = list('abcdefghijklmnopqrstuvwxyz')
    random_str = ''.join(np.random.choice(ascii_lowercase) for i in range(str_length))
    
    outdir = '../data/{}/'.format(random_str)

    print(outdir)

run_compas()
# -


