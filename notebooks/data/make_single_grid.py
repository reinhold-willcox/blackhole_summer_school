import numpy as np

with open('single_grid_coarse.txt', 'w') as f:
    for m1 in np.linspace(5, 50, 10):
        f.write("--initial-mass-1 {} --initial-mass-2 0.5 -a 10000\n".format(m1))

with open('single_grid_fine.txt', 'w') as f:
    for m1 in np.logspace(np.log10(0.5), 2, 250):
        f.write("--initial-mass-1 {} --initial-mass-2 0.5 -a 10000\n".format(m1))
