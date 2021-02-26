#! /usr/bin/env python3.8
import time

from astropy.io import ascii
import numpy as np
###############################################################################
t_i = time.time()
###############################################################################
working_dir = '/home/edgar/Documents/pyhacks/astro'
fname = 'GCs.txt'
###############################################################################
data = ascii.read(f'{working_dir}/{fname}',
    guess=False, data_start=3, delimiter='\t')
print(data)
###############################################################################
t_f = time.time()
print(f'Running time: {t_f - t_i: .2f} s')
