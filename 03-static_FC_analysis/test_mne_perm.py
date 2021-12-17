#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 09:31:14 2021

@author: alado
"""
import numpy as np
from math import sqrt
import numpy as np

from mne.utils import check_random_state, verbose, logger
from mne.parallel import parallel_func
from mne.stats.cluster_level import _get_1samp_orders

#%%
def _max_stat(X, X2, perms, dof_scaling):
    """Aux function for permutation_t_test (for parallel comp)."""
    n_samples = len(X)
    mus = np.dot(perms, X) / float(n_samples)
    stds = np.sqrt(X2[None, :] - mus * mus) * dof_scaling  # std with splitting
    max_abs = np.max(np.abs(mus) / (stds / sqrt(n_samples)), axis=1)  # t-max
    return max_abs
#%%
top_dir = '/media/alado/TOSHIBA_EXT/thesis/output'
folder = '03-spectrogram_FFT'
patient_id = 46
session_nr = 1

X_full = np.load(f'{top_dir}/{folder}/spectrograms/normalized/db/CSC1_LA1_0_Alison_post_db.npy')

X = np.zeros((10, X_full[:, :,:].shape[1]*X_full[:, :,:].shape[2]))
for sp in range(X_full.shape[0]):
    X[sp,:] = X_full[sp, :,:].flatten()

seed=0
n_permutations=10000
tail=0
n_jobs=8


n_samples, n_tests = X.shape
X2 = np.mean(X ** 2, axis=0)  # precompute moments
mu0 = np.mean(X, axis=0)
dof_scaling = sqrt(n_samples / (n_samples - 1.0))
std0 = np.sqrt(X2 - mu0 ** 2) * dof_scaling  # get std with var splitting
T_obs = np.mean(X, axis=0) / (std0 / sqrt(n_samples))
rng = check_random_state(seed)
orders, _, extra = _get_1samp_orders(n_samples, n_permutations, tail, rng)
perms = 2 * np.array(orders) - 1  # from 0, 1 -> 1, -1
logger.info('Permuting %d times%s...' % (len(orders), extra))
parallel, my_max_stat, n_jobs = parallel_func(_max_stat, n_jobs)
max_abs = np.concatenate(parallel(my_max_stat(X, X2, p, dof_scaling)
                                  for p in np.array_split(perms, n_jobs)))
max_abs = np.concatenate((max_abs, [np.abs(T_obs).max()]))
H0 = np.sort(max_abs)
if tail == 0:
    p_values = (H0 >= np.abs(T_obs[:, np.newaxis])).mean(-1)
elif tail == 1:
    p_values = (H0 >= T_obs[:, np.newaxis]).mean(-1)
elif tail == -1:
    p_values = (-H0 <= T_obs[:, np.newaxis]).mean(-1)