#Module allegiance matrix calculation
#Last edited: 04-10-2020

#Step 0: Loading libraries

import sys
sys.path.append("..")
import os

%matplotlib inline

import scipy.io as sio
import numpy as np
from nilearn import plotting 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from fctools import networks, figures

#---- matplotlib settings
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'Helvetica'

#%%Step 1: Getting modules names and color pallete

labels = pd.read_csv(f'../support/modules.txt', sep = " ", header = None)

power_colors_new = {'AU':'#d182c6', 
                'CER':'#9fc5e8', 
                'CO':'#7d009d', 
                'DA':'#75df33', 
                'DM':'#ed1126', 
                'FP':'#f6e838', 
                'MEM':'#bebab5', 
                'SAL':'#2a2a2a', 
                'SOM':'#6ccadf', 
                'SUB':'#980000', 
                'UNC':'#f58c00', 
                'VA':'#00a074', 
                'VIS':'#5131ac',}

modules = sorted(labels[0].values)
network_pal = (sns.color_palette(power_colors_new.values()))
sns.palplot(sns.color_palette(power_colors_new.values()))

network_lut = dict(zip(map(str, np.unique(modules)), network_pal))

network_colors = pd.Series(modules).map(network_lut)
network_colors = np.asarray(network_colors)

n_roi = len(labels)
n_net = len(np.unique(modules))

#%%Step 2: Loading module assignment matrices

top_dir = '/home/finc/Dropbox/Projects/LearningBrain/'
mat = sio.loadmat(f'{top_dir}data/neuroimaging/03-modularity/dynamic/02-module_assignment/power_modules.mat')

idx = np.argsort(labels[0])

module_assignment = mat['modules']
module_assignment = module_assignment[:, :, :, idx, :]

#%%Step 3: calculating allegiance matrices

# Calculating allegiance matrices (mean over optimizations)
n_sub = module_assignment.shape[0]
n_ses = module_assignment.shape[1]
n_opt = module_assignment.shape[2]
n_nod = module_assignment.shape[3]

P = np.zeros((n_sub, n_ses, n_nod, n_nod))

for i in range(n_sub):
    print(f'Subject {i+1}')
    for j in range(n_ses):
        P[i,j,:,:] = networks.allegiance_matrix_opti(module_assignment[i,j,:,:,:])

np.save(f'{top_dir}data/neuroimaging/03-modularity/dynamic/03-allegiance_matrices/allegiance_matrix_power_opt_mean.npy', P)

#%% Calculating allegiance matrices for each window (mean over optimizations)

n_sub = len(module_assignment.shape[0])
n_ses = len(module_assignment.shape[1])
n_nod = len(module_assignment.shape[3])
n_win = len(module_assignment.shape[4])

W = np.zeros((n_sub, n_ses, n_win, n_nod, n_nod))

for i in range(n_sub):
    print(f'Subject {i+1}')
    W[i,j,:,:,:] = networks.all_window_allegiance_mean(module_assignment[i, j, :, :, :])

np.save(f'{top_dir}data/neuroimaging/03-modularity/dynamic/03-allegiance_matrices/window_allegiance_matrix_power_dualnback.npy', W)





















