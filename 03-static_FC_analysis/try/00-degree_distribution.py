#STATIC CONNECTIVITY
#Degree distribution check
#Last edited: 17-10-2020 - last block doesn't work

#Step 0: Load libraries, load data and print useful information
#determining subjects included in the analysis (as indices list for 5-D data arrays)
#vectorizing connection matrices

from IPython.core.pylabtools import figsize
#from jupyterthemes import jtplot

from datetime import timedelta
from time import time

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import itertools
import random
import bct
#%matplotlib inline

#from load_data import *

# Subjects with complete neuroimaging data

top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/func_connect/05-motion_and_outlier_control/'

groups = pd.read_csv('/home/alado/datasets/RBH/behavioural/subj_info.csv')
#subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])
subs = groups[groups['group'].isin(['med', 'sport', 'control'])].reset_index()
subs = groups[~groups['group'].isnull()]
subs = subs.drop(subs.index[21])
subs = subs.drop(subs.index[12:17])
subs = subs.drop(subs.index[9])
subs = subs.drop(subs.index[8]).reset_index()
subs = subs.reset_index() # Index now matching numpy arrays
#sub_mat = grp_ass[~grp_ass['group'].isnull()]
#sub_mat = sub_mat.reset_index() # Index now matching numpy arrays
#del subs['index']
subs = subs.drop(['index', 'level_0', 'n_scans_ses-1_run-1', 'n_scans_ses-3_run-1'], axis=1)

#load matrices static connectivity
mat_pow = np.load(f'{top_dir}/func_connect/03-correlation_matrices/static/LB_nback_power_static_correlation_matrices.npy')
mat_sch = np.load(f'{top_dir}/func_connect/03-correlation_matrices/static/LB_nback_schaefer_static_correlation_matrices.npy')

# Subjects included in dual n-back analysis
#ex_dual_mot - create it based on previous analysis
#sub_dual = subs[~subs['sub'].isin(ex_dual_mot)] # Exclude dual
sub_dual = subs
print('=== Subjects in dual n-back: {}'.format(len(sub_dual)))
print(sub_dual.index.values)
# sub_dual.index[sub_dual['group'] == 'Control'].values
# sub_dual.index[sub_dual['group'] == 'Experimental'].values

# Subjects included in rest analysis
#ex_rest - exclude rest
#create it
#sub_rest = sub_mat[~sub_mat['sub'].isin(ex_rest)] 
sub_rest = subs
print('\n=== Subjects in resting state: {}'.format(len(sub_rest)))
print(sub_rest.index.values)

# Subjects included in both rest and task
#sub_both = sub_mat[~sub_mat['sub'].isin(ex_rest + ex_dual_mot)]
sub_both = subs
print('\n=== Subjects in both rest and task: {}'.format(len(sub_both)))
print(sub_both.index.values)

# Vectorize 5-D matrices into 4-D matrices returning upper diagonal elements 
# of connection matrix 
get_max_edge = lambda x: int(x * (x-1) / 2) 

n_pow = mat_pow.shape[-1]
n_sch = mat_sch.shape[-1]
m_pow = get_max_edge(n_pow)
m_sch = get_max_edge(n_sch)
triu_ind_pow = np.triu_indices(n_pow, k=1)
triu_ind_sch = np.triu_indices(n_sch, k=1)

shape = list(mat_pow.shape[:3])
mat_pow_v = np.empty(tuple(shape + [m_pow]))
mat_sch_v = np.empty(tuple(shape + [m_sch]))

for sub in range(shape[0]):
    for ses in range(shape[1]):
        for task in range(shape[2]):
            mat_pow_v[sub, ses, task] = mat_pow[sub, ses, task][triu_ind_pow]
            mat_sch_v[sub, ses, task] = mat_sch[sub, ses, task][triu_ind_sch]

print('mat_pow_v shape: {}'.format(mat_pow_v.shape))
print('mat_sch_v shape: {}'.format(mat_sch_v.shape))

#%%Step 1: Overall degree distribution
#Display weighted degree distribution using kernel density estimation depending on:

#task condition
#session
#experimental group

#NO REST HERE

def draw_kde_hist(data, color, label):
    sns.distplot(
        data,
        hist=False,
        kde=True,
        color=color,
        kde_kws={'linewidth': 2},
        label=label)
    
color = {
    '1': sns.color_palette("hls", 8)[4],
    '2': sns.color_palette("hls", 8)[3],
    '3': sns.color_palette("hls", 8)[2],
    '4': sns.color_palette("hls", 8)[1],
    '5': sns.color_palette("hls", 8)[0],}

#Schaefer atlas mean degree distributions (all sessions)

fig = plt.figure(figsize=(15, 5), facecolor='white')
                 
                 
for i, task in enumerate(['1', '2', '3', '4']):
    
    draw_kde_hist(
        data=mat_sch_v[sub_both.index.values, 0, i].flatten(),
        color=color[task],
        label=task)
    plt.title('All subjects, all sessions, Schaefer')
    plt.xlabel('FC strength')
    plt.ylabel('probability density')

#Schaefer atlas individual degree distributions (session 1)

fig = plt.figure(figsize=(15, 25), facecolor='white')

for idx, sub_idx in enumerate(sub_both.index.values[:36]):
    
    fig.add_subplot(9, 4, idx+1)
    
    for i, task in enumerate(['1', '2', '3', '4']):
        
        draw_kde_hist(
            data=mat_sch_v[(sub_idx, 0, i)].flatten(),
            color=color[task],
            label=task)
        plt.title(sub_both['sub'][sub_idx])
        plt.tight_layout()


#Power atlas mean degree distributions (all sessions)

fig = plt.figure(figsize=(15, 5), facecolor='white')

for i, task in enumerate(['1', '2', '3', '4']):
    
    draw_kde_hist(
        data=mat_pow_v[sub_both.index.values, 0, i].flatten(),
        color=color[task],
        label=task)
    plt.title('All subjects, all sessions, Power')
    plt.xlabel('FC strength')
    plt.ylabel('probability density')
    

#Power atlas individual degree distributions (session 1)

fig = plt.figure(figsize=(15, 25), facecolor='white')

for idx, sub_idx in enumerate(sub_both.index.values[:36]):
    
    fig.add_subplot(9, 4, idx+1)
    
    for i, task in enumerate(['1', '2', '3', '4']):
        
        draw_kde_hist(
            data=mat_pow_v[(sub_idx, 0, i)].flatten(),
            color=color[task],
            label=task)
        plt.title(sub_both['sub'][sub_idx])
        plt.tight_layout()
#%%Step 2: Degree distribution differences between groups and sessions
#Perform basic statistical analysis of degree distribution (compare mean and std between groups and sessions).


def create_dataframe(metric, sub_df, var_name):
    '''Creating dataframe from 2-D numpy array
    
    Parameters:
        metric (ndarray): Two dimensional array of shape (N_sub, 4).
        sub_df (DataFrame): Subject group assignment.
        var_name (str): Specifies name of metric.
        
    Note:
        Index of sub_df should match corresponding rows of metric. 
        
    Returns:
        df (DataFrame): Tidy format with group, session and value indices.'''

    df = pd.DataFrame(
        metric, 
        columns=(['ses-1', 'ses-3']))

    df = pd.merge(df, sub_df, left_index=True, right_index=True)
    df = df.drop(['sub'], axis=1)
    df = df.set_index('group')
    df = df.stack()
    df.index = df.index.rename('session', level=1)
    df.name = var_name
    df = df.reset_index()
    return df

# Calculate mean network degree 
k_pow = mat_pow_v.mean(axis=3)
k_std_pow = mat_pow_v.std(axis=3)
k_sch = mat_sch_v.mean(axis=3)
k_std_sch = mat_sch_v.std(axis=3)


#Schaefer atlas mean connectivity and std of connectivity

metrics = ['k_sch', 'k_std_sch']
tasks = ['1', '2', '3', '4']

fig, ax = plt.subplots(4, 2, 
                       figsize=(20, 15),
                       facecolor='w',
                       edgecolor='k')

for iy, task in enumerate(tasks):
    for ix, var_name in enumerate(metrics):
            
        if var_name == metrics[0]:
            df = create_dataframe(k_sch[:,:,iy], sub_both, var_name)
        else:
            df = create_dataframe(k_std_sch[:,:,iy], sub_both, var_name)

        sns.boxplot(
            ax=ax[iy, ix],
            x='session',
            y=var_name,
            hue='group',
            data=df,
            palette='husl')
        
        ax[iy, ix].set_axisbelow(True)
        if ix == 0:
            ax[iy, ix].set_ylabel(task, size='large')
        else:
            ax[iy, ix].set_ylabel('')
        if iy == 0:
            ax[iy, ix].set_title(var_name, size='large')
        ax[iy, ix].set_xlabel('')
            
plt.tight_layout()
  
#Power atlas mean connectivity and std of connectivity

metrics = ['k_pow', 'k_std_pow']
tasks = ['1', '2', '3', '4']

fig, ax = plt.subplots(4, 2, 
                       figsize=(20, 15),
                       facecolor='w',
                       edgecolor='k')

for iy, task in enumerate(tasks):
    for ix, var_name in enumerate(metrics):
            
        if var_name == metrics[0]:
            df = create_dataframe(k_pow[:,:,iy], sub_both, var_name)
        else:
            df = create_dataframe(k_std_pow[:,:,iy], sub_both, var_name)

        sns.boxplot(
            ax=ax[iy, ix],
            x='session',
            y=var_name,
            hue='group',
            data=df,
            palette='husl')
        
        ax[iy, ix].set_axisbelow(True)
        if ix == 0:
            ax[iy, ix].set_ylabel(task, size='large')
        else:
            ax[iy, ix].set_ylabel('')
        if iy == 0:
            ax[iy, ix].set_title(var_name, size='large')
        ax[iy, ix].set_xlabel('')
            
plt.tight_layout()    

#%% 
subs = sub_both.index.values
task = 2

corr = np.empty((len(subs), 4, 4))

for sub_id, sub in enumerate(subs):
    for prod in itertools.product(list(range(4)), repeat=2):
        corr[(sub_id, *prod)] = np.corrcoef(
            x=mat_pow_v[sub, prod[0], task, :],         
            y=mat_pow_v[sub, prod[1], task, :]
        )[0][1]
        
print(np.mean(corr, axis=0))

np.max(corr, axis=0)








    
    
    
    
    
    
    
    
    
    
    
    