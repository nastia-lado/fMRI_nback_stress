#Static connectivity t-test
#Last edited: 20-01-2021

import sys
sys.path.append("..")
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import nilearn
import mne.stats
import nilearn.connectome

from nilearn import datasets, plotting, input_data, signal
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure, sym_matrix_to_vec
from nilearn.plotting import plot_roi
from nilearn.image.image import mean_img
from sklearn.covariance import EmpiricalCovariance
from permute.core import two_sample, one_sample
from statsmodels.stats import multitest
import nibabel as nib
from scipy import stats
import parallelPermutationTest as ppt
from pathlib import Path

import seaborn as sns
sns.reset_orig()

#%%Step 1: Loading data
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/func_connect/05-fc_statistics/'
suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
denoising = '_denoised_acompcor_task_effects.nii.gz'

groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])
sess = ['ses-1', 'ses-3']
cond = ['1', '2', '3', '4', '5']
tasks = ['nback']
task = 'nback'
n_scans_ses_1 = pd.Series.tolist(groups['n_scans_ses-1_run-1'])
n_scans_ses_3 = pd.Series.tolist(groups['n_scans_ses-3_run-1'])
t_r = 3
alpha = 0.01 
subs_sport = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport'])])
subs_med = pd.Series.tolist(groups['sub'][groups['group'].isin(['med'])])
subs_control = pd.Series.tolist(groups['sub'][groups['group'].isin(['control'])])
#here [0] is to convert from tuple to array
idx_subs_sport = np.array(np.where(np.isin(subs,subs_sport)))[0]
idx_subs_med = np.array(np.where(np.isin(subs,subs_med)))[0]
idx_subs_control = np.array(np.where(np.isin(subs,subs_control)))[0]
n_subs_sport = len(subs_sport)
n_subs_med = len(subs_med)
n_subs_control = len(subs_control)

idx_subs_grouped = [idx_subs_sport, idx_subs_med, idx_subs_control]
n_subs_grouped = [n_subs_sport, n_subs_med, n_subs_control]
subs_labels_grouped = ['sport', 'med', 'control']

multitest_methods_names = ['bonferroni','holm','holm-sidak','simes-hochberg',
                           'fdr_bh','fdr_by','fdr_tsbh','fdr_tsbky','fdr_gbs']
                           

#%%Step 2: loading atlases for labels on matrices
# Loading Power ROIs coordinates
power = datasets.fetch_coords_power_2011()
power_coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T
#doesn't work
power_labels = power_coords

#Harvard-Oxford 2 atlases (48+21+2=71)
ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
ho_cort_labels = ho_cort.labels
ho_sub_labels = ho_sub.labels
ho_labels = ho_cort_labels + ho_sub_labels
#drop bg at 0 and 49
ho_labels.remove('Background')
ho_labels.remove('Background')

#Talairach (72)
talairach = datasets.fetch_atlas_talairach('ba')
talairach_filename = talairach.maps
talairach_labels = talairach.labels
#drop bg at 0
talairach_labels.remove('Background')

#Craddock
craddock = datasets.fetch_atlas_craddock_2012()
craddock_filename = craddock.scorr_mean
craddock_masker = input_data.NiftiMapsMasker(maps_img=craddock_filename,
                                            standardize=True, 
                                            memory='nilearn_cache', verbose=5)

aal = datasets.fetch_atlas_aal()
aal_filename = aal.maps
aal_labels = aal.labels

#nback_power = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_power_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)
#nback_craddock = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_craddock_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)
nback_ho_cort = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_harvard-oxford_cort_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)
nback_ho_sub = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_harvard-oxford_sub_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)
nback_ho = np.concatenate((nback_ho_cort, nback_ho_sub), axis=3)
nback_talairach = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_talairach_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)
nback_aal = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_aal_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)

nback = np.asarray([[nback_ho, 'nback_harvard-oxford'],
                    [nback_talairach, 'nback_talairach'],
                    [nback_aal, 'nback_aal'],
                    #[nback_power, 'nback_power'], 
                    #[nback_craddock, 'nback_craddock']
                    ])

atlas_labels =  np.asarray([[ho_labels, 'harvard-oxford'],
                    [talairach_labels, 'nback_talairach'],
                    [aal_labels, 'nback_aal'],
                    #[power_labels, 'power'],
                    #[craddock_filename, 'craddock']
                    ])

#%%Step 3: Load correlation matrices
#atlas:
#-power
#-Harvard-Oxford cortical-subcortical atlases (Makris 2006)
#-Talairach atlas
#-Craddock atlas

#correlation matrices weighted atlas-based 
correl_matrices_wei_z = []
for p, (i, atlas) in enumerate(nback):
    correlation_matrices_wei_z = np.load(f'{top_dir}/func_connect/04-correlation_matrices/static/LB_{nback[p,1]}_static_correlation_matrices_wei_z.npy')
    correl_matrices_wei_z.append(correlation_matrices_wei_z)

#correlation matrices atlas-based non-weighted
# correl_matrices_z = []
# for p, (i, atlas) in enumerate(nback):
#     correlation_matrices_z = np.load(f'{top_dir}/func_connect/04-correlation_matrices/static/LB_{nback[p,1]}_static_correlation_matrices_z.npy')
#     correl_matrices_z.append(correlation_matrices_z)

#%%
#visualize weighted atlas-based correlations
for p, atlas in enumerate(nback):
    for ses in range(len(sess)):
        for con in range(len(cond)):    
            correl_matrix_wei_z = correl_matrices_wei_z[p]
            correl_matrix_wei_z = correl_matrix_wei_z[:,ses,con,:,:]        
            plotting.plot_matrix(np.mean(correl_matrix_wei_z, axis=0), vmin=-1., vmax=1., colorbar=True,
                  title=f'{nback[p,1]} ses-{ses} condition-{con} correlation matrix weighted')

# for p, atlas in enumerate(nback):
#     for ses in range(len(sess)):
#         for con in range(len(cond)):    
#             correl_matrix_z = correl_matrices_z[p]
#             correl_matrix_z = correl_matrix_z[:,ses,con,:,:]
#             plotting.plot_matrix(np.mean(correl_matrix_z, axis=0), vmin=-1., vmax=1., colorbar=True,
#                   title=f'{nback[p,1]} ses-{ses} condition-{con} correlation matrix non-weighted')

#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 12:54:33 2021

@author: alado
"""
from scipy.stats import ttest_ind, ttest_rel

def perm_test(X, Y, perms=1000, alpha=.05, nan='propagate', ttype = 'ind'):
    """Takes two equally shaped sets of data and performs a 2 sample permutation 
    T-test. This test can handel nan values, so if your sampels are not equal 
    in size you can inset nan's were needed. Data can be 1, 2, 3, or 4 
    dimensions. Data sets can either be arrays, up to 4 dimension, or the 
    outermost dimesion can be a list.
    Args:
        X: first data set where the first dimension is the observations.
        Y: second data set where the first dimension is the observations.
        perms:  the number of times to permute the data, typically 1000 or greater.
        alpha: significance value for the probability of rejecting the null hypothesis
            when it is true. Typically set at 0.05.
    Returns: 
        Global: array of T values where the observed T values were larger than 1-alpha% of 
            all the null permuted data. Best for data that is 1D or 2D unless 
            you want separate comparisons for each index. 
        Part: array of T values where the observed T value was larger than the permuted
            T distributions in the same index point of the array. Very conservative, observed 
            values must be greater than All permuted values in the same index point.
        P_array: array showing p values where the observed t values exceeded 1-alpha% 
            of the permuted values at each index point of a multidimensional array. 
            Typical permutation test output, best for multidimensional data where index matters. 
        null_Tmax: The single T value from the permuted null distribution that is above 1-alpha 
        t_score: array of T values from the observed data.
        nullT: list of arrays of T values from the permuted data.
        max_null: identify outliers in null permuted data after ttest
        max_obs: identify outliers in observed data after ttest"""
    
    from scipy.stats.mstats import mquantiles
    import numpy as np
        
    if np.array(X).shape != np.array(Y).shape:
        raise ValueError('datasets have diffrent dimensions')
    
    dims = len(np.array(X).shape) # find dimensions of data
         
    if dims == 2:
        _, dim1 = np.array(X).shape
        t_score = np.zeros((dim1))      

    ##### test real data ######
        for i in range(dim1):
            which_subs = ~np.isnan(np.array(X)[:, i])
            samp1 = np.array(X)[which_subs, i]
            samp2 = np.array(Y)[which_subs, i]
            
            diff = samp1 - samp2
            n_sub = diff.shape[0]
            t_score[i] = diff.mean(0)/(diff.std(0)/np.sqrt(n_sub))
   
    ## set up datasets to be permuted over perms times##    
        nullT=[]
        null_count=np.zeros((dim1))
        
        for iteration in range(perms):
            permA=np.copy(X)
            permB=np.copy(Y)
            for a in range(len(permA)):    
                
    ### create dummy permutation of 1s & 0s in a dim1 x dim2 x dim3 array ###
                perm_frame = np.random.randint(0,2,size=(dim1))
        ###### seperate permuted data into two artifical groups A and B ######
                permA[a]=perm_frame
                permB[a]=perm_frame
                
    #### replace dummy code with the condition data ####
                permA[a]=np.where(permA[a]==0, X[a], Y[a])
                permB[a]=np.where(permB[a]==0, Y[a], X[a])
   
    ##### t-test permuted data conditions A & B ####
            t_perms = np.zeros((dim1))
            
            for i in range(dim1):
                which_subs = ~np.isnan(np.array(X)[:, i])
                perm1 = permA[which_subs, i]
                perm2 = permB[which_subs, i]
                
                diff = perm1 - perm2
                n_sub = diff.shape[0]
                #t_perms[i] = diff.mean(0)/(diff.std(0)/np.sqrt(n_sub))
                if ttype == 'ind':
                    #print(ttest_ind(perm1, perm2))
                    t_perms[i] = ttest_ind(perm1, perm2)[0]
                    
                elif ttype == 'rel':
                    t_perms[i] = ttest_rel(perm1, perm2)[0]
    
    # store values from each of the 1000 iterations to create Tmax distribution #
            nullT.append(t_perms)
            
    ##### get p values across all dimentions. 
        null_count=np.zeros((perms,dim1))
        for i in range(perms):
            null_count[i]=nullT[i]>=t_score

        null_count=np.sum(null_count, axis=0)
    
        P_val=null_count/perms  
        P_array = np.where(P_val<alpha, P_val,np.nan)
        
    ##### identify Tmax in the permuted data #####
        null_Tmax=mquantiles(np.array(nullT).max((1)), 1-alpha)
        
        ###identify outliers 
        max_null=np.max(nullT)
        max_obs=np.max(t_score)       

    #####compare observed diffrences vs Tmax permuted diffrences ######
        # index by index difference #
        Part=np.zeros((dim1))
        TorF = t_score[np.newaxis, :] > nullT
        mask = TorF.sum(0) == perms
        Part[mask] = t_score[mask]
        Part[mask == False] = np.nan
    
        # Global difference #    
        Global=np.zeros((dim1))
        mask = t_score > null_Tmax
        Global[mask] = t_score[mask]
        Global[mask == False] = np.nan
    
############################################################################
    
    else: raise ValueError('X has to be 2 dimensional')
    return Global, Part, P_array, null_Tmax, t_score, nullT, max_obs, max_null   

#%%
list_strings = []
n_permutations = 10000
for group in range(len(subs_labels_grouped)):
    idx_subs = idx_subs_grouped[group]
    n_subs = n_subs_grouped[group]
    string = f'results for {subs_labels_grouped[group]} group'
    list_strings.append(string)
    print(f'results for {subs_labels_grouped[group]} group')
    
    for i, (j, atlas) in enumerate(nback):
        print(f'atlas {atlas}')
        for con in range(len(cond)):  
            print(f'condition {con}')
            
            corr_matr_s0 = correl_matrices_wei_z[i]
            corr_matr_s0 = corr_matr_s0[idx_subs,0,con,:,:]
                    
            corr_matr_s1 = correl_matrices_wei_z[i]
            corr_matr_s1 = corr_matr_s1[idx_subs,1,con,:,:]
            
            #take only lower triangle
            corr_vec_s0 = []
            corr_vec_s1 = []
            for k in range(n_subs):
                matrix = corr_matr_s0[k,:,:]
                corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
                corr_vec_s0.append(corr_vec)
                matrix = corr_matr_s1[k,:,:]
                corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
                corr_vec_s1.append(corr_vec)
            corr_vec_s0 = np.array(corr_vec_s0)
            corr_vec_s1 = np.array(corr_vec_s1)
    
            #compare the actual distribution with the t-distribution
            data = np.concatenate((corr_vec_s0, corr_vec_s1))
    
            #T0, p_values, H0 = mne.stats.permutation_t_test(data, n_permutations, n_jobs=8)
            H0, p_values = perm_test(corr_vec_s0, corr_vec_s1, perms=n_permutations)
                
            p_values = np.array(p_values)
            #T0 = np.array(T0)
            H0 = np.array(H0)
            
            np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_p_values_mne.npy', p_values)
            #np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_T0_mne.npy', T0)
            np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_H0_mne.npy', H0)
    
            #n, bins, patches = plt.hist(H0, 25, histtype='bar', density=True)
            #plt.title('Permutation Null Distribution')
            #plt.axvline(x=T0[0], color='red')
            #x = np.linspace(stats.t.ppf(0.0001, df), stats.t.ppf(0.9999, df), 100)
            #plt.plot(x, stats.t.pdf(x, df), lw=2, alpha=0.6)
            #plt.show()
            
            p_values = np.array(p_values)
            diagonal = np.ones(len(corr_matr_s0[0,:,0]))
            p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(p_values, diagonal)
            
            if p_values <= 0.05:
                string = f'significant results {nback[i,1]} condition-{con}'
                list_strings.append(string)
                print(f'significant results {nback[i,1]} condition-{con}')
                correlation_matrices_wei_z = correl_matrices_wei_z[i]
                diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(p_values, diagonal)
                
                #get locations of the significant results to get labels
                a1,a2 = np.where(p_vals_matrix <= 0.05)
                sign_labels1 = []
                sign_labels2 = []
                atlas_lab = atlas_labels[i,0]
                for k in range(len(a1)):
                    sign_labels1.append(atlas_lab[a1[k]])
                    sign_labels2.append(atlas_lab[a2[k]])
                
                string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                list_strings.append(string)
                print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
                string = f'The connectivity strength differs significantly between pre and post for {subs_labels_grouped[group]} group'
                list_strings.append(string)
                print(f'The connectivity strength differs significantly between pre and post for {subs_labels_grouped[group]} group')
                for k in range(int(len(sign_labels1)/2)):
                    string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                    list_strings.append(string)
                    print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
            else:
                string = f"no significant results {nback[i,1]} condition-{con}"
                list_strings.append(string)
                print(f"no significant results {nback[i,1]} condition-{con}")

#%%approximate permutation test CPU one sample pre/post mne
from netneurotools import stats as nnstats
#import perm_2sample

list_strings = []
n_permutations = 50000
for group in range(len(subs_labels_grouped)):
    idx_subs = idx_subs_grouped[group]
    n_subs = n_subs_grouped[group]
    string = f'results for {subs_labels_grouped[group]} group'
    list_strings.append(string)
    print(f'results for {subs_labels_grouped[group]} group')
    
    for i, (j, atlas) in enumerate(nback):
        print(f'atlas {atlas}')
        for con in range(len(cond)):  
            print(f'condition {con}')
            
            corr_matr_s0 = correl_matrices_wei_z[i]
            corr_matr_s0 = corr_matr_s0[idx_subs,0,con,:,:]
                    
            corr_matr_s1 = correl_matrices_wei_z[i]
            corr_matr_s1 = corr_matr_s1[idx_subs,1,con,:,:]
            
            #take only lower triangle
            corr_vec_s0 = []
            corr_vec_s1 = []
            for k in range(n_subs):
                matrix = corr_matr_s0[k,:,:]
                corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
                corr_vec_s0.append(corr_vec)
                matrix = corr_matr_s1[k,:,:]
                corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
                corr_vec_s1.append(corr_vec)
            corr_vec_s0 = np.array(corr_vec_s0)
            corr_vec_s1 = np.array(corr_vec_s1)
    
            #compare the actual distribution with the t-distribution
            data = np.concatenate((corr_vec_s0, corr_vec_s1))
    
            #T0, p_values, H0 = mne.stats.permutation_t_test(data, n_permutations, n_jobs=8)
            H0, p_values = nnstats.permtest_rel(corr_vec_s0, corr_vec_s1, n_perm=n_permutations)
                
            p_values = np.array(p_values)
            #T0 = np.array(T0)
            H0 = np.array(H0)
            
            np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_p_values_mne.npy', p_values)
            #np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_T0_mne.npy', T0)
            np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_H0_mne.npy', H0)
    
            #n, bins, patches = plt.hist(H0, 25, histtype='bar', density=True)
            #plt.title('Permutation Null Distribution')
            #plt.axvline(x=T0[0], color='red')
            #x = np.linspace(stats.t.ppf(0.0001, df), stats.t.ppf(0.9999, df), 100)
            #plt.plot(x, stats.t.pdf(x, df), lw=2, alpha=0.6)
            #plt.show()
            
            p_values = np.array(p_values)
            diagonal = np.ones(len(corr_matr_s0[0,:,0]))
            p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(p_values, diagonal)
        
            for method in multitest_methods_names:
                output = multitest.multipletests(p_values, alpha=alpha, method=method)
                reject = output[0]
                pval_corrected = output[1]
        
                if True in reject:
                    string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                    list_strings.append(string)
                    print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
                    correlation_matrices_wei_z = correl_matrices_wei_z[i]
                    diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                    p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                    
                    #get locations of the significant results to get labels
                    a1,a2 = np.where(p_vals_matrix <= 0.05)
                    sign_labels1 = []
                    sign_labels2 = []
                    atlas_lab = atlas_labels[i,0]
                    for k in range(len(a1)):
                        sign_labels1.append(atlas_lab[a1[k]])
                        sign_labels2.append(atlas_lab[a2[k]])
                    
                    string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                    list_strings.append(string)
                    print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
                    string = f'The connectivity strength differs significantly between pre and post for {subs_labels_grouped[group]} group'
                    list_strings.append(string)
                    print(f'The connectivity strength differs significantly between pre and post for {subs_labels_grouped[group]} group')
                    for k in range(int(len(sign_labels1)/2)):
                        string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                        list_strings.append(string)
                        print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
                else:
                    string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                    list_strings.append(string)
                    print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
        
with open(f'{out_dir}static/permut_pre-post_all_matrices_wei_mne.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item)

#%%approximate permutation test CPU one sample

list_strings = []
n_permutations = 50000

for group in range(len(subs_labels_grouped)):
    idx_subs = idx_subs_grouped[group]
    n_subs = n_subs_grouped[group]
    df = n_subs + n_subs - 2
    string = f'results for {subs_labels_grouped[group]} group'
    list_strings.append(string)
    print(f'results for {subs_labels_grouped[group]} group')
    for i, (j, atlas) in enumerate(nback):
        print(f'atlas {atlas}')
        for con in range(len(cond)):  
            print(f'condition {con}')
            
            corr_matr_s0 = correl_matrices_wei_z[i]
            corr_matr_s0 = corr_matr_s0[idx_subs,0,con,:,:]
                    
            corr_matr_s1 = correl_matrices_wei_z[i]
            corr_matr_s1 = corr_matr_s1[idx_subs,1,con,:,:]
            
            #take only lower triangle
            corr_vec_s0 = []
            corr_vec_s1 = []
            for k in range(n_subs):
                matrix = corr_matr_s0[k,:,:]
                corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
                corr_vec_s0.append(corr_vec)
            for k in range(n_subs):
                matrix = corr_matr_s1[k,:,:]
                corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
                corr_vec_s1.append(corr_vec)
            corr_vec_s0 = np.array(corr_vec_s0)
            corr_vec_s1 = np.array(corr_vec_s1)
            
            p_values = []
            T0 = []
            H0 = []
            #add saving p-values here
            for test in range(corr_vec.shape[0]): 
                p, t, dist = one_sample(corr_vec_s0[:, test], corr_vec_s1[:, test], stat='t', reps=n_permutations, keep_dist=True, seed=55)
                p_values.append(p)
                T0.append(t)
                H0.append(dist)
                
            p_values = np.array(p_values)
            T0 = np.array(T0)
            H0 = np.array(H0)
            
            np.save(f'{out_dir}static/{atlas}_group-{group}_con-{con}_static_correl_matrices_wei_z_p_values_mne.npy', p_values)
            np.save(f'{out_dir}static/{atlas}_group-{group}_con-{con}_static_correl_matrices_wei_z_T0_mne.npy', T0)
            np.save(f'{out_dir}static/{atlas}_group-{group}_con-{con}_static_correl_matrices_wei_z_H0_mne.npy', H0)
    
            #n, bins, patches = plt.hist(H0, 25, histtype='bar', density=True)
            #plt.title('Permutation Null Distribution')
            #plt.axvline(x=T0[0], color='red')
            #x = np.linspace(stats.t.ppf(0.0001, df), stats.t.ppf(0.9999, df), 100)
            #plt.plot(x, stats.t.pdf(x, df), lw=2, alpha=0.6)
            #plt.show()
        
            for method in multitest_methods_names:
                output = multitest.multipletests(p_values, alpha=alpha, method=method)
                reject = output[0]
                pval_corrected = output[1]
        
                if True in reject:
                    string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                    list_strings.append(string)
                    print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
                    correlation_matrices_wei_z = correl_matrices_wei_z[i]
                    diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                    p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                    
                    #get locations of the significant results to get labels
                    a1,a2 = np.where(p_vals_matrix <= 0.05)
                    sign_labels1 = []
                    sign_labels2 = []
                    atlas_lab = atlas_labels[i,0]
                    for k in range(len(a1)):
                        sign_labels1.append(atlas_lab[a1[k]])
                        sign_labels2.append(atlas_lab[a2[k]])
                    
                    string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                    list_strings.append(string)
                    print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
                    string = f'The connectivity strength differs significantly between pre and post for {subs_labels_grouped[group]} group'
                    list_strings.append(string)
                    print(f'The connectivity strength differs significantly between pre and post for {subs_labels_grouped[group]} group')
                    for k in range(int(len(sign_labels1))):
                        string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                        list_strings.append(string)
                        print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
                else:
                    string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                    list_strings.append(string)
                    print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
    
with open(f'{out_dir}static/permut_pre-post_all_matrices_wei_mne.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item)


#%%GPU permutation test pre/post all groups
list_strings = []

for group in range(len(subs_labels_grouped)):
    idx_subs = idx_subs_grouped[group]
    n_subs = n_subs_grouped[group]
    string = f'results for {subs_labels_grouped[group]} group'
    list_strings.append(string)
    print(f'results for {subs_labels_grouped[group]} group')
    for i, (j, atlas) in enumerate(nback):
        print(f'atlas {atlas}')
        for con in range(len(cond)):  
            print(f'condition {con}')
            p_vals_path = Path(f'/{out_dir}static/{atlas}_con-{con}_{subs_labels_grouped[group]}_static_correl_matrices_wei_z_p_values_gpu.npy')
            if p_vals_path.is_file():
                p_values = np.load(p_vals_path)
                
                p_values = np.array(p_values)
                correlation_matrices_wei_z = correl_matrices_wei_z[i]
                diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(p_values, diagonal)
                
                for method in multitest_methods_names:
                    output = multitest.multipletests(p_values, alpha=alpha, method=method)
                    reject = output[0]
                    pval_corrected = output[1]
            
                    if True in reject:
                        string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                        list_strings.append(string)
                        print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
                        correlation_matrices_wei_z = correl_matrices_wei_z[i]
                        diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                        p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                        
                        #get locations of the significant results to get labels
                        a1,a2 = np.where(p_vals_matrix <= 0.05)
                        sign_labels1 = []
                        sign_labels2 = []
                        atlas_lab = atlas_labels[i,0]
                        for k in range(len(a1)):
                            sign_labels1.append(atlas_lab[a1[k]])
                            sign_labels2.append(atlas_lab[a2[k]])
                        
                        string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                        list_strings.append(string)
                        print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
                        string = f'The connectivity strength differs significantly between pre and post for {subs_labels_grouped[group]} group'
                        list_strings.append(string)
                        for k in range(int(len(sign_labels1))):
                            string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                            list_strings.append(string)
                            print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
                    else:
                        string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                        list_strings.append(string)
                        print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
                
            else:
                print('No p-values file')
        
with open(f'{out_dir}static/permut_gpu_pre-post_matrices_wei.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item)
            
#%%GPU permutation test post sport/control
for i, (j, atlas) in enumerate(nback):
    print(f'atlas {atlas}')
    for con in range(len(cond)):  
        print(f'condition {con}')
        p_vals_path = Path(f'/{out_dir}static/{atlas}_con-{con}_sport-control_static_correl_matrices_wei_z_p_values_gpu.npy')
        if p_vals_path.is_file():
            p_values = np.load(p_vals_path)
            
            p_values = np.array(p_values)
            correlation_matrices_wei_z = correl_matrices_wei_z[i]
            diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
            p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(p_values, diagonal)
            
            for method in multitest_methods_names:
                output = multitest.multipletests(p_values, alpha=alpha, method=method)
                reject = output[0]
                pval_corrected = output[1]
        
                if True in reject:
                    string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                    list_strings.append(string)
                    print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
                    correlation_matrices_wei_z = correl_matrices_wei_z[i]
                    diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                    p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                    
                    #get locations of the significant results to get labels
                    a1,a2 = np.where(p_vals_matrix <= 0.05)
                    sign_labels1 = []
                    sign_labels2 = []
                    atlas_lab = atlas_labels[i,0]
                    for k in range(len(a1)):
                        sign_labels1.append(atlas_lab[a1[k]])
                        sign_labels2.append(atlas_lab[a2[k]])
                    
                    string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                    list_strings.append(string)
                    print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
                    string = f'The connectivity strength differs significantly between sport and control groups post'
                    list_strings.append(string)
                    print(f'The connectivity strength differs significantly between sport and control groups post')
                    for k in range(int(len(sign_labels1))):
                        string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                        list_strings.append(string)
                        print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
                else:
                    string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                    list_strings.append(string)
                    print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
            
        else:
            print('No p-values file')
        
with open(f'{out_dir}static/permut_gpu_post_sport-control_matrices_wei.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item)           

#%%GPU permutation test post sport/med
for i, (j, atlas) in enumerate(nback):
    print(f'atlas {atlas}')
    for con in range(len(cond)):  
        print(f'condition {con}')
        p_vals_path = Path(f'/{out_dir}static/{atlas}_con-{con}_sport-med_static_correl_matrices_wei_z_p_values_gpu.npy')
        if p_vals_path.is_file():
            p_values = np.load(p_vals_path)
            
            p_values = np.array(p_values)
            correlation_matrices_wei_z = correl_matrices_wei_z[i]
            diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
            p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(p_values, diagonal)
            
            for method in multitest_methods_names:
                output = multitest.multipletests(p_values, alpha=alpha, method=method)
                reject = output[0]
                pval_corrected = output[1]
        
                if True in reject:
                    string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                    list_strings.append(string)
                    print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
                    correlation_matrices_wei_z = correl_matrices_wei_z[i]
                    diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                    p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                    
                    #get locations of the significant results to get labels
                    a1,a2 = np.where(p_vals_matrix <= 0.05)
                    sign_labels1 = []
                    sign_labels2 = []
                    atlas_lab = atlas_labels[i,0]
                    for k in range(len(a1)):
                        sign_labels1.append(atlas_lab[a1[k]])
                        sign_labels2.append(atlas_lab[a2[k]])
                    
                    string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                    list_strings.append(string)
                    print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
                    string = f'The connectivity strength differs significantly between sport and meditation groups post'
                    list_strings.append(string)
                    print(f'The connectivity strength differs significantly between sport and meditation groups post')
                    for k in range(int(len(sign_labels1))):
                        string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                        list_strings.append(string)
                        print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
                else:
                    string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                    list_strings.append(string)
                    print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
            
        else:
            print('No p-values file')
        
with open(f'{out_dir}static/permut_gpu_post_sport-meditation_matrices_wei.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item)   
            
#%%GPU permutation test post meditation/control
for i, (j, atlas) in enumerate(nback):
    print(f'atlas {atlas}')
    for con in range(len(cond)):  
        print(f'condition {con}')
        p_vals_path = Path(f'/{out_dir}static/{atlas}_con-{con}_med-control_static_correl_matrices_wei_z_p_values_gpu.npy')
        if p_vals_path.is_file():
            p_values = np.load(p_vals_path)
            
            p_values = np.array(p_values)
            correlation_matrices_wei_z = correl_matrices_wei_z[i]
            diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
            p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(p_values, diagonal)
            
            for method in multitest_methods_names:
                output = multitest.multipletests(p_values, alpha=alpha, method=method)
                reject = output[0]
                pval_corrected = output[1]
        
                if True in reject:
                    string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                    list_strings.append(string)
                    print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
                    correlation_matrices_wei_z = correl_matrices_wei_z[i]
                    diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                    p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                    
                    #get locations of the significant results to get labels
                    a1,a2 = np.where(p_vals_matrix <= 0.05)
                    sign_labels1 = []
                    sign_labels2 = []
                    atlas_lab = atlas_labels[i,0]
                    for k in range(len(a1)):
                        sign_labels1.append(atlas_lab[a1[k]])
                        sign_labels2.append(atlas_lab[a2[k]])
                    
                    string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                    list_strings.append(string)
                    print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
                    string = f'The connectivity strength differs significantly between meditation and control groups post'
                    list_strings.append(string)
                    print(f'The connectivity strength differs significantly between meditation and control groups post')
                    for k in range(int(len(sign_labels1))):
                        string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                        list_strings.append(string)
                        print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
                else:
                    string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                    list_strings.append(string)
                    print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
            
        else:
            print('No p-values file')
        
with open(f'{out_dir}static/permut_gpu_post_meditation-control_matrices_wei.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item)    

#%%one sample permutation test for sport group permute core
n_permutations = 10000
df = len(subs_sport) + len(subs_sport) - 2
list_strings = []

for i, (j, atlas) in enumerate(nback):
    print(f'atlas {atlas}')
    for con in range(len(cond)):  
        print(f'condition {con}')
        
        corr_matr_s0 = correl_matrices_wei_z[i]
        corr_matr_s0 = corr_matr_s0[idx_subs_sport,0,con,:,:]
                
        corr_matr_s1 = correl_matrices_wei_z[i]
        corr_matr_s1 = corr_matr_s1[idx_subs_sport,1,con,:,:]
        
        #take only lower triangle
        corr_vec_s0 = []
        corr_vec_s1 = []
        for k in range(n_subs_sport):
            matrix = corr_matr_s0[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s0.append(corr_vec)
            matrix = corr_matr_s1[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s1.append(corr_vec)
        corr_vec_s0 = np.array(corr_vec_s0)
        corr_vec_s1 = np.array(corr_vec_s1)

        #compare the actual distribution with the t-distribution
        data = np.concatenate((corr_vec_s0, corr_vec_s1))

        p_values = []
        T0 = []
        H0 = []
        for k in range(corr_vec_s0.shape[1]):    
            p, t, distr = one_sample(corr_vec_s0[:,k], corr_vec_s1[:,k], stat='t', reps=n_permutations, alternative='two-sided', keep_dist=True, seed=55)
            p_values.append(p)
            T0.append(t)
            H0.append(distr)
            
        p_values = np.array(p_values)
        T0 = np.array(T0)
        H0 = np.array(H0)
        
        np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_p_values_permutecore.npy', p_values)
        np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_T0_permutecore.npy', T0)
        np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_H0_permutecore.npy', H0)

        n, bins, patches = plt.hist(H0[0], 25, histtype='bar', density=True)
        plt.title('Permutation Null Distribution')
        plt.axvline(x=T0[0], color='red')
        x = np.linspace(stats.t.ppf(0.0001, df), stats.t.ppf(0.9999, df), 100)
        plt.plot(x, stats.t.pdf(x, df), lw=2, alpha=0.6)
        plt.show()
        
        p_values = np.array(p_values)
        diagonal = np.ones(len(corr_matr_s0[0,:,0]))
        p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(p_values, diagonal)
        
        for method in multitest_methods_names:
            output = multitest.multipletests(p_values, alpha=alpha, method=method)
            reject = output[0]
            pval_corrected = output[1]
    
            if True in reject:
                string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                list_strings.append(string)
                print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
                correlation_matrices_wei_z = correl_matrices_wei_z[i]
                diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                
                #get locations of the significant results to get labels
                a1,a2 = np.where(p_vals_matrix <= 0.05)
                sign_labels1 = []
                sign_labels2 = []
                atlas_lab = atlas_labels[i,0]
                for k in range(len(a1)):
                    sign_labels1.append(atlas_lab[a1[k]])
                    sign_labels2.append(atlas_lab[a2[k]])
                
                string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                list_strings.append(string)
                print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
                string = f'The connectivity strength differs significantly between meditation and control groups post'
                list_strings.append(string)
                print(f'The connectivity strength differs significantly between meditation and control groups post')
                for k in range(int(len(sign_labels1))):
                    string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                    list_strings.append(string)
                    print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
            else:
                string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                list_strings.append(string)
                print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
                
with open(f'{out_dir}static/permut_permute-core_pre-post_sport_matrices_wei.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item) 
        
#%%one sample permutation test for med group permute core
n_permutations = 10000
df = len(subs_med) + len(subs_med) - 2
list_strings = []

for i, (j, atlas) in enumerate(nback):
    print(f'atlas {atlas}')
    for con in range(len(cond)):  
        print(f'condition {con}')
        
        corr_matr_s0 = correl_matrices_wei_z[i]
        corr_matr_s0 = corr_matr_s0[idx_subs_med,0,con,:,:]
                
        corr_matr_s1 = correl_matrices_wei_z[i]
        corr_matr_s1 = corr_matr_s1[idx_subs_med,1,con,:,:]
        
        #take only lower triangle
        corr_vec_s0 = []
        corr_vec_s1 = []
        for k in range(n_subs_med):
            matrix = corr_matr_s0[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s0.append(corr_vec)
            matrix = corr_matr_s1[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s1.append(corr_vec)
        corr_vec_s0 = np.array(corr_vec_s0)
        corr_vec_s1 = np.array(corr_vec_s1)

        #compare the actual distribution with the t-distribution
        data = np.concatenate((corr_vec_s0, corr_vec_s1))

        p_values = []
        T0 = []
        H0 = []
        for k in range(corr_vec_s0.shape[1]):    
            p, t, distr = one_sample(corr_vec_s0[:,k], corr_vec_s1[:,k], stat='t', reps=n_permutations, alternative='two-sided', keep_dist=True, seed=55)
            p_values.append(p)
            T0.append(t)
            H0.append(distr)
            
        p_values = np.array(p_values)
        T0 = np.array(T0)
        H0 = np.array(H0)
        
        np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_p_values_permutecore.npy', p_values)
        np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_T0_permutecore.npy', T0)
        np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_H0_permutecore.npy', H0)

        n, bins, patches = plt.hist(H0[0], 25, histtype='bar', density=True)
        plt.title('Permutation Null Distribution')
        plt.axvline(x=T0[0], color='red')
        x = np.linspace(stats.t.ppf(0.0001, df), stats.t.ppf(0.9999, df), 100)
        plt.plot(x, stats.t.pdf(x, df), lw=2, alpha=0.6)
        plt.show()
        
        p_values = np.array(p_values)
        diagonal = np.ones(len(corr_matr_s0[0,:,0]))
        p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(p_values, diagonal)
        
        for method in multitest_methods_names:
            output = multitest.multipletests(p_values, alpha=alpha, method=method)
            reject = output[0]
            pval_corrected = output[1]
    
            if True in reject:
                string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                list_strings.append(string)
                print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
                correlation_matrices_wei_z = correl_matrices_wei_z[i]
                diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                
                #get locations of the significant results to get labels
                a1,a2 = np.where(p_vals_matrix <= 0.05)
                sign_labels1 = []
                sign_labels2 = []
                atlas_lab = atlas_labels[i,0]
                for k in range(len(a1)):
                    sign_labels1.append(atlas_lab[a1[k]])
                    sign_labels2.append(atlas_lab[a2[k]])
                
                string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                list_strings.append(string)
                print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
                string = f'The connectivity strength differs significantly between meditation and control groups post'
                list_strings.append(string)
                print(f'The connectivity strength differs significantly between meditation and control groups post')
                for k in range(int(len(sign_labels1))):
                    string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                    list_strings.append(string)
                    print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
            else:
                string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                list_strings.append(string)
                print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
                
with open(f'{out_dir}static/permut_permute-core_pre-post_sport_matrices_wei.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item) 
#%%
n_permutations = 10000
df = len(subs_sport) + len(subs_sport) - 2

for i, (j, atlas) in enumerate(nback):
    print(f'atlas {atlas}')
    for con in range(len(cond)):  
        print(f'condition {con}')
        
        corr_matr_s0 = correl_matrices_wei_z[i]
        corr_matr_s0 = corr_matr_s0[idx_subs_sport,0,con,:,:]
                
        corr_matr_s1 = correl_matrices_wei_z[i]
        corr_matr_s1 = corr_matr_s1[idx_subs_sport,1,con,:,:]
        
        #take only lower triangle
        corr_vec_s0 = []
        corr_vec_s1 = []
        for k in range(n_subs_sport):
            matrix = corr_matr_s0[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s0.append(corr_vec)
            matrix = corr_matr_s1[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s1.append(corr_vec)
        corr_vec_s0 = np.array(corr_vec_s0)
        corr_vec_s1 = np.array(corr_vec_s1)

        #compare the actual distribution with the t-distribution
        data = np.concatenate((corr_vec_s0, corr_vec_s1))

 
        p_values, t, distr = one_sample(data, stat='t', reps=n_permutations, alternative='two-sided', keep_dist=True, seed=55)
        
        #np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_p_values_permutecore.npy', p_values)
        #np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_T0_permutecore.npy', T0)
        #np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_H0_permutecore.npy', H0)

        n, bins, patches = plt.hist(H0, 25, histtype='bar', density=True)
        plt.title('Permutation Null Distribution')
        plt.axvline(x=T0[0], color='red')
        x = np.linspace(stats.t.ppf(0.0001, df), stats.t.ppf(0.9999, df), 100)
        plt.plot(x, stats.t.pdf(x, df), lw=2, alpha=0.6)
        plt.show()
        
        p_values = np.array(p_values)
        diagonal = np.ones(len(corr_matr_s0[0,:,0]))
        p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(p_values, diagonal)
        
        minimum = np.min(p_values)
        maximum = np.max(p_values)
        plotting.plot_matrix(p_vals_matrix, colorbar=True, vmin=minimum, vmax=maximum,
                      title=f'{atlas} scondition-{con} p_vals_matrix')
                      
        #print significant results
        sign_array = np.zeros((p_vals_matrix.shape))
        for k in range(p_vals_matrix.shape[0]):
            for l in range(p_vals_matrix.shape[0]):
                if p_vals_matrix[k,l] <= 0.05:
                    sign_array[k,l] = 1
                else:
                    sign_array[k,l] = 0
        
        plotting.plot_matrix(sign_array, colorbar=True, vmin=0, vmax=1,
                      title=f'{nback[i,1]} ses-{ses} condition-{con} p_vals_matrix')
        
        #get locations of the significant results to get labels
        a1,a2 = np.where(p_vals_matrix <= 0.05)
        sign_labels1 = []
        sign_labels2 = []
        for k in range(len(a1)):
            sign_labels1.append(ho_labels[a1[k]])
            sign_labels2.append(ho_labels[a2[k]])
        
        print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
        print('The connectivity strength differs significantly between pre and post for sport group')
        for k in range(int(len(sign_labels1))):
            print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
#%%one sample permutation test for sport group permute core

from netneurotools import stats as nnstats
n_permutations = 10000
df = len(subs_sport) + len(subs_sport) - 2
list_strings = []

for i, (j, atlas) in enumerate(nback):
    print(f'atlas {atlas}')
    for con in range(len(cond)):  
        print(f'condition {con}')
        
        corr_matr_s0 = correl_matrices_wei_z[i]
        corr_matr_s0 = corr_matr_s0[idx_subs_sport,0,con,:,:]
                
        corr_matr_s1 = correl_matrices_wei_z[i]
        corr_matr_s1 = corr_matr_s1[idx_subs_sport,1,con,:,:]
        
        #take only lower triangle
        corr_vec_s0 = []
        corr_vec_s1 = []
        for k in range(n_subs_sport):
            matrix = corr_matr_s0[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s0.append(corr_vec)
            matrix = corr_matr_s1[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s1.append(corr_vec)
        corr_vec_s0 = np.array(corr_vec_s0)
        corr_vec_s1 = np.array(corr_vec_s1)

        #compare the actual distribution with the t-distribution
        #data = np.concatenate((corr_vec_s0, corr_vec_s1))
           
        stat, p_values = nnstats.permtest_rel(corr_vec_s0, corr_vec_s1, n_perm=n_permutations, seed=55)
            
        #p_values = np.array(p_values)
        #T0 = np.array(T0)
        #H0 = np.array(H0)
        
        #np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_p_values_permutecore.npy', p_values)
        #np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_T0_permutecore.npy', T0)
        #np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_H0_permutecore.npy', H0)

        # n, bins, patches = plt.hist(H0, 25, histtype='bar', density=True)
        # plt.title('Permutation Null Distribution')
        # plt.axvline(x=T0[0], color='red')
        # x = np.linspace(stats.t.ppf(0.0001, df), stats.t.ppf(0.9999, df), 100)
        # plt.plot(x, stats.t.pdf(x, df), lw=2, alpha=0.6)
        # plt.show()
        
        p_values = np.array(p_values)
        diagonal = np.ones(len(corr_matr_s0[0,:,0]))
        p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(p_values, diagonal)
        
        minimum = np.min(p_values)
        maximum = np.max(p_values)
        #plotting.plot_matrix(p_vals_matrix, colorbar=True, vmin=minimum, vmax=maximum,
        #              title=f'{atlas} scondition-{con} p_vals_matrix')
                      
        #print significant results
        sign_array = np.zeros((p_vals_matrix.shape))
        for k in range(p_vals_matrix.shape[0]):
            for l in range(p_vals_matrix.shape[0]):
                if p_vals_matrix[k,l] <= 0.05:
                    sign_array[k,l] = 1
                else:
                    sign_array[k,l] = 0
        
        #plotting.plot_matrix(sign_array, colorbar=True, vmin=0, vmax=1,
        #              title=f'{nback[atlas,1]} ses-{ses} condition-{con} p_vals_matrix')
        
        #get locations of the significant results to get labels
        a1,a2 = np.where(p_vals_matrix <= 0.05)
        sign_labels1 = []
        sign_labels2 = []
        atlas_lab = atlas_labels[i,0]
        for k in range(len(a1)):
            sign_labels1.append(atlas_lab[a1[k]])
            sign_labels2.append(atlas_lab[a2[k]])
        
        print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
        print('The connectivity strength differs significantly between pre and post for sport group')
        for k in range(int(len(sign_labels1)/2)):
            print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')

        for method in multitest_methods_names:
            output = multitest.multipletests(p_values, alpha=alpha, method=method)
            reject = output[0]
            pval_corrected = output[1]
    
            if True in reject:
                string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                list_strings.append(string)
                print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
                correlation_matrices_wei_z = correl_matrices_wei_z[i]
                diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                
                #get locations of the significant results to get labels
                a1,a2 = np.where(p_vals_matrix <= 0.05)
                sign_labels1 = []
                sign_labels2 = []
                atlas_lab = atlas_labels[i,0]
                for k in range(len(a1)):
                    sign_labels1.append(atlas_lab[a1[k]])
                    sign_labels2.append(atlas_lab[a2[k]])
                
                string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                list_strings.append(string)
                print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
                string = f'The connectivity strength differs significantly between pre and post for {subs_labels_grouped[group]} group'
                list_strings.append(string)
                print(f'The connectivity strength differs significantly between pre and post for {subs_labels_grouped[group]} group')
                for k in range(int(len(sign_labels1))):
                    string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                    list_strings.append(string)
                    print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
            else:
                string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                list_strings.append(string)
                print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")


#%%one sample permutation test for sport group permute core

from netneurotools import stats as nnstats
n_permutations = 10000
df = len(subs_sport) + len(subs_sport) - 2
list_strings = []

for i, (j, atlas) in enumerate(nback):
    print(f'atlas {atlas}')
    for con in range(len(cond)):  
        print(f'condition {con}')
        
        corr_matr_s0 = correl_matrices_wei_z[i]
        corr_matr_s0 = corr_matr_s0[idx_subs_sport,0,con,:,:]
                
        corr_matr_s1 = correl_matrices_wei_z[i]
        corr_matr_s1 = corr_matr_s1[idx_subs_sport,1,con,:,:]
        
        #take only lower triangle
        corr_vec_s0 = []
        corr_vec_s1 = []
        for k in range(n_subs_sport):
            matrix = corr_matr_s0[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s0.append(corr_vec)
            matrix = corr_matr_s1[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s1.append(corr_vec)
        corr_vec_s0 = np.array(corr_vec_s0)
        corr_vec_s1 = np.array(corr_vec_s1)

        #compare the actual distribution with the t-distribution
        data = np.concatenate((corr_vec_s0, corr_vec_s1))
           
        p_values, stat = one_sample(data, reps=n_permutations,
                                    alternative="two-sided", seed=55)
            
        #p_values = np.array(p_values)
        #T0 = np.array(T0)
        #H0 = np.array(H0)
        
        #np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_p_values_permutecore.npy', p_values)
        #np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_T0_permutecore.npy', T0)
        #np.save(f'{out_dir}static/{atlas}_con-{con}_static_correl_matrices_wei_z_H0_permutecore.npy', H0)

        # n, bins, patches = plt.hist(H0, 25, histtype='bar', density=True)
        # plt.title('Permutation Null Distribution')
        # plt.axvline(x=T0[0], color='red')
        # x = np.linspace(stats.t.ppf(0.0001, df), stats.t.ppf(0.9999, df), 100)
        # plt.plot(x, stats.t.pdf(x, df), lw=2, alpha=0.6)
        # plt.show()
        
        p_values = np.array(p_values)
        diagonal = np.ones(len(corr_matr_s0[0,:,0]))
        p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(p_values, diagonal)
        
        minimum = np.min(p_values)
        maximum = np.max(p_values)
        #plotting.plot_matrix(p_vals_matrix, colorbar=True, vmin=minimum, vmax=maximum,
        #              title=f'{atlas} scondition-{con} p_vals_matrix')
                      
        #print significant results
        sign_array = np.zeros((p_vals_matrix.shape))
        for k in range(p_vals_matrix.shape[0]):
            for l in range(p_vals_matrix.shape[0]):
                if p_vals_matrix[k,l] <= 0.05:
                    sign_array[k,l] = 1
                else:
                    sign_array[k,l] = 0
        
        #plotting.plot_matrix(sign_array, colorbar=True, vmin=0, vmax=1,
        #              title=f'{nback[atlas,1]} ses-{ses} condition-{con} p_vals_matrix')
        
        #get locations of the significant results to get labels
        a1,a2 = np.where(p_vals_matrix <= 0.05)
        sign_labels1 = []
        sign_labels2 = []
        atlas_lab = atlas_labels[i,0]
        for k in range(len(a1)):
            sign_labels1.append(atlas_lab[a1[k]])
            sign_labels2.append(atlas_lab[a2[k]])
        
        print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
        print('The connectivity strength differs significantly between pre and post for sport group')
        for k in range(int(len(sign_labels1)/2)):
            print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')

        for method in multitest_methods_names:
            output = multitest.multipletests(p_values, alpha=alpha, method=method)
            reject = output[0]
            pval_corrected = output[1]
    
            if True in reject:
                string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                list_strings.append(string)
                print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
                correlation_matrices_wei_z = correl_matrices_wei_z[i]
                diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                
                #get locations of the significant results to get labels
                a1,a2 = np.where(p_vals_matrix <= 0.05)
                sign_labels1 = []
                sign_labels2 = []
                atlas_lab = atlas_labels[i,0]
                for k in range(len(a1)):
                    sign_labels1.append(atlas_lab[a1[k]])
                    sign_labels2.append(atlas_lab[a2[k]])
                
                string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                list_strings.append(string)
                print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
                string = f'The connectivity strength differs significantly between pre and post for {subs_labels_grouped[group]} group'
                list_strings.append(string)
                print(f'The connectivity strength differs significantly between pre and post for {subs_labels_grouped[group]} group')
                for k in range(int(len(sign_labels1))):
                    string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                    list_strings.append(string)
                    print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
            else:
                string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                list_strings.append(string)
                print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")

#%%

fig, ax = plt.subplots()
im = ax.imshow(p_vals_matrix)

# We want to show all ticks...
ax.set_xticks(np.arange(len(ho_labels)))
ax.set_yticks(np.arange(len(ho_labels)))
# ... and label them with the respective list entries
ax.set_xticklabels(ho_labels)
ax.set_yticklabels(ho_labels)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

for i in range(len(ho_labels)):
    for j in range(len(ho_labels)):
        text = ax.text(j, i, p_vals_matrix[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Significant results")
fig.tight_layout()
plt.show()































