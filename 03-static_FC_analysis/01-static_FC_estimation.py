#Static connectivity
#Last edited: 30-10-2020

import sys
sys.path.append("..")

import os

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.reset_orig()

from nilearn import datasets, plotting, input_data, signal  # for fetching atlas
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.glm.first_level import make_first_level_design_matrix
from sklearn.covariance import EmpiricalCovariance
from nilearn.plotting import plot_roi
from nilearn.image.image import mean_img
from nilearn import masking
import nibabel as nib
from scipy import stats

#%% functions we need
def m_wei(x, w):
    """Weighted Mean""" 
    return np.sum(x * w) / np.sum(w)

def cov_wei(x, y, w):
    """Weighted Covariance"""
    return np.sum(w * (x - m_wei(x, w)) * (y - m_wei(y, w))) / np.sum(w)

def corr_wei(x, y, w):
    """Weighted Correlation"""
    return cov_wei(x, y, w) / np.sqrt(cov_wei(x, x, w) * cov_wei(y, y, w))

#Calculate seed-based func connect
def seed_correlation(timeseries, seed_timeseries):
    """Compute the correlation between a seed voxel vs. other voxels 
    Parameters
    ----------
    timeseries [2d array], n_stimuli x n_voxels 
    seed_timeseries, 2d array, n_stimuli x 1

    Return
    ----------    
    seed_corr [2d array], n_stimuli x 1
    seed_corr_fishZ [2d array], n_stimuli x 1
    """
    num_voxels = timeseries.shape[1]
    seed_corr = np.zeros((num_voxels, 1))

    for v in range(num_voxels):    
        seed_corr[v, 0] = stats.pearsonr(seed_timeseries.flatten(), timeseries[:, v])[0]
        #alternative
        #seed_corr[v, 0] = np.corrcoef(seed_timeseries.flatten(), timeseries[:, v])[0, 1]
           
    seed_corr_fishZ = np.arctanh(seed_corr)
    return seed_corr, seed_corr_fishZ

#%%Step 1: Loading data
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/func_connect/04-correlation_matrices/'
suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
denoising = '_denoised_acompcor_task_effects.nii.gz'

groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])
sess = ['ses-1', 'ses-3']
cond = ['1', '2', '3', '4', '5']
tasks = ['nback']
task = 'nback'
n_scans_ses_1_run_1 = pd.Series.tolist(groups['n_scans_ses-1_run-1'])
n_scans_ses_3_run_1 = pd.Series.tolist(groups['n_scans_ses-3_run-1'])
n_scans_ses_1_run_2 = pd.Series.tolist(groups['n_scans_ses-1_run-2'])
n_scans_ses_3_run_2 = pd.Series.tolist(groups['n_scans_ses-3_run-2'])
t_r = 3
n_subs  = len(subs)

#nback_power_1 = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_power_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)
#nback_craddock_1 = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_craddock_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)
nback_ho_cort_1 = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_harvard-oxford_cort_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)
nback_ho_sub_1 = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_harvard-oxford_sub_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)
nback_ho_1 = np.concatenate((nback_ho_cort_1, nback_ho_sub_1), axis=3)
nback_talairach_1 = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_talairach_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)

nback_1 = np.asarray([
                    #[nback_power_1, 'nback_power'], 
                    #[nback_craddock_1, 'nback_craddock'],
                    [nback_ho_cort_1, 'nback_harvard-oxford_cort'],
                    [nback_ho_sub_1, 'nback_harvard-oxford_sub'],
                    [nback_ho_1, 'nback_harvard-oxford'],
                    [nback_talairach_1, 'nback_talairach']
                    ])

#nback_power_2 = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_power_denoised_acompcor_no_smooth_run-2.npy', allow_pickle=True)
#nback_craddock_2 = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_craddock_denoised_acompcor_no_smooth_run-2.npy', allow_pickle=True)
nback_ho_cort_2 = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_harvard-oxford_cort_denoised_acompcor_no_smooth_run-2.npy', allow_pickle=True)
nback_ho_sub_2 = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_harvard-oxford_sub_denoised_acompcor_no_smooth_run-2.npy', allow_pickle=True)
nback_ho_2 = np.concatenate((nback_ho_cort_2, nback_ho_sub_2), axis=3)
nback_talairach_2 = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_talairach_denoised_acompcor_no_smooth_run-2.npy', allow_pickle=True)

nback_2 = np.asarray([
                    #[nback_power_2, 'nback_power'],
                    #[nback_craddock_2, 'nback_craddock'],
                    [nback_ho_cort_2, 'nback_harvard-oxford_cort'],
                    [nback_ho_sub_2, 'nback_harvard-oxford_sub'],
                    [nback_ho_2, 'nback_harvard-oxford'],
                    [nback_talairach_2, 'nback_talairach']
                    ])

#%%Step 2: Calculate correlation matrices weighted atlas-based 
#atlas:
#-power
#-Harvard-Oxford cortical atlases (Makris 2006)
#-Harvard-Oxford subcortical atlases (Makris 2006)
#-Talairach atlas
#-Craddock atlas

print('Calculate weighted atlas-based correlational matrices')
for p in range(nback_1.shape[0]):
    data_1 = nback_1[p,0]  
    data_2 = nback_2[p,0]  
    sub_n = len(data_1[:, 0, 0, 0])
    ses_n = len(data_1[0, :, 0, 0])
    rois_n = len(data_1[0, 0, 0, :])
    A = np.zeros((rois_n, rois_n))
    correlation_matrices_wei_z = np.zeros((sub_n, ses_n, len(cond), rois_n, rois_n))

    for a, sub in enumerate(subs):
        print(f'Calculating correlations: {sub}')
        for b, ses in enumerate(sess):
            sub_name = f'{sub}_{ses}_task-nback_run-1' 
            events_dir = f'{top_dir}/Nifti/{sub}/{ses}/func/'
            events_path = f'{events_dir}{sub_name}_events.tsv'
            events_1 = pd.read_csv(events_path, delimiter='\t')
            if ses == 'ses-1':
                n_scans_run_1 = n_scans_ses_1_run_1[a]
            else:
                n_scans_run_1 = n_scans_ses_3_run_1[a]     
            
            sub_name = f'{sub}_{ses}_task-nback_run-2' 
            events_dir = f'{top_dir}/Nifti/{sub}/{ses}/func/'
            events_path = f'{events_dir}{sub_name}_events.tsv'
            events_2 = pd.read_csv(events_path, delimiter='\t')
            if ses == 'ses-1':
                n_scans_run_2 = n_scans_ses_1_run_2[a]
            else:
                n_scans_run_2 = n_scans_ses_3_run_2[a]     
            
            #Design specification - can be a separate step is n_scans same for all subj
            #we need it to account for hemodynamic lag
            frame_times_1 = np.arange(n_scans_run_1) * t_r
            box_hrf_1 = make_first_level_design_matrix(frame_times_1, events_1, hrf_model = 'glover')
            box_hrf_1 = box_hrf_1.reset_index()
            
            #plt.plot(box_hrf)    
            #plotting.plot_design_matrix(box_hrf_1)
            #plt.show()
            
            frame_times_2 = np.arange(n_scans_run_2) * t_r
            box_hrf_2 = make_first_level_design_matrix(frame_times_2, events_2, hrf_model = 'glover')
            box_hrf_2 = box_hrf_2.reset_index()
            
            for con in range(len(cond)):
                #zeroing negative values
                rect_box_hrf_1 = np.array([0 if elem < 0 else elem for elem in box_hrf_1[int(cond[con])]])
                #plt.plot(rect_box_hrf_1)
                #plt.title(f'run-1 {con}')
                #plt.show()
                
                rect_box_hrf_2 = np.array([0 if elem < 0 else elem for elem in box_hrf_2[int(cond[con])]])
                #plt.plot(rect_box_hrf_2)
                #plt.title(f'run-2 {con}')    
                #plt.show()
                
                #concatenating nonzeros blocs
                rect_nnz_1 = rect_box_hrf_1[np.nonzero(rect_box_hrf_1)]
                
                #filtering           
                data_new_1 = np.zeros((1,1, len(rect_box_hrf_1), rois_n))
                data_new_1 = data_1[a, b, 0:len(rect_box_hrf_1), :]
                timeseries_1 = data_new_1[rect_box_hrf_1 > 0, :]
                #if all sess have same n_scans
                #timeseries = data[a, b, rect_box_hrf > 0, :]
            
                rect_nnz_2 = rect_box_hrf_2[np.nonzero(rect_box_hrf_2)]          
                data_new_2 = np.zeros((1,1, len(rect_box_hrf_2), rois_n))
                data_new_2 = data_2[a, b, 0:len(rect_box_hrf_2), :]
                timeseries_2 = data_new_2[rect_box_hrf_2 > 0, :]    
            
                #do concatenation here
                #run-1 nbackA, has 1 in column if it was first
                if ses == 'ses-1':
                    if groups.iloc[a]['ses-1_run-1'] == 1:
                        timeseries = np.concatenate((timeseries_1, timeseries_2))
                        rect_nnz = np.concatenate((rect_nnz_1, rect_nnz_2))
                    else:
                        timeseries = np.concatenate((timeseries_2, timeseries_1))
                        rect_nnz = np.concatenate((rect_nnz_2, rect_nnz_1))  
                else:
                    if groups.iloc[a]['ses-3_run-1'] == 1:
                        timeseries = np.concatenate((timeseries_1, timeseries_2))
                        rect_nnz = np.concatenate((rect_nnz_1, rect_nnz_2))
                    else:
                        timeseries = np.concatenate((timeseries_2, timeseries_1))
                        rect_nnz = np.concatenate((rect_nnz_2, rect_nnz_1))          
                
                #calculating weighted correlation coefficient
                for i in range(rois_n):
                    for j in range(i):
                        if i == j:
                            continue
                        else:
                            A[i, j] = corr_wei(timeseries[:, i], timeseries[:, j], rect_nnz)

                fc = A + A.T
                
                #Fisher's transformation = inverse hyperbolic tangent = arctanh
                correlation_matrices_wei_z[a, b, con, :, :] = np.arctanh(fc)
    print(correlation_matrices_wei_z.shape)
    np.save(f'{out_dir}static/LB_{nback_1[p,1]}_static_correlation_matrices_wei_z.npy', correlation_matrices_wei_z)

#%%Step 2: Calculate correlation matrices weighted atlas-based for run 1
print('Calculate weighted atlas-based correlational matrices run 1')
for p in range(nback_1.shape[0]):
    data_1 = nback_1[p,0]  
    data_2 = nback_2[p,0]  
    sub_n = len(data_1[:, 0, 0, 0])
    ses_n = len(data_1[0, :, 0, 0])
    rois_n = len(data_1[0, 0, 0, :])
    A = np.zeros((rois_n, rois_n))
    correlation_matrices_wei_z = np.zeros((sub_n, ses_n, len(cond), rois_n, rois_n))

    for a, sub in enumerate(subs):
        print(f'Calculating correlations: {sub}')
        for b, ses in enumerate(sess):
            sub_name = f'{sub}_{ses}_task-nback_run-1' 
            events_dir = f'{top_dir}/Nifti/{sub}/{ses}/func/'
            events_path = f'{events_dir}{sub_name}_events.tsv'
            events_1 = pd.read_csv(events_path, delimiter='\t')
            if ses == 'ses-1':
                n_scans_run_1 = n_scans_ses_1_run_1[a]
            else:
                n_scans_run_1 = n_scans_ses_3_run_1[a]     
            
            sub_name = f'{sub}_{ses}_task-nback_run-2' 
            events_dir = f'{top_dir}/Nifti/{sub}/{ses}/func/'
            events_path = f'{events_dir}{sub_name}_events.tsv'
            events_2 = pd.read_csv(events_path, delimiter='\t')
            if ses == 'ses-1':
                n_scans_run_2 = n_scans_ses_1_run_2[a]
            else:
                n_scans_run_2 = n_scans_ses_3_run_2[a]     
            
            #Design specification - can be a separate step is n_scans same for all subj
            #we need it to account for hemodynamic lag
            frame_times_1 = np.arange(n_scans_run_1) * t_r
            box_hrf_1 = make_first_level_design_matrix(frame_times_1, events_1, hrf_model = 'glover')
            box_hrf_1 = box_hrf_1.reset_index()
            
            #plt.plot(box_hrf)    
            #plotting.plot_design_matrix(box_hrf_1)
            #plt.show()
            
            frame_times_2 = np.arange(n_scans_run_2) * t_r
            box_hrf_2 = make_first_level_design_matrix(frame_times_2, events_2, hrf_model = 'glover')
            box_hrf_2 = box_hrf_2.reset_index()
            
            for con in range(len(cond)):
                #zeroing negative values
                rect_box_hrf_1 = np.array([0 if elem < 0 else elem for elem in box_hrf_1[int(cond[con])]])
                #plt.plot(rect_box_hrf_1)
                #plt.title(f'run-1 {con}')
                #plt.show()
                
                rect_box_hrf_2 = np.array([0 if elem < 0 else elem for elem in box_hrf_2[int(cond[con])]])
                #plt.plot(rect_box_hrf_2)
                #plt.title(f'run-2 {con}')    
                #plt.show()
                
                #concatenating nonzeros blocs
                rect_nnz_1 = rect_box_hrf_1[np.nonzero(rect_box_hrf_1)]
                
                #filtering           
                data_new_1 = np.zeros((1,1, len(rect_box_hrf_1), rois_n))
                data_new_1 = data_1[a, b, 0:len(rect_box_hrf_1), :]
                timeseries_1 = data_new_1[rect_box_hrf_1 > 0, :]
                #if all sess have same n_scans
                #timeseries = data[a, b, rect_box_hrf > 0, :]
            
                rect_nnz_2 = rect_box_hrf_2[np.nonzero(rect_box_hrf_2)]          
                data_new_2 = np.zeros((1,1, len(rect_box_hrf_2), rois_n))
                data_new_2 = data_2[a, b, 0:len(rect_box_hrf_2), :]
                timeseries_2 = data_new_2[rect_box_hrf_2 > 0, :]    
            
                #do concatenation here
                #run-1 nbackA, has 1 in column if it was first
                if ses == 'ses-1':
                    if groups.iloc[a]['ses-1_run-1'] == 1:
                        timeseries = np.concatenate((timeseries_1, timeseries_2))
                        rect_nnz = np.concatenate((rect_nnz_1, rect_nnz_2))
                    else:
                        timeseries = np.concatenate((timeseries_2, timeseries_1))
                        rect_nnz = np.concatenate((rect_nnz_2, rect_nnz_1))  
                else:
                    if groups.iloc[a]['ses-3_run-1'] == 1:
                        timeseries = np.concatenate((timeseries_1, timeseries_2))
                        rect_nnz = np.concatenate((rect_nnz_1, rect_nnz_2))
                    else:
                        timeseries = np.concatenate((timeseries_2, timeseries_1))
                        rect_nnz = np.concatenate((rect_nnz_2, rect_nnz_1))          
                
                #calculating weighted correlation coefficient
                for i in range(rois_n):
                    for j in range(i):
                        if i == j:
                            continue
                        else:
                            A[i, j] = corr_wei(timeseries_1[:, i], timeseries_1[:, j], rect_nnz_1)

                fc = A + A.T
                
                #Fisher's transformation = inverse hyperbolic tangent = arctanh
                correlation_matrices_wei_z[a, b, con, :, :] = np.arctanh(fc)
    print(correlation_matrices_wei_z.shape)
    np.save(f'{out_dir}static/LB_{nback_1[p,1]}_static_correlation_matrices_wei_z_run_1.npy', correlation_matrices_wei_z)

#%%Step 2: Calculate correlation matrices weighted atlas-based for run 2
print('Calculate weighted atlas-based correlational matrices run 2')
for p in range(nback_1.shape[0]):
    data_1 = nback_1[p,0]  
    data_2 = nback_2[p,0]  
    sub_n = len(data_1[:, 0, 0, 0])
    ses_n = len(data_1[0, :, 0, 0])
    rois_n = len(data_1[0, 0, 0, :])
    A = np.zeros((rois_n, rois_n))
    correlation_matrices_wei_z = np.zeros((sub_n, ses_n, len(cond), rois_n, rois_n))

    for a, sub in enumerate(subs):
        print(f'Calculating correlations: {sub}')
        for b, ses in enumerate(sess):
            sub_name = f'{sub}_{ses}_task-nback_run-1' 
            events_dir = f'{top_dir}/Nifti/{sub}/{ses}/func/'
            events_path = f'{events_dir}{sub_name}_events.tsv'
            events_1 = pd.read_csv(events_path, delimiter='\t')
            if ses == 'ses-1':
                n_scans_run_1 = n_scans_ses_1_run_1[a]
            else:
                n_scans_run_1 = n_scans_ses_3_run_1[a]     
            
            sub_name = f'{sub}_{ses}_task-nback_run-2' 
            events_dir = f'{top_dir}/Nifti/{sub}/{ses}/func/'
            events_path = f'{events_dir}{sub_name}_events.tsv'
            events_2 = pd.read_csv(events_path, delimiter='\t')
            if ses == 'ses-1':
                n_scans_run_2 = n_scans_ses_1_run_2[a]
            else:
                n_scans_run_2 = n_scans_ses_3_run_2[a]     
            
            #Design specification - can be a separate step is n_scans same for all subj
            #we need it to account for hemodynamic lag
            frame_times_1 = np.arange(n_scans_run_1) * t_r
            box_hrf_1 = make_first_level_design_matrix(frame_times_1, events_1, hrf_model = 'glover')
            box_hrf_1 = box_hrf_1.reset_index()
            
            #plt.plot(box_hrf)    
            #plotting.plot_design_matrix(box_hrf_1)
            #plt.show()
            
            frame_times_2 = np.arange(n_scans_run_2) * t_r
            box_hrf_2 = make_first_level_design_matrix(frame_times_2, events_2, hrf_model = 'glover')
            box_hrf_2 = box_hrf_2.reset_index()
            
            for con in range(len(cond)):
                #zeroing negative values
                rect_box_hrf_1 = np.array([0 if elem < 0 else elem for elem in box_hrf_1[int(cond[con])]])
                #plt.plot(rect_box_hrf_1)
                #plt.title(f'run-1 {con}')
                #plt.show()
                
                rect_box_hrf_2 = np.array([0 if elem < 0 else elem for elem in box_hrf_2[int(cond[con])]])
                #plt.plot(rect_box_hrf_2)
                #plt.title(f'run-2 {con}')    
                #plt.show()
                
                #concatenating nonzeros blocs
                rect_nnz_1 = rect_box_hrf_1[np.nonzero(rect_box_hrf_1)]
                
                #filtering           
                data_new_1 = np.zeros((1,1, len(rect_box_hrf_1), rois_n))
                data_new_1 = data_1[a, b, 0:len(rect_box_hrf_1), :]
                timeseries_1 = data_new_1[rect_box_hrf_1 > 0, :]
                #if all sess have same n_scans
                #timeseries = data[a, b, rect_box_hrf > 0, :]
            
                rect_nnz_2 = rect_box_hrf_2[np.nonzero(rect_box_hrf_2)]          
                data_new_2 = np.zeros((1,1, len(rect_box_hrf_2), rois_n))
                data_new_2 = data_2[a, b, 0:len(rect_box_hrf_2), :]
                timeseries_2 = data_new_2[rect_box_hrf_2 > 0, :]    
            
                #do concatenation here
                #run-1 nbackA, has 1 in column if it was first
                if ses == 'ses-1':
                    if groups.iloc[a]['ses-1_run-1'] == 1:
                        timeseries = np.concatenate((timeseries_1, timeseries_2))
                        rect_nnz = np.concatenate((rect_nnz_1, rect_nnz_2))
                    else:
                        timeseries = np.concatenate((timeseries_2, timeseries_1))
                        rect_nnz = np.concatenate((rect_nnz_2, rect_nnz_1))  
                else:
                    if groups.iloc[a]['ses-3_run-1'] == 1:
                        timeseries = np.concatenate((timeseries_1, timeseries_2))
                        rect_nnz = np.concatenate((rect_nnz_1, rect_nnz_2))
                    else:
                        timeseries = np.concatenate((timeseries_2, timeseries_1))
                        rect_nnz = np.concatenate((rect_nnz_2, rect_nnz_1))          
                
                #calculating weighted correlation coefficient
                for i in range(rois_n):
                    for j in range(i):
                        if i == j:
                            continue
                        else:
                            A[i, j] = corr_wei(timeseries_2[:, i], timeseries_2[:, j], rect_nnz_2)

                fc = A + A.T
                
                #Fisher's transformation = inverse hyperbolic tangent = arctanh
                correlation_matrices_wei_z[a, b, con, :, :] = np.arctanh(fc)
    print(correlation_matrices_wei_z.shape)
    np.save(f'{out_dir}static/LB_{nback_1[p,1]}_static_correlation_matrices_wei_z_run_2.npy', correlation_matrices_wei_z)

#%%Correlation matrices atlas-based non-wei for entire nback time without fix

print('Calculate correlational matrices non-weighted')
for p in range(nback_1.shape[0]):
    data_1 = nback_1[p,0]  
    data_2 = nback_2[p,0]     
    sub_n = len(data_1[:, 0, 0, 0])
    ses_n = len(data_1[0, :, 0, 0])
    rois_n = len(data_1[0, 0, 0, :])
    correlation_matrix = np.zeros((sub_n, ses_n, rois_n, rois_n))
    correlation_matrix_z = np.zeros((sub_n, ses_n, rois_n, rois_n))

    for a, sub in enumerate(subs):
        print(f'Calculating correlations: {sub}')
        for b, ses in enumerate(sess):
            sub_name = f'{sub}_{ses}_task-nback_run-1' 
            events_dir = f'{top_dir}/Nifti/{sub}/{ses}/func/'
            events_path = f'{events_dir}{sub_name}_events.tsv'
            events_1 = pd.read_csv(events_path, delimiter='\t')
            if ses == 'ses-1':
                n_scans_run_1 = n_scans_ses_1_run_1[a]
            else:
                n_scans_run_1 = n_scans_ses_3_run_1[a]     
            
            sub_name = f'{sub}_{ses}_task-nback_run-2' 
            events_dir = f'{top_dir}/Nifti/{sub}/{ses}/func/'
            events_path = f'{events_dir}{sub_name}_events.tsv'
            events_2 = pd.read_csv(events_path, delimiter='\t')
            if ses == 'ses-1':
                n_scans_run_2 = n_scans_ses_1_run_2[a]
            else:
                n_scans_run_2 = n_scans_ses_3_run_2[a]     
            
            frame_times_1 = np.arange(n_scans_run_1) * t_r
            box_hrf_1 = make_first_level_design_matrix(frame_times_1, events_1, hrf_model = 'glover')
            box_hrf_1 = box_hrf_1.reset_index()
            box_hrf_1 = box_hrf_1.drop(columns=['index', 'constant', 'drift_1', 'drift_2', 'drift_3', 'drift_4', 
                                    'drift_5', 'drift_6', 'drift_7', 'drift_8', 'drift_9', 'drift_10'])                
            box_hrf_1[box_hrf_1 < 0] = 0
            box_hrf_1[box_hrf_1 > 0] = 1
            sum_col= box_hrf_1[1]+box_hrf_1[2]+box_hrf_1[3]+box_hrf_1[4]
            box_hrf_1['0'] = sum_col
            box_hrf_1[box_hrf_1 > 0] = 1

            
            frame_times_2 = np.arange(n_scans_run_2) * t_r
            box_hrf_2 = make_first_level_design_matrix(frame_times_2, events_2, hrf_model = 'glover')
            box_hrf_2 = box_hrf_2.reset_index()
            box_hrf_2 = box_hrf_2.drop(columns=['index', 'constant', 'drift_1', 'drift_2', 'drift_3', 'drift_4', 
                                    'drift_5', 'drift_6', 'drift_7', 'drift_8', 'drift_9', 'drift_10']) 
            box_hrf_2[box_hrf_2 < 0] = 0
            box_hrf_2[box_hrf_2 > 0] = 1
            sum_col= box_hrf_2[1]+box_hrf_2[2]+box_hrf_2[3]+box_hrf_2[4]
            box_hrf_2['0'] = sum_col
            box_hrf_2[box_hrf_2 > 0] = 1            

            #zeroing negative values
            rect_box_hrf_1 = np.array(box_hrf_1['0'])
            rect_box_hrf_2 = np.array(box_hrf_2['0'])
            #concatenating nonzeros blocs
            rect_nnz_1 = rect_box_hrf_1[np.nonzero(rect_box_hrf_1)]
            rect_nnz_2 = rect_box_hrf_2[np.nonzero(rect_box_hrf_2)]
            #filtering           
            data_new_1 = np.zeros((1,1, len(rect_box_hrf_1), rois_n))
            data_new_1 = data_1[a, b, 0:len(rect_box_hrf_1), :]
            timeseries_1 = data_new_1[rect_box_hrf_1 > 0, :]
            
            data_new_2 = np.zeros((1,1, len(rect_box_hrf_2), rois_n))
            data_new_2 = data_2[a, b, 0:len(rect_box_hrf_2), :]
            timeseries_2 = data_new_2[rect_box_hrf_2 > 0, :]            
                        
            #do concatenation here
            #run-1 nbackA, has 1 in column if it was first
            if ses == 'ses-1':
                if groups.iloc[a]['ses-1_run-1'] == 1:
                    timeseries = np.concatenate((timeseries_1, timeseries_2))
                    rect_nnz = np.concatenate((rect_nnz_1, rect_nnz_2))
                else:
                    timeseries = np.concatenate((timeseries_2, timeseries_1))
                    rect_nnz = np.concatenate((rect_nnz_2, rect_nnz_1))  
            else:
                if groups.iloc[a]['ses-3_run-1'] == 1:
                    timeseries = np.concatenate((timeseries_1, timeseries_2))
                    rect_nnz = np.concatenate((rect_nnz_1, rect_nnz_2))
                else:
                    timeseries = np.concatenate((timeseries_2, timeseries_1))
                    rect_nnz = np.concatenate((rect_nnz_2, rect_nnz_1))   
            
            
            correlation_measure = ConnectivityMeasure(kind='correlation')
            correlation_matrix = correlation_measure.fit_transform([timeseries])[0]

            #Fisher's transformation = inverse hyperbolic tangent = arctanh
            correlation_matrix_z[a, b, :, :] = np.arctanh(correlation_matrix)
    
    print(correlation_matrix_z.shape)
    np.save(f'{out_dir}static/LB_{nback_1[p,1]}_static_correlation_matrices_z.npy', correlation_matrix_z)
    
#%%Correlation matrices seed-based voxel-wise non-wei
#masking data to extract timeseries.
#there is a possibility to specify mask_strategy={'background', 'epi', 'template'} in the NiftiMasker
#template strategy will result in the same mask for all subjects, but in this case it didn't look good
#another approach: use 
#nilearn.masking.compute_multi_epi_mask(imgs, lower_cutoff=0.2, upper_cutoff=0.85, connected=True, opening=2, threshold=0.5)

#%%Seed-based correlation
#create a mask for all subj/sess and ROI
# areas for seed-based corr (right): 
#-PFC (BA10) [(36, 52, 14)]
#-dlPFC [(30, 48, 22)]
#-postrerior parietal cortex (PCC) [(0, -53, 26)]
#-(dorsal) ACC [(6, 22, 30,)]
#-amygdala [(22, -2, -15)]
#-striatum [(22, 6, -2)]

imgs = []
for i, sub in enumerate(subs):
    for j, ses in enumerate(sess):
         sub_dir = f'{top_dir}/preprocessed/derivatives/fmriprep/{sub}/{ses}/func/'
         sub_name = f'{sub}_{ses}_task-{task}_run-1' 
         epi_preproc_path = f'{sub_dir}{sub_name}{suffix}'
         imgs.append(epi_preproc_path)
         
mask = masking.compute_multi_epi_mask(imgs, lower_cutoff=0.2, upper_cutoff=0.85, connected=True, opening=2, threshold=0.5)

#Create spherial ROI
# Prepare seeds
seeds = np.asarray([['PFC', [(36, 52, 14)]], 
                    ['DLPFC', [(30, 48, 22)]],
                    ['PCC', [(0, -53, 26)]],
                    ['dACC', [(6, 22, 30,)]],
                    ['Amygdala', [(22, -2. -15)]],
                    ['Striatum', [(22, 6, -2)]]])

#%%
#use here clean img from
#f'/home/alado/datasets/RBH/Lipsia/{sub}/{ses}/func/{sub}_{ses}_space-MNI152NLin2009cAsym_denoised.nii.gz'

for seed in range(seeds.shape[0]):
    masker_seed = input_data.NiftiSpheresMasker(
                    seeds[seed, 0],
                    radius=8, standardize=True, t_r=t_r,
                    memory='nilearn_cache', memory_level=1, verbose=0)
    
    sub_n = len(subs)
    ses_n = len(sess)
    vox_n = 55538
    seed_corr_all = np.zeros((sub_n, ses_n, 1, vox_n))
    seed_corr_fz_all = np.zeros((sub_n, ses_n, 1, vox_n))
    
    for i, sub in enumerate(subs):
        for j, ses in enumerate(sess):
            #Step 0: Load data
            sub_dir = f'{top_dir}/preprocessed/fmriprep/{sub}/{ses}/func/'
            sub_name = f'{sub}_{ses}_task-{task}_run-1'
            epi_preproc_path = f'{sub_dir}{sub_name}{suffix}'
            denoised_data_path = f'{top_dir}/func_connect/01-conf_clean_acompcor/{sub}/{sub_name}{denoising}'
            confounds_clean_path = f'{top_dir}/func_connect/01-conf_clean_acompcor/{sub}/{sub_name}_bold_confounds_clean_acompcor.csv'
            confounds_clean = pd.read_csv(confounds_clean_path, delimiter=',')
            
            #Step 1: Mask epi imgs to extract timeseries
            timeseries = nilearn.masking.apply_mask(epi_preproc_path, mask)
            
            #display: for visual inspection - no need here
            #mean_dataset = mean_img(epi_preproc_path)
            #nilearn.plotting.plot_epi(mean_dataset)
            #nilearn.plotting.plot_roi(mask, mean_dataset)
            
            #mean_func_img = mean_img(epi_preproc_path)
            #plot_roi(mask_img, mean_func_img, display_mode='y', cut_coords=4, title="Mask")
            
            #Step 2: Denoise
            timeseries = nilearn.signal.clean(timeseries,
                                    detrend=False,
                                    standardize=True,
                                    confounds=confounds_clean_path,
                                    low_pass=0.08, 
                                    high_pass=0.008, 
                                    t_r=t_r,
                                    )
            
            #Step 3: Mask the epi data and get a time series for the seed
            seed_timeseries = masker_seed.fit_transform(epi_preproc_path)
            #another way?
            # Extract time series from seed region
            #seed_timeseries = np.mean(denoised_data[pcc_labels], axis=0)
            
            #denoise seed_timeseries
            seed_timeseries = signal.clean(seed_timeseries, 
                            low_pass=0.25, 
                            high_pass=0.008, 
                            t_r=t_r, 
                            detrend=False, 
                            standardize=True,
                            confounds=confounds_clean_path
                            )
            
            #Step 4: CORRELATION
            seed_corr, seed_corr_fz = seed_correlation(timeseries, seed_timeseries)
            
            seed_corr_all[i,j,:,:] = seed_corr
            seed_corr_fz_all[i,j,:,:] = seed_corr_fz
            
    #Step 5: Save
    np.save(f'{out_dir}static/seed-based/seed_{seeds[seed,0]}_correlation_matrix_z.npy', seed_corr_fz)
     
#%%
#tranform the correlation array back to a Nifti image object and save
img_corr = masker.inverse_transform(seed_corr_fz.T)
# img_corr.to_filename('seed_rtstim.nii.gz')

#%%Plot seed correlations (correlation of the seed with every voxel)
threshold = .8
pcc_coords = [(0, -53, 26)]

# Nilearn
r_map_ar = plotting.plot_stat_map(
    img_corr, 
    threshold=threshold,
    cut_coords=pcc_coords[0],
)

# Add the seed
r_map_ar.add_markers(
    marker_coords=pcc_coords, 
    marker_color='g',
    marker_size=50
)

#%%
# Create a glass brain
plotting.plot_glass_brain(
    img_corr, 
    threshold=threshold,
    colorbar=True, 
    plot_abs=False,
    display_mode='lyrz', 
);







#%% Creating a seed from an atlas
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = atlas.maps

# This is where the atlas is saved.
print("Atlas path: " + atlas_filename + "\n\n")

# Plot the ROIs
plotting.plot_roi(atlas_filename);
print('Harvard-Oxford cortical atlas')

# Print the labels
# Label 0 (Background) refers to the brain image, not background connectivity

# Create a Pandas dataframe of the atlas data for easy inspection.
atlas_pd = pd.DataFrame(atlas)
print(atlas_pd['labels'])

#%%
# Create a masker object that we can use to select ROIs
masker_ho = NiftiLabelsMasker(labels_img=atlas_filename)
print(masker_ho.get_params())

denoised_data = 
# Apply our atlas to the Nifti object so we can pull out data from single parcels/ROIs
bold_ho = masker_ho.fit_transform(denoised_data)
print('shape: parcellated bold time courses: ', np.shape(bold_ho))

#data structure
print("Parcellated data shape (time points x num ROIs)")
print("All time points  ", bold_ho.shape)
#%%
# Pull out a single ROI corresponding to the posterior parahippocampal cortex
# Label #35 is the Parahippocampal Gyrus, posterior division. 
roi_id = 35
bold_ho_pPHG = np.array(bold_ho[:, roi_id])
bold_ho_pPHG = bold_ho_pPHG.reshape(bold_ho_pPHG.shape[0],-1)
print("Posterior PPC (region 35) rightward attention trials: ", bold_ho_pPHG.shape)

plt.figure(figsize=(14,4))
plt.plot(bold_ho_pPHG)
plt.ylabel('Evoked activity');
plt.xlabel('Timepoints');
sns.despine()

#%%correlate the whole brain time course with the seed

corr_pPHG, corr_fz_pPHG = seed_correlation(
    timeseries, bold_ho_pPHG
) 

# Print the range of correlations.
print("PHG correlation Fisher-z transformed: min = %.3f; max = %.3f" % (
    corr_fz_pPHG.min(), corr_fz_pPHG.max())
)

# Plot a histogram
plt.hist(corr_fz_pPHG)
plt.ylabel('Frequency');data
plt.xlabel('Fisher-z score');

#%%
# Map back to the whole brain image
img_corr_pPHG = masker.inverse_transform(corr_fz_pPHG.T)

threshold = .5 

# Find the cut coordinates of this ROI, using parcellation.
# This function takes the atlas path and the hemisphere and outputs all centers of the ROIs
roi_coords = plotting.find_parcellation_cut_coords(atlas_filename,label_hemisphere='left')

# Pull out the coordinate for this ROI
roi_coord = roi_coords[roi_id,:]

# Plot the correlation as a map on a standard brain. 
# For comparison, we also plot the position of the sphere we created ealier.
h2 = plotting.plot_stat_map(
    img_corr_pPHG, 
    threshold=threshold,
    cut_coords=roi_coord,
)

# Create a glass brain
plotting.plot_glass_brain(
    img_corr_pPHG, 
    threshold=threshold,
    colorbar=True, 
    display_mode='lyrz', 
    plot_abs=False
)

#%%compute connectivity across parcells = correlation across multiple brain regions
# Set up the connectivity object
correlation_measure = ConnectivityMeasure(kind='correlation')

# Calculate the correlation of each parcel with every other parcel

corr_mat_ho = correlation_measure.fit_transform([bold_ho])[0]

# Remove the diagonal for visualization (guaranteed to be 1.0)
np.fill_diagonal(corr_mat_ho, np.nan)

# Plot the correlation matrix
# The labels of the Harvard-Oxford Cortical Atlas that we are using 
# start with the background (0), hence we skip the first label
fig = plt.figure(figsize=(11,10))
plt.imshow(corr_mat_ho, interpolation='None', cmap='RdYlBu_r')
plt.yticks(range(len(atlas.labels)), atlas.labels[1:]);
plt.xticks(range(len(atlas.labels)), atlas.labels[1:], rotation=90);
plt.title('Parcellation correlation matrix')
plt.colorbar();

#%%# different way to plot connectivity - Nilearn's own plotting function
plotting.plot_matrix(
    corr_mat_ho, 
    cmap='RdYlBu_r', 
    figure=(11, 10), 
    labels=atlas.labels[1:], 
)

#%%Background connectivity
#Potential problem in analyzing functional connectivity during tasks.
#Consider brain regions A and B. Let's assume that the task activates both regions.
#If we were to examine correlations between the regions, they would have strong connectivity,
#but not because they are necessarily communicating or interacting in any way.
#Rather, they share the stimulus as a third variable.
#One solution is to regress out the stimulus-evoked responses from our signal and re-examine the correlations.
#If region A and B are still correlated, we are on more solid footing that they are functionally
#connected during the task. Insofar as this "background connectivity" differs between task conditions
#(e.g., attend left vs. right), we can conclude that the task is modulating the scaffold of noise
#correlations in the brain.
#see Functional Interactions as Big Data in the Human Brain. Nicholas B. Turk-Browne

#In background connectivity analysis, stimulus-driven activation is not the desired effect of interest,
#but potentially a confound. Thus, we need to remove "stimulus confounds" and run connectivity.
#We need to regress out the evoked activity and and other nuisance variables, and repeat the previous analyses.

#regress out "stimulus" confounds
#????? run GLM and calculate FC on residuals


#%%
# Get the seed data
bold_lPPA_mcg = masker_lPPA.fit_transform(epi_in_mcg)
bold_lPPA_r_mcg = bold_lPPA_mcg[right_stim_lag==1,:]
bold_lPPA_l_mcg = bold_lPPA_mcg[left_stim_lag==1,:]

# Get the whole brain data
boldWB_mcg = masker_wb.fit_transform(epi_in_mcg)
boldWB_r_mcg = boldWB_mcg[right_stim_lag==1,:] 
boldWB_l_mcg = boldWB_mcg[left_stim_lag==1,:] 

# plot the data
plt.figure(figsize=(14,4))
plt.plot(bold_lPPA_r_mcg)
plt.plot(bold_lPPA_l_mcg)
plt.legend(('Attend Right', 'Attend Left'));
plt.ylabel('BOLD signal, standardized')
plt.xlabel('TRs of right attention blocks')
plt.title('Background activity in seed region')
sns.despine()



#%%Classification
from nilearn import datasets

development_dataset = datasets.fetch_development_fmri(n_subjects=30)

msdl_data = datasets.fetch_atlas_msdl()
msdl_coords = msdl_data.region_coords
n_regions = len(msdl_coords)
print('MSDL has {0} ROIs, part of the following networks :\n{1}.'.format(
    n_regions, msdl_data.networks))

#%%Regions signal extraction
from nilearn import input_data

masker = input_data.NiftiMapsMasker(
    msdl_data.maps, resampling_target="data", t_r=2, detrend=True,
    low_pass=.1, high_pass=.01, memory='nilearn_cache', memory_level=1).fit()

children = []
pooled_subjects = []
groups = []  # child or adult
for func_file, confound_file, phenotypic in zip(
        development_dataset.func,
        development_dataset.confounds,
        development_dataset.phenotypic):
    time_series = masker.transform(func_file, confounds=confound_file)
    pooled_subjects.append(time_series)
    if phenotypic['Child_Adult'] == 'child':
        children.append(time_series)
    groups.append(phenotypic['Child_Adult'])

print('Data has {0} children.'.format(len(children)))

#%%

# Matrix plotting from Nilearn: nilearn.plotting.plot_matrix
import numpy as np
import matplotlib.pylab as plt
from nilearn import plotting

def plot_matrices(matrices, matrix_kind):
    n_matrices = len(matrices)
    fig = plt.figure(figsize=(n_matrices * 4, 4))
    for n_subject, matrix in enumerate(matrices):
        plt.subplot(1, n_matrices, n_subject + 1)
        matrix = matrix.copy()  # avoid side effects
        # Set diagonal to zero, for better visualization
        np.fill_diagonal(matrix, 0)
        vmax = np.max(np.abs(matrix))
        title = '{0}, subject {1}'.format(matrix_kind, n_subject)
        plotting.plot_matrix(matrix, vmin=-vmax, vmax=vmax, cmap='RdBu_r',
                             title=title, figure=fig, colorbar=False)
        
#%%

from nilearn import datasets

adhd_data = datasets.fetch_adhd(n_subjects=20)

#%%mdsl atlas
msdl_data = datasets.fetch_atlas_msdl()
msdl_coords = msdl_data.region_coords
n_regions = len(msdl_coords)
print('MSDL has {0} ROIs, part of the following networks :\n{1}.'.format(
    n_regions, msdl_data.networks))

#%%ROI signal extraction
from nilearn import input_data

masker = input_data.NiftiMapsMasker(
    msdl_data.maps, resampling_target="data", t_r=2.5, detrend=True,
    low_pass=.1, high_pass=.01, memory='nilearn_cache', memory_level=1)


adhd_subjects = []
pooled_subjects = []
site_names = []
adhd_labels = []  # 1 if ADHD, 0 if control
for func_file, confound_file, phenotypic in zip(
        adhd_data.func, adhd_data.confounds, adhd_data.phenotypic):
    time_series = masker.fit_transform(func_file, confounds=confound_file)
    pooled_subjects.append(time_series)
    is_adhd = phenotypic['adhd']
    if is_adhd:
        adhd_subjects.append(time_series)

    site_names.append(phenotypic['site'])
    adhd_labels.append(is_adhd)

print('Data has {0} ADHD subjects.'.format(len(adhd_subjects)))
#%%ROI to ROI correlation

from nilearn.connectome import ConnectivityMeasure

correlation_measure = ConnectivityMeasure(kind='correlation')

correlation_matrices = correlation_measure.fit_transform(adhd_subjects)

# All individual coefficients are stacked in a unique 2D matrix.
print('Correlations of ADHD patients are stacked in an array of shape {0}'
      .format(correlation_matrices.shape))

mean_correlation_matrix = correlation_matrices.mean(axis=0)
print('Mean correlation has shape {0}.'.format(mean_correlation_matrix.shape))

correlation_matrices.mean(axis=0).shape

plot_matrices(correlation_matrices[:4], 'correlation')
plotting.plot_connectome(mean_correlation_matrix, msdl_coords,
                         title='mean correlation over 13 ADHD subjects')
#%%Examine partial correlations
partial_correlation_measure = ConnectivityMeasure(kind='partial correlation')
partial_correlation_matrices = partial_correlation_measure.fit_transform(
    adhd_subjects)
plot_matrices(partial_correlation_matrices[:4], 'partial')
plotting.plot_connectome(
    partial_correlation_measure.mean_, msdl_coords,
    title='mean partial correlation over 13 ADHD subjects')


#%%Extract subjects variabilities around a robust group connectivity
#We can use both correlations and partial correlations to capture reproducible 
#connectivity patterns at the group-level and build a robust group connectivity matrix. This is done by the tangent kind.

tangent_measure = ConnectivityMeasure(kind='tangent')

tangent_matrices = tangent_measure.fit_transform(adhd_subjects)
#tangent_matrices model individual connectivities as perturbations of the group connectivity matrix
#tangent_measure.mean_. Keep in mind that these subjects-to-group variability matrices do not straight reflect
#individual brain connections. For instance negative coefficients can not be interpreted as anticorrelated regions.

plot_matrices(tangent_matrices[:4], 'tangent variability')
plotting.plot_connectome(
    tangent_measure.mean_, msdl_coords,
    title='mean tangent connectivity over 13 ADHD subjects')


#%%What kind of connectivity is most powerful for classification?
connectivity_biomarkers = {}
kinds = ['correlation', 'partial correlation', 'tangent']
for kind in kinds:
    conn_measure = ConnectivityMeasure(kind=kind, vectorize=True)
    connectivity_biomarkers[kind] = conn_measure.fit_transform(pooled_subjects)

# For each kind, all individual coefficients are stacked in a unique 2D matrix.
print('{0} correlation biomarkers for each subject.'.format(
    connectivity_biomarkers['correlation'].shape[1]))

#Note that we use the pooled groups. This is crucial for tangent kind, to get the displacements from
#a unique mean_ of all subjects.
#We stratify the dataset into homogeneous classes according to phenotypic and scan site.
#We then split the subjects into 3 folds with the same proportion of each class as in the whole cohort

from sklearn.cross_validation import StratifiedKFold

classes = ['{0}{1}'.format(site_name, adhd_label)
           for site_name, adhd_label in zip(site_names, adhd_labels)]
cv = StratifiedKFold(classes, n_folds=3)

from sklearn.svm import NuSVC
from sklearn.cross_validation import cross_val_score

mean_scores = []
for kind in kinds:
    svc = NuSVC(random_state=0, nu=.1)
    cv_scores = cross_val_score(svc, connectivity_biomarkers[kind],
                                y=adhd_labels, cv=cv, scoring='accuracy')
    mean_scores.append(cv_scores.mean())
    
plt.figure(figsize=(6, 4))
positions = np.arange(len(kinds)) * .1 + .1
plt.barh(positions, mean_scores, align='center', height=.05)
yticks = [kind.replace(' ', '\n') for kind in kinds]
plt.yticks(positions, yticks)
plt.xlabel('Classification accuracy')
plt.grid(True)
plt.tight_layout()

plt.show()

from sklearn.svm import NuSVC
from sklearn.cross_validation import cross_val_score

mean_scores = []
for kind in kinds:
    svc = NuSVC(random_state=0, nu=.1)
    cv_scores = cross_val_score(svc, connectivity_biomarkers[kind],
                                y=adhd_labels, cv=cv, scoring='accuracy')
    mean_scores.append(cv_scores.mean())
    
plt.figure(figsize=(6, 4))
positions = np.arange(len(kinds)) * .1 + .1
plt.barh(positions, mean_scores, align='center', height=.05)
yticks = [kind.replace(' ', '\n') for kind in kinds]
plt.yticks(positions, yticks)
plt.xlabel('Classification accuracy')
plt.grid(True)
plt.tight_layout()

plt.show()

from sklearn.svm import LinearSVC
from sklearn.cross_validation import cross_val_score

mean_scores = []
for kind in kinds:
    svc = LinearSVC(random_state=0)
    cv_scores = cross_val_score(svc, connectivity_biomarkers[kind],
                                y=adhd_labels, cv=cv, scoring='accuracy')
    mean_scores.append(cv_scores.mean())



plt.figure(figsize=(6, 4))
positions = np.arange(len(kinds)) * .1 + .1
plt.barh(positions, mean_scores, align='center', height=.05)
yticks = [kind.replace(' ', '\n') for kind in kinds]
plt.yticks(positions, yticks)
plt.xlabel('Classification accuracy')
plt.grid(True)
plt.tight_layout()

plt.show()








