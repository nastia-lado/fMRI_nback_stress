#Dynamic connectivity estimation
#Last edited: 10-10-2020

#%%Step 0: Loading libraries
import sys
sys.path.append("..")
import os

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

from nilearn import datasets, plotting, input_data, signal  # for fetching atlas

from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure
from nistats.reporting import plot_design_matrix
from nistats.design_matrix import make_first_level_design_matrix
from sklearn.covariance import EmpiricalCovariance
import nibabel as nib

import scipy.io as sio

import seaborn as sns
sns.reset_orig()

from fctools import denoise, stats

#%%Step 1: Loading data
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/func_connect/03-correlation_matrices/'

nback_power = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_power_denoised_acompcor_no_smooth.npy', allow_pickle=True)
nback_schaefer = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_schaefer_denoised_acompcor_no_smooth.npy', allow_pickle=True)

nback = np.asarray([
                   [nback_power, 'nback_power'], 
                   [nback_schaefer, 'nback_schaefer']])

t_r = 3

#%%Step 2: Design specification
#t_r = 2
#n_scans = 340

#onsets_dir = '../support/onsets_dualnback.csv'
#events = pd.read_csv(onsets_dir)
#frame_times = np.arange(n_scans) * t_r

#events = events[(events.trial_type == '1-back') | (events.trial_type == '2-back')].reset_index()
#events['trial_type'] = np.arange(20)

# Step 1
#box = make_first_level_design_matrix(frame_times, events, hrf_model = None)
#box = box.reset_index()

# Step 2
#box_hrf = make_first_level_design_matrix(frame_times, events, hrf_model = 'glover')
#box_hrf  = box_hrf.reset_index()
#plt.plot(box_hrf.iloc[:,6])
#plt.plot(box.iloc[:,6])

#%%Step 3: Weighted correlation - dynamic connectivity

suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

groups = pd.read_csv('/home/alado/datasets/RBH/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])
subs.remove('sub-16')
subs.remove('sub-17')
subs.remove('sub-20')
subs.remove('sub-21')
subs.remove('sub-22')
subs.remove('sub-23')
subs.remove('sub-24')
subs.remove('sub-50')
sess = ['ses-1', 'ses-3']

for p in range(nback.shape[0]): #range(rest.shape[0]):
    data = nback[p,0]

    #NB! we had to pad the timeseries array with zeros so that n_scan dimension matches for all subj
    #but the actual n_scans is different. need to account for that
    
    sub_n = len(data[:, 0, 0, 0])
    ses_n = len(data[0, :, 0, 0])
    cond = ['1', '2', '3', '4']
    rois_n = len(data[0, 0, 0, :])
    A = np.zeros((rois_n, rois_n))
    correlation_matrices_dyn_wei = np.zeros((sub_n, ses_n, len(cond), rois_n, rois_n))

    #for sub in range(n_sub):
    for a, sub in enumerate(subs):
        print(f'Calculating correlations: {sub}')
        #for ses in range(ses_n):                 
            #correlation_measure = ConnectivityMeasure(cov_estimator=EmpiricalCovariance(store_precision=True, assume_centered=False), kind = 'correlation', discard_diagonal=True)
        for b, ses in enumerate(sess):
            # Getting directory/file names
            sub_dir = f'{top_dir}/preprocessed/fmriprep/{sub}/{ses}/func/'
            sub_name = f'{sub}_{ses}_task-nback_run-1' 
            epi_preproc_path = f'{sub_dir}{sub_name}{suffix}'
            events_dir = f'{top_dir}/Nifti/{sub}/{ses}/func/'
                
            #Step 2: Design specification
        
            events_path = f'{events_dir}{sub_name}_events.tsv'
            events = pd.read_csv(events_path, delimiter='\t')
        
            #load it from csv
            subject_data = nib.load(epi_preproc_path)
            n_scans = subject_data.shape[-1]
            frame_times = np.arange(n_scans) * t_r
        
        
            # Step 1
            box = make_first_level_design_matrix(frame_times, events, hrf_model = None)
            box = box.reset_index()
        
            # Step 2
            box_hrf = make_first_level_design_matrix(frame_times, events, hrf_model = 'glover')
            box_hrf  = box_hrf.reset_index()
                
            plt.plot(box_hrf)


            for con in range(len(cond)):
                # Zeroing negative values
                rect_box_hrf = np.array([0 if elem < 0 else elem for elem in box_hrf[int(cond[con])]])
                # Concatenating nonzeros blocs
                rect_nnz = rect_box_hrf[np.nonzero(rect_box_hrf)]
                # Filtering        
                data_new = np.zeros((1,1, len(rect_box_hrf), rois_n))
                data_new = data[a, b, 0:len(rect_box_hrf), :]
                timeseries_dual = data_new[rect_box_hrf > 0, :]
                #original if all sess have same n_scans
                #timeseries_dual = data[sub, ses, rect_box_hrf > 0, :]
                # Calculating weighted correlation coefficient
                for i in range(rois_n):
                    for j in range(i):
                        if i == j:
                            continue
                        else:
                            A[i, j] = stats.corr_wei(timeseries_dual[:, i], timeseries_dual[:, j], rect_nnz)

                fc = A + A.T
                correlation_matrices_dyn_wei[a, b, con, :, :] = np.arctanh(fc)

    print(correlation_matrices_dyn_wei.shape)
    
    np.save(f'{out_dir}dynamic/LB_{nback[p,1]}_dynamic_correlation_matrices.npy', correlation_matrices_dyn_wei)
    sio.savemat(f'{out_dir}dynamic/LB_{nback[p,1]}_dynamic_correlation_matrices.mat', {'correlation_matrices_dyn_wei': correlation_matrices_dyn_wei})
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
