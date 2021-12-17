#timeseries extraction of data from working memory and stress experiment.

#Last edited: 30-09-2020

#%%Step 0: Load libraries

import sys
sys.path.append("..")

import pandas as pd
import nibabel as nib
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from nilearn import datasets
from nilearn.input_data import NiftiLabelsMasker

#%%Step 1: Prepare path to files

# Setting directories for input and output files
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/func_connect/02-extracted_timeseries/'
suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
denoising = '_denoised_acompcor_task_effects.nii.gz'
denoising_smooth = '_denoised_acompcor_task_effects_smooth_fwhm3.nii.gz'

# Selecting subjects who finished the study
groups = pd.read_csv('/home/alado/datasets/RBH/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])

# Setting sessions and task names
sess = ['ses-1', 'ses-3']
runs = ['run-1', 'run-2']
tasks = ['nback']
t_r = 3

print(f'Sample size: {len(subs)}')

#%%Step 2: Creating parcellations

#Power ROIs coordinates
#power = datasets.fetch_coords_power_2011()
#power_coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T
#power = input_data.NiftiSpheresMasker(seeds = power_coords, radius = 5)

#Craddock
#craddock = datasets.fetch_atlas_craddock_2012('scorr_mean')
#craddock_filename = craddock.scorr_mean
#craddock_masker = input_data.NiftiMapsMasker(maps_img=craddock_filename, standardize=True, memory='nilearn_cache', verbose=5)

#Harvard-Oxford 2 atlases (49+22=71)
ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
ho_cort_filename = ho_cort.maps
ho_sub_filename = ho_sub.maps
ho_cort_masker = NiftiLabelsMasker(labels_img=ho_cort_filename, 
                                   standardize=True,
                                   resampling_target = 'labels',
                                   memory='nilearn_cache', verbose=0)
ho_sub_masker = NiftiLabelsMasker(labels_img=ho_sub_filename, standardize=True, 
                                  resampling_target = 'labels',
                                  memory='nilearn_cache', verbose=0)

#Talairach (72)
talairach = datasets.fetch_atlas_talairach('ba')
talairach_filename = talairach.maps
talairach_masker = NiftiLabelsMasker(labels_img=talairach_filename, 
                                     standardize=True, 
                                     resampling_target = 'labels',
                                     memory='nilearn_cache', verbose=0)

#AAL atlas (116)
aal = datasets.fetch_atlas_aal()
aal_filename = aal.maps
aal_masker = NiftiLabelsMasker(labels_img=aal_filename, standardize=True, 
                               resampling_target = 'labels',
                               memory='nilearn_cache', verbose=0)

parcellations = np.asarray([
                            #[power, 'power', 264], 
                            #[craddock_masker, 'craddock', 43],
                            [ho_cort_masker, 'harvard-oxford_cort', 48],
                            [ho_sub_masker, 'harvard-oxford_sub', 21],
                            [talairach_masker, 'talairach', 71],
                            [aal_masker, 'aal', 116]
                            ])
#%%Step 3: Extract timeseries for each ROI
# Iterating over parcellations
for p in range(parcellations.shape[0]):
    n_roi = parcellations[p,2]
    
    # Iterating over tasks   
    for task in tasks:
        
        for run in runs:
            print(run)
            
            print(f'Extracting timeseries: {parcellations[p,1]} parcellation, {task}')
            #this number i obtain at at a previous step
            n_scans = 220 #max of n_scans
            timeseries_all = np.zeros((len(subs), len(sess), n_scans, n_roi))
            timeseries_smooth_all = np.zeros((len(subs), len(sess), n_scans, n_roi))
            #timeseries_all = []
            
            # Iterating over subjects, sessions
            for i, sub in enumerate(subs):
                print(f'Extracting timeseries {sub}')
                for j, ses in enumerate(sess):
                    
                    # Getting directory/file names
                    sub_dir = f'{top_dir}/preprocessed/derivatives/fmriprep/{sub}/{ses}/func/'
                    sub_name = f'{sub}_{ses}_task-{task}_{run}' 
                    denoised_data_path = f'{top_dir}/func_connect/01-conf_clean_acompcor/{sub}/{sub_name}{denoising}'
                    denoised_smooth_data_path = f'{top_dir}/func_connect/01-conf_clean_acompcor/{sub}/{sub_name}{denoising_smooth}'
                    epi_preproc_path = f'{sub_dir}{sub_name}{suffix}'
                    subject_data = nib.load(epi_preproc_path)
                    epi_preproc_path = f'{sub_dir}{sub_name}{suffix}'
                    events_path = f'{sub_dir}{sub_name}_events.tsv'
                    events = pd.read_csv(events_path, delimiter='\t')
    
                    #if not os.path.exists(denoised_data):
                    #    print(f'{sub}{ses}{task} does not exist')
                    #else:
                    # Extracting timeseries for specified atlas)
                    timeseries = parcellations[p,0].fit_transform(denoised_data_path, confounds=None)
                    timeseries_smooth = parcellations[p,0].fit_transform(denoised_smooth_data_path, confounds=None)
                    
                    #here because the n_scans is different for every subj i need to pad the array w zeros
                    #to make it 216xn_roi
                    #zeros at the end
                    x_offset = 0  # 0 would be what we wanted
                    y_offset = 0  # 0 in our case
                    timeseries_all[i, j, x_offset:timeseries.shape[0]+x_offset,y_offset:timeseries.shape[1]+y_offset] = timeseries
                    timeseries_smooth_all[i, j, x_offset:timeseries.shape[0]+x_offset,y_offset:timeseries.shape[1]+y_offset] = timeseries_smooth
                    
                    #if n_scans is the same for all subj use 1 line
                    #timeseries_all[i, j, :, :] = timeseries
                    #timeseries_all.append(timeseries)
                           
            np.save(f'{out_dir}{task}/LB_{task}_timeseries_{parcellations[p,1]}_denoised_acompcor_no_smooth_{run}.npy', timeseries_all)
            np.save(f'{out_dir}{task}/LB_{task}_timeseries_{parcellations[p,1]}_denoised_acompcor_smooth_{run}.npy', timeseries_smooth_all)
      
          
          