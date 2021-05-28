#%%
# start this after running fMRIprep and MRIQC

#Denoising procedure
#Procedure for denoisng of data from working memory training experiment.

#Confound regression:

#aCompCor
#24 motion parameters (Satterthwaite et al., 2013)
#outlier scans based on FD (> 0.5 mm) and DVARS (> 3 SD)
#task effects + temporal derivatives
#Last edited: 26-09-2020

#%% Step 0: Load libraries

import sys
sys.path.append("..")
                
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import warnings
import nibabel as nib

from nilearn import plotting, input_data, signal, image  
from nistats.design_matrix import make_first_level_design_matrix
from fctools import denoise

warnings.filterwarnings('ignore')
#%% Step 1: Prepare paths to files
# Setting directories for input and output files
top_dir = '/home/alado/datasets/RBH'
data_dir = '/home/alado/datasets/RBH/preprocessed/derivatives/fmriprep'
out_dir = '/home/alado/datasets/RBH/Lipsia/'
suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
mask_suffix = '_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'

# Selecting subjects who finished the study
groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])

# Setting sessions and task names
sess = ['ses-1', 'ses-3']
runs = ['run-1', 'run-2']
task = 'nback'
n_scans_1_list = []
n_scans_3_list = []
t_r = 3

print(f'Sample size: {len(subs)}')

#%% Step 2: generate design matrix if it's the same for all subjects
# Settings
#t_r = 3
#n_scans = 214
#frame_times = np.arange(n_scans) * t_r

# Load task-block onsets
#events = pd.read_csv('../support/onsets_nback.csv')

# Convolving box car function with HRF
#box_hrf = make_first_level_design_matrix(frame_times, events, hrf_model='glover')
#box_hrf  = box_hrf.reset_index()

# Visualise
#plotting_visible = [1,2,3,4,5]
#plt.figure(figsize=(15, 3))
#plt.title('HRF convolved with task block box-car function')
#for cond in plotting_visible:
#    plt.plot(box_hrf[cond], label=cond)
#plt.legend(loc='upper center', bbox_to_anchor=(1.05, 1), fancybox=True, shadow=True);

#%% Step 3: Denoising data for connectivity
for sub in subs:
    print(f'Denoising {sub}')
    for ses in sess:
        print(f'- {ses}')
        for run in runs:
            print(run)

            # Getting directory/file names
            sub_dir = f'{top_dir}/preprocessed/derivatives/fmriprep/{sub}/{ses}/func/'
            sub_name = f'{sub}_{ses}_task-{task}_{run}' 
            epi_preproc_path = f'{sub_dir}{sub_name}{suffix}'
            events_dir = f'{top_dir}/Nifti/{sub}/{ses}/func/'
            fmri_mask_path = f'{sub_dir}{sub_name}{mask_suffix}'
            
            fmri_mask = nib.load(fmri_mask_path)
            
            # Loading confound data
            confounds_path = f'{sub_dir}{sub_name}_desc-confounds_regressors.tsv'
            if not os.path.exists(confounds_path):
                 print(f'{sub}-{ses}-{task} confounds file does not exist')
            else:
                confounds = pd.read_csv(confounds_path, delimiter='\t')
                confounds = confounds.replace([np.inf, -np.inf], np.nan)
                confounds = confounds.fillna(0)
                # Selecting columns of interest 
                confounds_motion = confounds[confounds.filter(regex='trans_x|trans_y|trans_z|rot_x|rot_y|rot_z').columns]
                confounds_acompcor = confounds[confounds.filter(regex='a_comp_cor').columns]
                confounds_acompcor = confounds_acompcor.drop(confounds_acompcor.columns[7:], axis=1)
                confounds_cosine = confounds[confounds.filter(regex='cosine').columns]
                confounds_outlier = confounds[confounds.filter(regex='std_dvars|framewise_displacement').columns]
                confounds_anatomical = confounds[confounds.filter(regex='csf|white_matter').columns]
                
                # estimate box_hrf here 
                events_path = f'{events_dir}{sub_name}_events.tsv'
                events = pd.read_csv(events_path, delimiter='\t')
                
                subject_data = nib.load(epi_preproc_path)
                n_scans = subject_data.shape[-1]
                frame_times = np.arange(n_scans) * t_r
                if ses == sess[0]:
                    n_scans_1_list.append(n_scans)
                else:
                    n_scans_3_list.append(n_scans)
                
                box_hrf = make_first_level_design_matrix(frame_times, events, hrf_model='glover')
                box_hrf = box_hrf.reset_index()
                box_hrf.columns = box_hrf.columns.astype(str)
                
                plotting_visible = ['1','2','3','4','5']
                plt.figure(figsize=(15, 3))
                plt.title('HRF convolved with task block box-car function')
                for cond in plotting_visible:
                    plt.plot(box_hrf[cond], label=cond)
                plt.legend(loc='upper center', bbox_to_anchor=(1.05, 1), fancybox=True, shadow=True);
                
                # Calculating task effects
                task_effects_td = denoise.temp_deriv(pd.DataFrame(box_hrf, columns=['1', '2', '3', '4', '5']), quadratic=False)
                task_effects_td = task_effects_td.replace([np.inf, -np.inf], np.nan)
                task_effects_td = task_effects_td.fillna(0)
                
                # Concatenating columns
                confounds_clean = pd.concat([confounds_motion, 
                                         confounds_acompcor,
                                         confounds_cosine,
                                         confounds_outlier,
                                         confounds_anatomical,
                                         task_effects_td], 
                                         axis=1)
                
                confounds_clean = confounds_clean.replace([np.inf, -np.inf], np.nan)
                confounds_clean = confounds_clean.fillna(0)
                
                # Saving preprocessed confound file
                confounds_clean_path = f'{top_dir}/func_connect/01-conf_clean_acompcor/{sub}/{sub_name}_bold_confounds_clean_acompcor.csv'
                confounds_clean.to_csv(confounds_clean_path, sep=',', index=False)
               
                # Voxel-vise signal denoising
                denoised_nifti = image.clean_img(subject_data, 
                                                 confounds=np.asarray(confounds_clean), 
                                                 detrend=True, 
                                                 standardize=True,
                                                 ensure_finite=True,
                                                 low_pass=0.09, 
                                                 high_pass=0.008,
                                                 t_r=t_r, 
                                                 mask_img=fmri_mask)
                
                # Saving denoised file
                denoised_nifti.to_filename(f'{top_dir}/func_connect/01-conf_clean_acompcor/{sub}/{sub_name}_denoised_acompcor_task_effects.nii.gz')
                #for Lipsia
                nib.save(denoised_nifti, f'{out_dir}/denoised/{sub}/{ses}/func/{sub_name}_space-MNI152NLin2009cAsym_denoised.nii.gz')
                
                # Smooth
                smooth_img = image.smooth_img(denoised_nifti, fwhm=3)
                smooth_img.to_filename(f'{top_dir}/func_connect/01-conf_clean_acompcor/{sub}/{sub_name}_denoised_acompcor_task_effects_smooth_fwhm3.nii.gz')
                










