#First-level GLM analysis of working memory/stress fMRI data
#Supplementary analysis - standard GLM analysis of activation patterns changes between 0-, 1-, 2-, 4-back, and fixation.

#First level n-back task GLM analysis performed for each subject and sessions 1 and 3.

#contrasts of interest:
#included confounds regressors analogical to fc (aCompCor, 24 motion parameters, outlier scans)
#Outputs:

#z-maps for each subject and session

import sys
sys.path.append("..")

import os
import pandas as pd
import numpy as np
import nibabel as nib

from nilearn import plotting
from nilearn import masking
from nilearn.image import mean_img, resample_to_img, concat_imgs
from nilearn.plotting import plot_stat_map, plot_anat, plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.first_level import compute_regressor
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.datasets import fetch_icbm152_brain_gm_mask
from nilearn.glm.contrasts import compute_fixed_effects


#%%Setup
data_dir = '/home/alado/datasets/RBH/preprocessed/derivatives/fmriprep'
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/GLM/FLA/'
#conf_suffix = '_pipeline-24HMP8PhysSpikeReg_desc-confounds_confounds.tsv'
#conf_suffix = '_pipeline-24HMP8PhysSpikeReg4GS_desc-confounds_confounds.tsv'
#conf_suffix = '_pipeline-24HMPaCompCorSpikeReg_desc-confounds_confounds.tsv'
conf_suffix = '_pipeline-24HMPaCompCorSpikeReg4GSR_desc-confounds_confounds.tsv'
suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
mask_suffix = '_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
gm_mask_suffix = '_space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz'

groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])

sess = ['ses-1', 'ses-3']
task = 'nback'
t_r = 3.0
#n_voxels = 265265
#timeseries = np.zeros((len(subs), len(sess), n_voxels))        
        
#%%create masks
#use MNI grey matter mask
icbm_mask = fetch_icbm152_brain_gm_mask()

fmri_masks = []
for i, sub in enumerate(subs):
    for j, ses in enumerate(sess):
        sub_dir = f'{top_dir}/preprocessed/derivatives/fmriprep/{sub}/{ses}/func/'
        sub_name = f'{sub}_{ses}_task-nback_run-1'
        fmri_mask_path = f'{sub_dir}{sub_name}{mask_suffix}'
        fmri_mask = nib.load(fmri_mask_path)
        fmri_masks.append(fmri_mask)

        sub_name = f'{sub}_{ses}_task-nback_run-2'
        fmri_mask_path = f'{sub_dir}{sub_name}{mask_suffix}'
        fmri_mask = nib.load(fmri_mask_path)
        fmri_masks.append(fmri_mask)
        

common_mask = masking.intersect_masks(fmri_masks, threshold=0.5, connected=False)
nib.save(common_mask, f'{top_dir}/GLM/common_whole_brain_mask.nii.gz')

#%% Compute first-level contrasts with grey matter mask
for i, sub in enumerate(subs):
    print(f'Computing {sub}')
    for j, ses in enumerate(sess):
        print(f'- {ses}')

        sub_dir = f'{top_dir}/preprocessed/derivatives/fmriprep/{sub}/{ses}/func/'
        #events_dir = f'{sub_dir}/{sub}/{ses}/func/'
        
        sub_name = f'{sub}_{ses}_task-nback_run-1'
        fmri_img_path = f'{sub_dir}{sub_name}{suffix}'
        fmri_img_1 = nib.load(fmri_img_path)
        events_path = f'{sub_dir}{sub_name}_events.tsv'
        events_1 = pd.read_csv(events_path, delimiter='\t')
        #events_1['modulation'] = events_1['accuracy']
        events_1['trial_type'] = events_1['trial_type'].astype(str)
        #events = events.loc[:, ['onset', 'duration', 'trial_type']]
        # create a "modulation" column with only ones (indicating no modulation)  
        #events.loc[:, 'modulation'] = 1  # this automatically fills all rows with 1
        confounds_path = f'{top_dir}/preprocessed/derivatives/fmridenoise/{ses}/{sub}/{sub_name}{conf_suffix}'
        confounds_1 = pd.read_csv(confounds_path, delimiter='\t')
        
        sub_name = f'{sub}_{ses}_task-nback_run-2'
        fmri_img_path = f'{sub_dir}{sub_name}{suffix}'
        fmri_img_2 = nib.load(fmri_img_path)
        events_path = f'{sub_dir}{sub_name}_events.tsv'
        events_2 = pd.read_csv(events_path, delimiter='\t')
        #events_2['modulation'] = events_2['accuracy']
        events_2['trial_type'] = events_2['trial_type'].astype(str)
        confounds_path = f'{top_dir}/preprocessed/derivatives/fmridenoise/{ses}/{sub}/{sub_name}{conf_suffix}'
        confounds_2 = pd.read_csv(confounds_path, delimiter='\t')
        
        performance = pd.read_csv(f'{top_dir}/behavioural/{ses}_task_performance.csv', delimiter=',')

        #concatenate for the mask
        #run-1 nbackA, has 1 in column if it was first
        if ses == 'ses-1':
            if groups.iloc[i]['ses-1_run-1'] == 1:
                fmri_img = concat_imgs((fmri_img_1, fmri_img_2))
            else:
                fmri_img = concat_imgs((fmri_img_2, fmri_img_1))
        else:
            if groups.iloc[i]['ses-3_run-1'] == 1:
                fmri_img = concat_imgs((fmri_img_1, fmri_img_2))
            else:
                fmri_img = concat_imgs((fmri_img_2, fmri_img_1))
        
        #create mean img for plotting
        mean_img_ = mean_img(fmri_img)       
        
        #here standard MNI grey matter mask
        #resample icbm to img since mask has different resolution
        fmri_mask = resample_to_img(icbm_mask, fmri_img, interpolation='nearest')
        #fmri_mask = icbm_mask
        #is it better to use space-MNI152NLin2009cAsym_label-GM_probseg
        #fmri_mask_path = f'{top_dir}/preprocessed/derivatives/fmriprep/{sub}/anat/{sub}{gm_mask_suffix}'
        #fmri_mask = nib.load(fmri_mask_path)       

        confounds_1 = confounds_1.replace([np.inf, -np.inf], np.nan)
        confounds_1 = confounds_1.fillna(0)
        # Selecting columns of interest
        #motion + parameter expansion
        #confounds_motion_1 = confounds_1[confounds_1.filter(regex='trans_x|trans_y|trans_z|rot_x|rot_y|rot_z').columns]
        #confounds_acompcor_1 = confounds_1[confounds_1.filter(regex='a_comp_cor').columns]
        #confounds_acompcor_1 = confounds_acompcor_1.drop(confounds_acompcor_1.columns[3:], axis=1)
        #include either here or in the drift_model='cosine' in first level model
        #confounds_cosine = confounds[confounds.filter(regex='cosine').columns]
        #confounds_scrub_1 = confounds_1[confounds_1.filter(regex='std_dvars|framewise_displacement').columns]
        # Detecting outliers 
        #confounds_scrub_td_1 = denoise.temp_deriv(denoise.outliers_fd_dvars(confounds_scrub_1, fd=0.5, dvars=3), quadratic=False)
        # Concatenating columns
        #confounds_clean_1 = pd.concat([confounds_motion_1, 
        #                         confounds_acompcor_1,
        #                         #confounds_cosine,
        #                         confounds_scrub_td_1], 
        #                         axis=1)
        confounds_2 = confounds_2.replace([np.inf, -np.inf], np.nan)
        confounds_2 = confounds_2.fillna(0)        
        
        hrf_model = 'spm + derivative'
        #hrf_model = 'spm'
        
        #Setup GLM model
        glm = FirstLevelModel(
            t_r=t_r,   
            slice_time_ref = 0.5, #bc fmriprep realings in time to the middle of each TR
            hrf_model=hrf_model,
            #hrf_model='spm',
            drift_model='cosine',
            high_pass=0.01,
            mask_img = fmri_mask, #nifti image
            #smoothing_fwhm=3.5,   
            noise_model='ar1',
            standardize=True, #better True for stat analysis of connectivity!
            minimize_memory=False)
        
        glm_1 = glm.fit(fmri_img_1, events=events_1, confounds=confounds_1)
        
        glm = FirstLevelModel(
            t_r=t_r,   
            slice_time_ref = 0.5, #bc fmriprep realings in time to the middle of each TR
            hrf_model=hrf_model,
            #hrf_model='spm',
            drift_model='cosine',
            high_pass=0.01,
            mask_img = fmri_mask, #nifti image
            #smoothing_fwhm=3.5,   
            noise_model='ar1',
            standardize=True, #better True for stat analysis of connectivity!
            minimize_memory=False)
        
        glm_2 = glm.fit(fmri_img_2, events=events_2, confounds=confounds_2)
        
        
        #fit fist level model on 2 runs
        #if ses == 'ses-1':
        #    if groups.iloc[i]['ses-1_run-1'] == 1:
        #        glm = glm.fit([fmri_img_1, fmri_img_2], events=[events_1, events_2], confounds=[confounds_clean_1, confounds_clean_2])
        #    else:
        #        glm = glm.fit([fmri_img_2, fmri_img_1], events=[events_2, events_1], confounds=[confounds_clean_2, confounds_clean_1])
        #else:
        #    if groups.iloc[i]['ses-3_run-1'] == 1:
        #        glm = glm.fit([fmri_img_1, fmri_img_2], events=[events_1, events_2], confounds=[confounds_clean_1, confounds_clean_2])
        #    else:
        #        glm = glm.fit([fmri_img_2, fmri_img_1], events=[events_2, events_1], confounds=[confounds_clean_2, confounds_clean_1])
       
        #glm = glm.fit(fmri_img, events=events, confounds=confounds_clean)        
        
        
        if hrf_model == 'spm + derivative':
            # the design_matrices_ attribute is a list with, in our case, only a single element
            design_matrix_1 = glm_1.design_matrices_[0]
            #keep this only if you use derivatives
            cols = design_matrix_1.columns.tolist()
            cl = ['1', '2', '3', '4', '5', '1_derivative', '2_derivative', '3_derivative', '4_derivative', '5_derivative'] + cols[10:]
            design_matrix_1 = design_matrix_1[cl]
            
            design_matrix_2 = glm_2.design_matrices_[0]
            cols = design_matrix_2.columns.tolist()
            cl = ['1', '2', '3', '4', '5', '1_derivative', '2_derivative', '3_derivative', '4_derivative', '5_derivative'] + cols[10:]
            design_matrix_2 = design_matrix_2[cl]
        else:
            design_matrix_1 = glm_1.design_matrices_[0]
            design_matrix_2 = glm_2.design_matrices_[0]
        
        design_matrix_1['accuracy'] = performance.loc[i,'dPrime']
        design_matrix_2['accuracy'] = performance.loc[i,'dPrime']
        
        # #new regressor - convolved accuracy
        # n_scans1 = design_matrix_1.shape[0]
        # frame_times1 = np.linspace(0, (n_scans1 - 1) * t_r, n_scans1)
        # exp_condition1 = np.array((events_1['onset'], events_1['duration'], events_1['accuracy'])).reshape(3, events_1.shape[0])
        # signal1, name = compute_regressor(exp_condition1, 'spm', frame_times1, con_id='accuracy',oversampling=16)
        # design_matrix_1['accuracy'] = signal1
        
        # n_scans2 = design_matrix_2.shape[0]
        # frame_times2 = np.linspace(0, (n_scans2 - 1) * t_r, n_scans2)       
        # exp_condition2 = np.array((events_2['onset'], events_2['duration'], events_2['accuracy'])).reshape(3, events_2.shape[0])
        # signal2, name = compute_regressor(exp_condition2, 'spm', frame_times2, con_id='accuracy',oversampling=16)
        # design_matrix_2['accuracy'] = signal2
        
        # #refit GLM
        # glm = FirstLevelModel(
        #     t_r=t_r,   
        #     slice_time_ref = 0.5, #bc fmriprep realings in time to the middle of each TR
        #     hrf_model=hrf_model,
        #     drift_model='cosine',
        #     high_pass=0.01,
        #     mask_img = fmri_mask, #nifti image
        #     #smoothing_fwhm=3.5,   
        #     noise_model='ar1',
        #     standardize=True, #better True for stat analysis of connectivity!
        #     minimize_memory=False)
        # glm_1 = glm.fit(fmri_img_1, design_matrices = design_matrix_1)
        
        # glm = FirstLevelModel(
        #     t_r=t_r,   
        #     slice_time_ref = 0.5, #bc fmriprep realings in time to the middle of each TR
        #     hrf_model=hrf_model,
        #     drift_model='cosine',
        #     high_pass=0.01,
        #     mask_img = fmri_mask, #nifti image
        #     #smoothing_fwhm=3.5,   
        #     noise_model='ar1',
        #     standardize=True, #better True for stat analysis of connectivity!
        #     minimize_memory=False)        
        # glm_2 = glm.fit(fmri_img_2, design_matrices = design_matrix_2)       
        
        
        #plot
        #ax = plot_design_matrix(design_matrix)
        #ax.get_images()[0].set_clim(0, 0.2)
        
        #r`2: returns a list, take the first element, because we only have one run
        #r2_img = glm.r_square[0]
        #plotting.plot_stat_map(r2_img, threshold=0.2)
        #plotting.show()
        
        #resids = glm.residuals[0]
        
        n_columns = len(design_matrix_1.columns)
        #22 contrasts
        contrasts_1 = {
                    #'0back-1back': np.pad([1, -1, 0, 0, 0], (0,n_columns-5)),
                    #'0back-2back': np.pad([1, 0, -1, 0, 0], (0,n_columns-5)),
                    #'0back-4back': np.pad([1, 0, 0, -1, 0], (0,n_columns-5)),
                    #'0back-fix': np.pad([1, 0, 0, 0, -1], (0,n_columns-5)),
                    '1back-0back': np.pad([-1, 1, 0, 0, 0], (0,n_columns-5)),
                    #'1back-2back': np.pad([0, 1, -1, 0, 0], (0,n_columns-5)),
                    #'1back-4back': np.pad([0, 1, 0, -1, 0], (0,n_columns-5)),
                    '1back-fix': np.pad([0, 1, 0, 0, -1], (0,n_columns-5)),
                    '2back-0back': np.pad([-1, 0, 1, 0, 0], (0,n_columns-5)),
                    '2back-1back': np.pad([0, -1, 1, 0, 0], (0,n_columns-5)),
                    #'2back-4back': np.pad([0, 0, 1, -1, 0], (0,n_columns-5)),
                    '2back-fix': np.pad([0, 0, 1, 0, -1], (0,n_columns-5)),
                    '4back-0back': np.pad([-1, 0, 0, 1, 0], (0,n_columns-5)),
                    '4back-1back': np.pad([0, -1, 0, 1, 0], (0,n_columns-5)),
                    '4back-2back': np.pad([0, 0, -1, 1, 0], (0,n_columns-5)), 
                    '4back-fix': np.pad([0, 0, 0, 1, -1], (0,n_columns-5)),
                    'nback-fix': np.pad([0.2, 0.2, 0.2, 0.2, -0.8],(0,n_columns-5)),
                    'task-0back': np.pad([-0.75, 0.25, 0.25, 0.25, 0],(0,n_columns-5)),
                    'linear_param(slope)_effect': np.pad([0, -0.5, 0, 0.5, 0],(0,n_columns-5)),
                    'quadratic_param(slope)_effect': np.pad([0, -0.333, 0.667, -0.333, 0],(0,n_columns-5)),
                    #'0back': np.pad([1, 0, 0, 0, 0], (0,n_columns-5)),
                    #'1back': np.pad([0, 1, 0, 0, 0], (0,n_columns-5)),
                    #'2back': np.pad([0, 0, 1, 0, 0], (0,n_columns-5)),
                    #'4back': np.pad([0, 0, 0, 1, 0], (0,n_columns-5)),                  
                    #'fix': np.pad([0, 0, 0, 0, 1], (0,n_columns-5)),
                    #'f-contrast': np.eye(n_columns)[:5]
                    }
        
        n_columns = len(design_matrix_2.columns)
        contrasts_2 = {
                    #'0back-1back': np.pad([1, -1, 0, 0, 0], (0,n_columns-5)),
                    #'0back-2back': np.pad([1, 0, -1, 0, 0], (0,n_columns-5)),
                    #'0back-4back': np.pad([1, 0, 0, -1, 0], (0,n_columns-5)),
                    #'0back-fix': np.pad([1, 0, 0, 0, -1], (0,n_columns-5)),
                    '1back-0back': np.pad([-1, 1, 0, 0, 0], (0,n_columns-5)),
                    #'1back-2back': np.pad([0, 1, -1, 0, 0], (0,n_columns-5)),
                    #'1back-4back': np.pad([0, 1, 0, -1, 0], (0,n_columns-5)),
                    '1back-fix': np.pad([0, 1, 0, 0, -1], (0,n_columns-5)),
                    '2back-0back': np.pad([-1, 0, 1, 0, 0], (0,n_columns-5)),
                    '2back-1back': np.pad([0, -1, 1, 0, 0], (0,n_columns-5)),
                    #'2back-4back': np.pad([0, 0, 1, -1, 0], (0,n_columns-5)),
                    '2back-fix': np.pad([0, 0, 1, 0, -1], (0,n_columns-5)),
                    '4back-0back': np.pad([-1, 0, 0, 1, 0], (0,n_columns-5)),
                    '4back-1back': np.pad([0, -1, 0, 1, 0], (0,n_columns-5)),
                    '4back-2back': np.pad([0, 0, -1, 1, 0], (0,n_columns-5)), 
                    '4back-fix': np.pad([0, 0, 0, 1, -1], (0,n_columns-5)),
                    'nback-fix': np.pad([0.2, 0.2, 0.2, 0.2, -0.8],(0,n_columns-5)),
                    'task-0back': np.pad([-0.75, 0.25, 0.25, 0.25, 0],(0,n_columns-5)),
                    'linear_param(slope)_effect': np.pad([0, -0.5, 0, 0.5, 0],(0,n_columns-5)),
                    'quadratic_param(slope)_effect': np.pad([0, -0.333, 0.667, -0.333, 0],(0,n_columns-5)),
                    #'0back': np.pad([1, 0, 0, 0, 0], (0,n_columns-5)),
                    #'1back': np.pad([0, 1, 0, 0, 0], (0,n_columns-5)),
                    #'2back': np.pad([0, 0, 1, 0, 0], (0,n_columns-5)),
                    #'4back': np.pad([0, 0, 0, 1, 0], (0,n_columns-5)),                  
                    #'fix': np.pad([0, 0, 0, 0, 1], (0,n_columns-5)),
                    #'f-contrast': np.eye(n_columns)[:5]
                    }

        
        print('Computing contrasts...')
        for index, (contrast_id, contrast_val) in enumerate(contrasts_1.items()):
            print('Contrast % 2i out of %i: %s' % (index + 1, len(contrasts_1), contrast_id))
            # estimate the contasts
            # note that the model implictly computes a fixed effect across the two sessions
            #summary_stats = glm.compute_contrast(contrast_val, output_type='all')
            summary_statistics_run1 = glm_1.compute_contrast(contrasts_1[contrast_id], output_type='all')
            summary_statistics_run2 = glm_2.compute_contrast(contrasts_2[contrast_id], output_type='all')
            
            contrast_imgs = [summary_statistics_run1['effect_size'], summary_statistics_run2['effect_size']]
            variance_imgs = [summary_statistics_run1['effect_variance'], summary_statistics_run2['effect_variance']]
            
            fixed_fx_contrast, fixed_fx_variance, fixed_fx_stat = compute_fixed_effects(contrast_imgs, variance_imgs, fmri_mask)
            #plotting.plot_stat_map(fixed_fx_stat, bg_img=mean_img_, threshold=3.0, cut_coords=cut_coords,
            #    title='{0}, fixed effects'.format(contrast_id))
            
            #The fixed effects version displays higher peaks than the input sessions.
            #Computing fixed effects enhances the signal-to-noise ratio of the resulting brain maps.
            #Note however that, technically, the output maps of the fixed effects map is a t statistic
            #(not a z statistic)
            #t to z conversion
            
            #z_map = glm.compute_contrast(contrast_val, output_type='z_score') if there is one run
            #plotting.plot_stat_map(z_map, bg_img=mean_img_, threshold=2.0,title='%s' % contrast_id)
            #timeseries[i,j,:] = masker.fit_transform(z_map)

            # Saving z_maps
            #or here we save the beta estimates of the fixed effects for the 2nd level
            save_path = os.path.join(out_dir, 'voxel')
            #os.mkdir(save_path)
            nib.save(fixed_fx_contrast, f'{save_path}/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast_id}.nii.gz')
            
            #Saving all outputs
            #np.save(summary_stats, f'{out_dir}/voxel/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast_id}_summary_stats.npy')
            
            #g = plot_glass_brain(z_map, colorbar=True, plot_abs=False, title=f'{sub}, {ses}')
            #g.savefig(f'{out_dir}/voxel/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast_id}_glass_brain.png')
        
            #s = plot_stat_map(z_map, threshold=3, title=f'{sub}, {ses}')
            #s.savefig(f'{out_dir}/voxel/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast_id}_slices.png')
        
        #np.save(f'{out_dir}GLM_power_0b_minus_1b_zmap_timeseries.npy', timeseries)
        
            #save resampled mask
            save_path = os.path.join(out_dir, 'resampled_masks')
            #os.mkdir(save_path)
            nib.save(fmri_mask, f'{save_path}/{sub}_{ses}_resampled_icbm_mask.nii.gz')
        #%%   
import sys
import os
sys.path.append("..")
import itertools
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import plotting
from nilearn import image
from nilearn.plotting import plot_stat_map, plot_anat
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.input_data import NiftiMasker
from nilearn.glm import threshold_stats_img, cluster_level_inference
from nilearn.image import get_data, math_img
from nilearn.reporting import make_glm_report
from nilearn.reporting import get_clusters_table

out_dir = '/home/alado/datasets/RBH/GLM/SLA/'
sections = ['z', 'x', 'y']
rand_state = 42
nperms = 10000

threshold = -np.log10(0.05) #5% corrected

contr_fl_interest = ['4back-2back', '4back-1back', '4back-0back', '4back-fix', 
                     '2back-1back', '2back-0back', '2back-fix', 'nback-fix', 
                     'task-0back', 'linear_param(slope)_effect', 'quadratic_param(slope)_effect']

groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])
#add subjects with high motion in at least one session
#high_motion = ['sub-']
#high_motion_filter = ~subjects_data_clean['sub'].isin(high_motion).values
# Removing high-motion subjects from dataset
#subjects_data_clean_lm = subjects_data_clean[~subjects_data_clean['sub'].isin(high_motion)].reset_index()

fmri_mask_path = f'{top_dir}/GLM/resampled_icbm_mask.nii.gz'
fmri_mask = nib.load(fmri_mask_path)

print('Pre one-samplet-test smoothing_fwhm=5.0 for all groups')
nsubs = len(groups)

for k, contrast in enumerate(contr_fl_interest):
    print(contrast)
    zmaps = [] 
    for i, sub in enumerate(subs):
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/{sub}_ses-1_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps.append(zmap)

    
    condition_effect = np.hstack([1] * nsubs)
    
    design_matrix = pd.DataFrame(condition_effect[:, np.newaxis], columns=['pre'])
    contrast_val = 'pre'
    
    second_level_model = SecondLevelModel(mask_img=fmri_mask,smoothing_fwhm=5.0)
    second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)       
    nib.save(z_map, f'{save_path}/allgroups_pre_permuted_{contrast}_zmap.nii.gz')
    
    #% Computing the (corrected) negative log p-values with permutation test
    neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
                                  design_matrix=design_matrix,
                                  second_level_contrast=contrast_val,
                                  model_intercept=True, n_perm=nperms,
                                  two_sided_test=False, mask=fmri_mask,
                                  smoothing_fwhm=5.0, random_state=rand_state,
                                  n_jobs=4)
    
    THRESH = -np.log10(0.05)  #5% corrected
    affine = neg_log_pvals_permuted_ols_unmasked.affine
    header = neg_log_pvals_permuted_ols_unmasked.header
    pvals_array = neg_log_pvals_permuted_ols_unmasked.get_fdata()
    z_map_array = z_map.get_fdata()
    z_map_array[pvals_array < THRESH] = 0
    z_map_thresh = nib.Nifti1Image(z_map_array, affine)
    
    #plot the (corrected) negative log p-values
    title = ('Group-level association between \n'
              'neg-log of non-parametric corrected p-values (FWER < 5%)')
    s = plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True, vmax=3, threshold=threshold,
                            title=title)
    plotting.show()
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre/figures/permut'
    if not os.path.exists(save_path):
        os.makedirs(save_path)       
    s.savefig(f'{save_path}/allgroups_{contrast}_thresh_zmap_permut.png')
    
    for section in sections:
        s = plot_stat_map(z_map_thresh, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      threshold=threshold,
                      colorbar=True,
                      annotate=True,
                      )
        s.savefig(f'{save_path}/allgroups_{contrast}_thresh_zmap_permut_{section}.png')
    plotting.show()
    
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre/'
    nib.save(neg_log_pvals_permuted_ols_unmasked, f'{save_path}/allgroups_pre_permuted_{contrast}.nii.gz')
    nib.save(z_map_thresh, f'{save_path}/allgroups_pre_permuted_{contrast}_zmap_thresh.nii.gz')
    
    # Since we are plotting negative log p-values and using a threshold equal to 0.05,
    # it corresponds to corrected p-values lower than 5%, meaning that there
    # is less than 5% probability to make a single false discovery
    # (95% chance that we make no false discovery at all).
    table = get_clusters_table(z_map_thresh, stat_threshold=0,
                               cluster_threshold=2, min_distance=0)
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre/tables/permut'
    if not os.path.exists(save_path):
        os.makedirs(save_path)   
    table.to_csv(f'{save_path}/allgroups_clusters_table_{contrast}_neg_log_pvals_permuted_ols.csv')
    
    # The neg-log p-values obtained with nonparametric testing are capped at 3
    # since the number of permutations is 1e3. 

#%%

"""
Created on Wed Dec  9 17:39:41 2020

@author: alado
"""

#Working memory training: Second level GLM analysis of working memory training fMRI data
#Second-level GLM analysis of activation patterns

import sys
import os
sys.path.append("..")
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import plotting, image, masking
from nilearn.plotting import plot_stat_map, plot_anat, plot_img, show, plot_glass_brain, plot_connectome
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.input_data import NiftiMasker
from nilearn.glm import threshold_stats_img, cluster_level_inference
from nilearn.image import get_data, math_img
from nilearn.reporting import make_glm_report
from nilearn.reporting import get_clusters_table

#%%Step 1: Loading data
data_dir = '/home/alado/datasets/RBH/preprocessed/derivatives/fmriprep'
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/GLM/SLA'
suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
mask_suffix = '_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
sess = ['ses-1', 'ses-3']
task = 'nback'
t_r = 3.0
sections = ['z', 'x', 'y']

contrasts_fl = ['0back-1back','0back-2back','0back-4back','0back-fix', '1back-0back',
             '1back-2back','1back-4back','1back-fix',  '2back-0back','2back-1back',
             '2back-4back','2back-fix','4back-0back', '4back-1back', '4back-2back',
             '4back-fix', '0back','1back','2back','4back','fix', 'effects_of_interest']

#We are not interested in all contrasts, we are mainly interested in contrasts corresponding to the
#increased cognitive load, like 4back-1, 4back-2, and 2back-1
contr_fl_interest = ['4back-2back', '4back-1back', '4back-0back', '4back-fix', 
                     '2back-1back', '2back-0back', '2back-fix', 'nback-fix', 
                     'task-0back', 'linear_param(slope)_effect', 'quadratic_param(slope)_effect']

groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])
#add subjects with high motion in at least one session
#high_motion = ['sub-']
#high_motion_filter = ~subjects_data_clean['sub'].isin(high_motion).values
# Removing high-motion subjects from dataset
#subjects_data_clean_lm = subjects_data_clean[~subjects_data_clean['sub'].isin(high_motion)].reset_index()

fmri_mask_path = f'{top_dir}/GLM/resampled_icbm_mask.nii.gz'
fmri_mask = nib.load(fmri_mask_path)
    
#INDEPENDENT T-TEST
#design_matrix['constant'] = [1] * nsubs * 2
#no need to include an intercept as it is simply a linear combination of the other two regressors
#design_matrix['pre'] = [1] * nsubs + [0] * nsubs
#design_matrix['post'] = [0] * nsubs + [1] * nsubs
    
#%%Pre/post (related) t-test smoothing_fwhm=5.0 for sport group with second-level model

print('Pre/post t-test smoothing_fwhm=5.0 for sport group with second-level model')

subs_sport = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport'])])
nsubs = len(subs_sport)

for k, contrast in enumerate(contr_fl_interest):
    print(contrast)
    zmaps_s1 = [] 
    for i, sub in enumerate(subs_sport):
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/{sub}_ses-1_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps_s1.append(zmap)
            
    zmaps_s3 = [] 
    for i, sub in enumerate(subs_sport):
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/{sub}_ses-3_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps_s3.append(zmap)
    
    zmaps = zmaps_s1 + zmaps_s3
    
    condition_effect = np.hstack(([-1] * nsubs, [1] * nsubs))
    subject_effect = np.vstack((np.eye(nsubs), np.eye(nsubs)))
    subjects = ['S%02d' % i for i in range(1, nsubs + 1)]
    
    design_matrix = pd.DataFrame(
        np.hstack((condition_effect[:, np.newaxis], subject_effect)),
        columns=['pre_vs_post'] + subjects)

    contrast_val = 'pre_vs_post'
    
    second_level_model = SecondLevelModel(mask_img=fmri_mask,smoothing_fwhm=5.0)
    second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    nib.save(z_map, f'{save_path}/sport_pre_post_zmap_{contrast}.nii.gz')
    
    #uncorrecter p<0.001
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, 
                                                       height_control=None, 
                                                       cluster_threshold=2)
    s = plotting.plot_stat_map(thresholded_map1, 
                               threshold=threshold1, colorbar=True)
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/uncorrected'
    if not os.path.exists(save_path):
        os.makedirs(save_path)   
    s.savefig(f'{save_path}/sport_{contrast}_thresh_zmap_uncorr<0.001.png')
    for section in sections:
        s = plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False)
        s.savefig(f'{save_path}/sport_{contrast}_thresh_zmap_uncorr<0.001_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map1, threshold=threshold1)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/sport_pre_post_uncorr_thresh-001_{contrast}.nii.gz')
      
    table = get_clusters_table(z_map, stat_threshold=threshold1,
                               cluster_threshold=2, min_distance=0)
    
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post/tables/uncorrected'
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    table.to_csv(f'{save_path}/sport_clusters_table_{contrast}_uncorrected.csv')
    
    #fpr-thresholded map
    #level 0.005 means that there is 0.5% chance of declaring an inactive voxel, active
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=7)
    title = ('Thresholded z map, fpr <.001, clusters > 7 voxels \n'
             f'sport {contrast} smoothing_fwhm=5.0')
    s = plotting.plot_stat_map(
        thresholded_map1, threshold=threshold1, colorbar=True,
        title=title)
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/fpr'
    if not os.path.exists(save_path):
        os.makedirs(save_path)   
    s.savefig(f'{save_path}/sport_{contrast}_thresh_zmap_fpr<0.005.png')
    for section in sections:
        s = plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/fpr/sport_{contrast}_thresh_zmap_fpr<0.005_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map1, threshold=threshold1)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/sport_pre_post_fpr_thresh-001_{contrast}.nii.gz')
      
    table = get_clusters_table(z_map, stat_threshold=threshold1,
                               cluster_threshold=2, min_distance=0)
    
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post/tables/fpr'
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    table.to_csv(f'{save_path}/sport_clusters_table_{contrast}_fpr.csv')
    
    #fdr
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    title = (f'Thresholded z map, sport {contrast} expected fdr = .05')
    print('The FDR=.05 threshold is %.3g' % threshold2)
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/fdr'
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    for section in sections:
        s = plot_stat_map(thresholded_map2, 
                      threshold=threshold2,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{save_path}/sport_{contrast}_thresh_zmap_fdr<0.05_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map2, threshold=threshold2)
    nib.save(thresh_zmap, 
             f'{out_dir}/voxel/grey_matter_mask/pre_post/sport_pre_fdr_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold2,
                               cluster_threshold=2, min_distance=0)
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post/tables/fdr'
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    table.to_csv(f'{save_path}/sport_clusters_table_{contrast}_fdr.csv')
    
    #Bonferroni (fwer) -thresholded map
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, 
                                                height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    title = (f'Thresholded z map, sport {contrast}\n'
             'expected fwer < .05, smoothing_fwhm=5.0')
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/bonferroni'
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    for section in sections:
        s = plot_stat_map(thresholded_map3, 
                      threshold=threshold3,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{save_path}/sport_{contrast}_thresh_zmap_bonferroni_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map3, threshold=threshold3)
    nib.save(thresh_zmap, 
             f'{out_dir}/voxel/grey_matter_mask/pre_post/sport_pre_post_bonferroni_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold3,
                               cluster_threshold=2, min_distance=0)
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post/tables/bonferroni'
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    table.to_csv(f'{save_path}/sport_clusters_table_{contrast}_bonferroni.csv')
    
    #% Computing the (corrected) negative log p-values with permutation test
    neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
                                  design_matrix=design_matrix,
                                  second_level_contrast=contrast_val,
                                  model_intercept=True, n_perm=10000,
                                  two_sided_test=False, mask=fmri_mask,
                                  random_state=rand_state,
                                  smoothing_fwhm=5.0, n_jobs=1)
    
    #THRESH = 0.5    
    #p_map_thresh = image.threshold_img(neg_log_pvals_permuted_ols_unmasked, 
    #                                   threshold=THRESH)
    #p_map_thresh = image.binarize_img(p_map_thresh)
    #z_map_thresh = image.math_img('img_z*img_p', 
    #                        img_p=p_map_thresh,
    #                        img_z=z_map)
    
    #convert to array
    THRESH = -np.log10(0.05)  
    affine = neg_log_pvals_permuted_ols_unmasked.affine
    header = neg_log_pvals_permuted_ols_unmasked.header
    pvals_array = neg_log_pvals_permuted_ols_unmasked.get_fdata()
    z_map_array = z_map.get_fdata()
    z_map_array[pvals_array < THRESH] = 0
    z_map_thresh = nib.Nifti1Image(z_map_array, affine)
    
    #plot the (corrected) negative log p-values
    title = ('Group-level association between \n'
              'neg-log of non-parametric corrected p-values (FWER < 10%)')
    s = plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                               colorbar=True, draw_cross = False, 
                               threshold=THRESH)
    plotting.show()
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/permut'
    if not os.path.exists(save_path):
        os.makedirs(save_path)       
    s.savefig(f'{save_path}/sport_{contrast}_thresh_pmap_permut.png')
    
    #plot z-map
    s = plotting.plot_stat_map(z_map_thresh, colorbar=True, annotate=True,
                               draw_cross=False)
    plotting.show()
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/permut'
    if not os.path.exists(save_path):
        os.makedirs(save_path)       
    s.savefig(f'{save_path}/sport_{contrast}_thresh_zmap_permut.png')
    
    for section in sections:
        s = plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      colorbar=True,
                      annotate=True,
                      threshold=THRESH
                      )
        s.savefig(f'{save_path}/sport_{contrast}_thresh_pmap_permut_{section}.png')
    plotting.show()
    
    for section in sections:
        s = plot_stat_map(z_map_thresh, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      colorbar=True,
                      annotate=True
                      )
        s.savefig(f'{save_path}/sport_{contrast}_thresh_zmap_permut_{section}.png')
    plotting.show()

    nib.save(neg_log_pvals_permuted_ols_unmasked, 
             f'{out_dir}/voxel/grey_matter_mask/pre_post/sport_pre_post_permuted_{contrast}_pmap.nii.gz')

    nib.save(z_map_thresh, 
             f'{out_dir}/voxel/grey_matter_mask/pre_post/sport_pre_post_permuted_{contrast}_thresh_zmap.nii.gz')
    
    # Since we are plotting negative log p-values and using a threshold equal 
    # to 0.5, it corresponds to corrected p-values lower than 5%, meaning that 
    # there is less than 5% probability to make a single false discovery
    # (95% chance that we make no false discovery at all).
    #table = get_clusters_table(z_map_thresh, stat_threshold=0,
    table = get_clusters_table(z_map_thresh, stat_threshold=0,
                               cluster_threshold=2, min_distance=0)
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post/tables/permut'
    if not os.path.exists(save_path):
        os.makedirs(save_path)   
    table.to_csv(f'{save_path}/sport_clusters_table_{contrast}_neg_log_pvals_permuted_ols.csv')
    
    # The neg-log p-values obtained with nonparametric testing are capped at 3
    # since the number of permutations is 1e3. 

#%%Pre/post (related) t-test smoothing_fwhm=5.0 for meditation group with second-level model

print('Pre/post t-test smoothing_fwhm=5.0 for meditation group with second-level model')

subs_med = pd.Series.tolist(groups['sub'][groups['group'].isin(['med'])])
nsubs = len(subs_med)

for k, contrast in enumerate(contr_fl_interest):
    print(contrast)
    zmaps_s1 = [] 
    for i, sub in enumerate(subs_med):
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/{sub}_ses-1_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps_s1.append(zmap)
            
    zmaps_s3 = [] 
    for i, sub in enumerate(subs_med):
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/{sub}_ses-3_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps_s3.append(zmap)
    
    zmaps = zmaps_s1 + zmaps_s3
    
    condition_effect = np.hstack(([-1] * nsubs, [1] * nsubs))
    subject_effect = np.vstack((np.eye(nsubs), np.eye(nsubs)))
    subjects = ['S%02d' % i for i in range(1, nsubs + 1)]
    
    design_matrix = pd.DataFrame(
        np.hstack((condition_effect[:, np.newaxis], subject_effect)),
        columns=['pre_vs_post'] + subjects)

    contrast_val = 'pre_vs_post'

    second_level_model = SecondLevelModel(mask_img=fmri_mask,smoothing_fwhm=5.0)
    second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)
    
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    nib.save(z_map, f'{out_dir}/voxel/grey_matter_mask/pre_post/med_pre_post_zmap_{contrast}.nii.gz')
    
    #uncorrecter p<0.001
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, 
                                                       height_control=None, 
                                                       cluster_threshold=2)
    s = plotting.plot_stat_map(thresholded_map1, 
                               threshold=threshold1, colorbar=True)
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/uncorrected'
    if not os.path.exists(save_path):
        os.makedirs(save_path)   
    s.savefig(f'{save_path}/med_{contrast}_thresh_zmap_uncorr<0.001.png')
    for section in sections:
        s = plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False)
        s.savefig(f'{save_path}/med_{contrast}_thresh_zmap_uncorr<0.001_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map1, threshold=threshold1)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/med_pre_post_uncorr_thresh-001_{contrast}.nii.gz')
      
    table = get_clusters_table(z_map, stat_threshold=threshold1,
                               cluster_threshold=2, min_distance=0)
    
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post/tables/uncorrected'
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    table.to_csv(f'{save_path}/med_clusters_table_{contrast}_uncorrected.csv')
    
    #fpr-thresholded map
    #level 0.005 means that there is 0.5% chance of declaring an inactive voxel, active
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=7)
    title = ('Thresholded z map, fpr <.001, clusters > 7 voxels \n'
             f'med {contrast} smoothing_fwhm=5.0')
    s = plotting.plot_stat_map(
        thresholded_map1, threshold=threshold1, colorbar=True,
        title=title)
    s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/fpr/med_{contrast}_thresh_zmap_fpr<0.005.png')
    for section in sections:
        s = plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/fpr/med_{contrast}_thresh_zmap_fpr<0.005_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map1, threshold=threshold1)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/med_pre_post_fpr_thresh-001_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold1,
                               cluster_threshold=2, min_distance=0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/tables/fpr/med_clusters_table_{contrast}_fpr.csv'
    table.to_csv(save_dir)
    
    #fdr
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    title = (f'Thresholded z map, med {contrast} expected fdr = .05')
    print('The FDR=.05 threshold is %.3g' % threshold2)    
    for section in sections:
        s = plot_stat_map(thresholded_map2, 
                      threshold=threshold2,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/fdr/med_{contrast}_thresh_zmap_fdr<0.05_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map2, threshold=threshold2)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/med_pre_fdr_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold2,
                               cluster_threshold=2, min_distance=0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/tables/fdr/med_clusters_table_{contrast}_fdr.csv'
    table.to_csv(save_dir)
    
    #Bonferroni (fwer) -thresholded map
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    title = (f'Thresholded z map, med {contrast}\n'
             'expected fwer < .05, smoothing_fwhm=5.0')
    for section in sections:
        s = plot_stat_map(thresholded_map3, 
                      threshold=threshold3,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/bonferroni/med_{contrast}_thresh_zmap_bonferroni_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map3, threshold=threshold3)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/med_pre_post_bonferroni_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold3,
                               cluster_threshold=2, min_distance=0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/tables/bonferroni/med_clusters_table_{contrast}_bonferroni.csv'
    table.to_csv(save_dir)
    
    #% Computing the (corrected) negative log p-values with permutation test
    neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
                                  design_matrix=design_matrix,
                                  second_level_contrast=contrast_val,
                                  model_intercept=True, n_perm=10000,
                                  two_sided_test=False, mask=fmri_mask,
                                  random_state=rand_state,
                                  smoothing_fwhm=5.0, n_jobs=1)
    
    THRESH = -np.log10(0.05)  
    affine = neg_log_pvals_permuted_ols_unmasked.affine
    header = neg_log_pvals_permuted_ols_unmasked.header
    pvals_array = neg_log_pvals_permuted_ols_unmasked.get_fdata()
    z_map_array = z_map.get_fdata()
    z_map_array[pvals_array < THRESH] = 0
    z_map_thresh = nib.Nifti1Image(z_map_array, affine)
    
    #plot the (corrected) negative log p-values
    title = ('Group-level association between \n'
              'neg-log of non-parametric corrected p-values (FWER < 10%)')
    s = plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True, vmax=3,
                            title=title)
    plotting.show()
    s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/permut/med_{contrast}_thresh_pmap_permut.png')
    
    #plot z-map
    s = plotting.plot_stat_map(z_map_thresh, colorbar=True, annotate=True,
                               draw_cross=False)
    plotting.show()
    s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/permut/med_{contrast}_thresh_zmap_permut.png')
    
    for section in sections:
        s = plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      colorbar=True,
                      annotate=True,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/permut/med_{contrast}_thresh_permut_{section}_pmap.png')
    plotting.show()

    nib.save(neg_log_pvals_permuted_ols_unmasked, f'{out_dir}/voxel/grey_matter_mask/pre_post/med_pre_post_permuted_{contrast}_pmap.nii.gz')
    
    for section in sections:
        s = plot_stat_map(z_map_thresh, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      colorbar=True,
                      annotate=True)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/permut/med_{contrast}_thresh_permut_{section}_zmap.png')
    plotting.show()

    nib.save(neg_log_pvals_permuted_ols_unmasked, f'{out_dir}/voxel/grey_matter_mask/pre_post/med_pre_post_permuted_{contrast}_zmap.nii.gz')
    
    #table = get_clusters_table(z_map_thresh, stat_threshold=0,
    table = get_clusters_table(z_map_thresh, stat_threshold=0,
                               cluster_threshold=2, min_distance=0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/tables/permut/med_clusters_table_{contrast}_neg_log_pvals_permuted_ols.csv'
    table.to_csv(save_dir)
    
    # The neg-log p-values obtained with nonparametric testing are capped at 3
    # since the number of permutations is 1e3. 


#%%

"""
Created on Wed Dec  9 17:39:41 2020

@author: alado
"""

#Working memory training: Second level GLM analysis of working memory training fMRI data
#Second-level GLM analysis of activation patterns

import os
import sys
sys.path.append("..")
import itertools
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib as mpl
from nilearn import plotting
from nilearn import image
from nilearn.plotting import plot_stat_map, plot_anat, plot_img, show, plot_glass_brain, plot_connectome
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.input_data import NiftiMasker
from nilearn.glm import threshold_stats_img, cluster_level_inference
from nilearn.image import get_data, math_img
from nilearn.reporting import make_glm_report
from scipy.stats import norm
from nilearn.reporting import get_clusters_table

#%%Step 1: Loading data
data_dir = '/home/alado/datasets/RBH/preprocessed/derivatives/fmriprep'
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/GLM/SLA/'
suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
mask_suffix = '_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'
sess = ['ses-1', 'ses-3']
task = 'nback'
t_r = 3.0
sections = ['z', 'x', 'y']

contrasts_fl = ['0back-1back','0back-2back','0back-4back','0back-fix', '1back-0back',
             '1back-2back','1back-4back','1back-fix',  '2back-0back','2back-1back',
             '2back-4back','2back-fix','4back-0back', '4back-1back', '4back-2back',
             '4back-fix', '0back','1back','2back','4back','fix', 'effects_of_interest']

#We are not interested in all contrasts, we are mainly interested in contrasts corresponding to the
#increased cognitive load, like 4back-1 and 2 back-1 and vice versa
contr_fl_interest = ['4back-2back', '4back-1back', '4back-0back', '4back-fix', 
                     '2back-1back', '2back-0back', '2back-fix', 'nback-fix', 
                     'task-0back', 'linear_param(slope)_effect', 'quadratic_param(slope)_effect']

groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])
#add subjects with high motion in at least one session
#high_motion = ['sub-']
#high_motion_filter = ~subjects_data_clean['sub'].isin(high_motion).values
# Removing high-motion subjects from dataset
#subjects_data_clean_lm = subjects_data_clean[~subjects_data_clean['sub'].isin(high_motion)].reset_index()

fmri_mask_path = f'{top_dir}/GLM/resampled_icbm_mask.nii.gz'
fmri_mask = nib.load(fmri_mask_path)
    
#%%Post t-test smoothing_fwhm=5.0 second-level model

for k, contrast in enumerate(contr_fl_interest):
    zmaps = []
    for i, sub in enumerate(subs):
        zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_ses-3_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
                
    #%Step 2: Create second-level design matrix
    # 1's per subject model out each subj mean
    n_sub = len(subs)
    len_contr = len(contr_fl_interest)
    design_matrix = pd.DataFrame()
    design_matrix['constant'] = [1] * n_sub
    design_matrix['control'] = list(itertools.chain.from_iterable([(groups['group']=='control').values.astype(int)]))
    design_matrix['sport'] = list(itertools.chain.from_iterable([(groups['group']=='sport').values.astype(int)]))
    design_matrix['med'] = list(itertools.chain.from_iterable([(groups['group']=='med').values.astype(int)]))
    #ds_sub = np.eye(len(groups['sub']))
    #ds_subs = pd.DataFrame(ds_sub, columns = groups['sub'])
    #design_matrix = pd.concat((design_matrix, ds_subs), axis=1)
    #design_matrix = pd.concat([design_matrix]*len_contr, ignore_index=True)
    #plot
    ax = plot_design_matrix(design_matrix)
    ax.get_images()[0].set_clim(0, 0.2)
    
    #%Step 3: Second-level GLM
    
    second_level_model = SecondLevelModel(mask_img=fmri_mask, smoothing_fwhm=5.0)
    second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)
    
    contrasts_sl = ['control', 'sport', 'med']     
        
    for contrast_id, contrast_val in enumerate(contrasts_sl):
        print('Results for %s versus two remaining groups' % contrast_val)
        
        z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
        nib.save(z_map, f'{out_dir}/voxel/grey_matter_mask/post/{contrast_val}_post_zmap_{contrast}.nii.gz')
        
        #uncorrecter p<0.001
        thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, 
                                                           height_control=None, 
                                                           cluster_threshold=2)
        s = plotting.plot_stat_map(thresholded_map1, 
                                   threshold=threshold1, colorbar=True)
        save_path = f'{out_dir}/voxel/grey_matter_mask/post/figures/uncorrected'
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
        s.savefig(f'{save_path}/{contrast_val}_post_{contrast}_thresh_zmap_uncorr<0.001.png')
        for section in sections:
            s = plot_stat_map(thresholded_map1, 
                          threshold=threshold1,
                          display_mode=section, 
                          cut_coords=8, 
                          black_bg=False)
            s.savefig(f'{save_path}/{contrast_val}_post_{contrast}_thresh_zmap_uncorr<0.001_{section}.png')
        plotting.show()
        
        thresh_zmap = image.threshold_img(thresholded_map1, threshold=threshold1)
        nib.save(thresh_zmap, 
                 f'{out_dir}/voxel/grey_matter_mask/post/{contrast_val}_post_uncorr_thresh-001_{contrast}.nii.gz')
          
        table = get_clusters_table(z_map, stat_threshold=threshold1,
                                   cluster_threshold=2, min_distance=0)
        
        save_path = f'{out_dir}/voxel/grey_matter_mask/post/tables/uncorrected'
        if not os.path.exists(save_path):
            os.makedirs(save_path)    
        table.to_csv(f'{save_path}/{contrast_val}_post_clusters_table_{contrast}_uncorrected.csv')
        
        #fpr-thresholded map
        #level 0.005 means that there is 0.5% chance of declaring an inactive voxel, active
        thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, 
                                                          height_control='fpr', 
                                                          cluster_threshold=7)
        title = ('Thresholded z map, fpr <.001, clusters > 7 voxels \n'
                 f'{contrast_val} against 2 other groups, post smoothing_fwhm=5.0')
        s = plotting.plot_stat_map(
            thresholded_map1, threshold=threshold1, colorbar=True,
            title=title)
        
        save_path = f'{out_dir}/voxel/grey_matter_mask/post/figures/fpr/'
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        
        s.savefig(f'{save_path}/{contrast_val}_post_{contrast}_thresh_zmap_fpr<0.005.png')
        for section in sections:
            s = plot_stat_map(thresholded_map1, 
                          threshold=threshold1,
                          display_mode=section, 
                          cut_coords=8, 
                          black_bg=False,
                          title=title)
            s.savefig(f'{out_dir}/voxel/grey_matter_mask/post/figures/fpr/{contrast_val}_post_{contrast}_thresh_zmap_fpr<0.005_{section}.png')
        plotting.show()
    
        thresh_zmap = image.threshold_img(thresholded_map1, threshold=threshold1)
        nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/post/{contrast_val}_post_fpr_thresh-001_{contrast}.nii.gz')
        
        table = get_clusters_table(z_map, stat_threshold=threshold1,
                                   cluster_threshold=2, min_distance=0)
        save_path = f'{out_dir}/voxel/grey_matter_mask/post/tables/fpr'
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        save_dir = f'{save_path}/{contrast_val}_post_clusters_table_{contrast}_fpr.csv'    
        table.to_csv(save_dir)
    
        #fdr
        save_path = f'{out_dir}/voxel/grey_matter_mask/post/figures/fdr'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
        title = (f'Thresholded z map, {contrast_val} against 2 other groups {contrast} expected fdr = .05')
        print('The FDR=.05 threshold is %.3g' % threshold2)    
        for section in sections:
            s = plot_stat_map(thresholded_map2, 
                          threshold=threshold2,
                          display_mode=section, 
                          cut_coords=8, 
                          black_bg=False,
                          title=title)
            s.savefig(f'{save_path}/{contrast_val}_post_{contrast}_thresh_zmap_fdr<0.05_{section}.png')
        plotting.show()
        
        thresh_zmap = image.threshold_img(thresholded_map2, threshold=threshold2)
        nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/post/{contrast_val}_post_fdr_thresh-05_{contrast}.nii.gz')
        
        save_path = f'{out_dir}/voxel/grey_matter_mask/post/tables/fdr'
        if not os.path.exists(save_path):
            os.makedirs(save_path)        
        
        table = get_clusters_table(z_map, stat_threshold=threshold2,
                                   cluster_threshold=2, min_distance=0)
        save_dir = f'{save_path}/{contrast_val}_post_clusters_table_{contrast}_fdr.csv'
        table.to_csv(save_dir)
        
        #Bonferroni (fwer) -thresholded map
        save_path = f'{out_dir}/voxel/grey_matter_mask/post/figures/bonferroni'
        if not os.path.exists(save_path):
            os.makedirs(save_path) 
        table.to_csv(save_dir)
        thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, 
                                                           height_control='bonferroni')
        print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
        title = ('Thresholded z map, expected fwer < .05 \n'
                 f'{contrast_val} against 2 other groups, post, smoothing_fwhm=5.0')
        for section in sections:
            s = plot_stat_map(thresholded_map3, 
                          threshold=threshold3,
                          display_mode=section, 
                          cut_coords=8, 
                          black_bg=False,
                          title=title)
            s.savefig(f'{save_path}/{contrast_val}_post_{contrast}_thresh_zmap_bonferroni_{section}.png')
        plotting.show()
        
        thresh_zmap = image.threshold_img(thresholded_map3, threshold=threshold3)
        nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/post/{contrast_val}_post_bonferroni_thresh-05_{contrast}.nii.gz')
        
        table = get_clusters_table(z_map, stat_threshold=threshold3,
                                   cluster_threshold=2, min_distance=0)
        save_path = f'{out_dir}/voxel/grey_matter_mask/post/tables/bonferroni'
        if not os.path.exists(save_path):
            os.makedirs(save_path)         
        save_dir = f'{save_path}/{contrast_val}_post_clusters_table_{contrast}_bonferroni.csv'
        table.to_csv(save_dir)
        
        #% Computing the (corrected) negative log p-values with permutation test
        neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
                                      design_matrix=design_matrix,
                                      second_level_contrast=contrast_val,
                                      model_intercept=True, n_perm=10000,
                                      two_sided_test=False, mask=fmri_mask,
                                      random_state=rand_state,
                                      smoothing_fwhm=5.0, n_jobs=1)
        
        THRESH = -np.log10(0.05)  
        affine = neg_log_pvals_permuted_ols_unmasked.affine
        header = neg_log_pvals_permuted_ols_unmasked.header
        pvals_array = neg_log_pvals_permuted_ols_unmasked.get_fdata()
        z_map_array = z_map.get_fdata()
        z_map_array[pvals_array < THRESH] = 0
        z_map_thresh = nib.Nifti1Image(z_map_array, affine)
        
        #plot the (corrected) negative log p-values
        save_path = f'{out_dir}/voxel/grey_matter_mask/post/figure/permut'
        if not os.path.exists(save_path):
            os.makedirs(save_path)   
        title = (f'Group-level ({contrast_val}) association between \n'
                  'neg-log of non-parametric corrected p-values (FWER < 10%)')
        s = plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True,
                                title=title)
        plotting.show()
        s.savefig(f'{save_path}/{contrast_val}_{contrast}_thresh_pmap_permut.png')
        
        #plot z-map
        s = plotting.plot_stat_map(z_map_thresh, colorbar=True)
        plotting.show()
        s.savefig(f'{save_path}/{contrast_val}_{contrast}_thresh_zmap_permut.png')
        
        for section in sections:
            s = plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                          display_mode=section, 
                          cut_coords=8, 
                          black_bg=False,
                          colorbar=True,
                          annotate=True)
            s.savefig(f'{save_path}/{contrast_val}_{contrast}_thresh_pmap_permut_{section}.png')
        plotting.show()
        
        for section in sections:
            s = plot_stat_map(z_map_thresh, 
                          display_mode=section, 
                          cut_coords=8, 
                          black_bg=False,
                          colorbar=True,
                          annotate=True)
            s.savefig(f'{save_path}/{contrast_val}_{contrast}_thresh_zmap_permut_{section}.png')
        plotting.show()
    
        nib.save(neg_log_pvals_permuted_ols_unmasked, 
             f'{out_dir}/voxel/grey_matter_mask/post/{contrast_val}_post_permuted_{contrast}_pmap.nii.gz')

        nib.save(z_map_thresh, 
             f'{out_dir}/voxel/grey_matter_mask/post/{contrast_val}_post_permuted_{contrast}_thresh_zmap.nii.gz')
    
        save_path = f'{out_dir}/voxel/grey_matter_mask/post/tables/permut'
        if not os.path.exists(save_path):
            os.makedirs(save_path)  
        table = get_clusters_table(z_map_thresh, stat_threshold=0,
                                   cluster_threshold=2, min_distance=0)
        save_dir = f'{save_path}/{contrast_val}_clusters_table_{contrast}_neg_log_pvals_permuted_ols.csv'
        table.to_csv(save_dir)
        
        # The neg-log p-values obtained with nonparametric testing are capped at 3
        # since the number of permutations is 1e3. 
   
#%%Post t-test med vs sport smoothing_fwhm=5.0 second-level model

subs_sport = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport'])])
nsubs_sport = len(subs_sport)

subs_med = pd.Series.tolist(groups['sub'][groups['group'].isin(['med'])])
nsubs_med = len(subs_med)

for k, contrast in enumerate(contr_fl_interest):
    zmaps_sport = []
    for i, sub in enumerate(subs_sport):
        zmaps_sport.append(f'{top_dir}/GLM/FLA/voxel/{sub}_ses-3_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
    
    zmaps_med = []
    for i, sub in enumerate(subs_med):
        zmaps_med.append(f'{top_dir}/GLM/FLA/voxel/{sub}_ses-3_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
    
    zmaps = zmaps_sport + zmaps_med 
    
    #%Step 2: Create second-level design matrix
    # 1's per subject model out each subj mean
    n_sub = nsubs_sport + nsubs_med
    len_contr = len(contr_fl_interest)
    design_matrix = pd.DataFrame()
    design_matrix['sport'] = [1] * nsubs_sport + [0] * nsubs_med
    design_matrix['med'] = [0] * nsubs_sport + [1] * nsubs_med

    ax = plot_design_matrix(design_matrix)
    ax.get_images()[0].set_clim(0, 0.2)
    
    #%Step 3: Second-level GLM
    
    second_level_model = SecondLevelModel(mask_img=fmri_mask, smoothing_fwhm=5.0)
    second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)
    
    contrast_val = 'med'  
        
    #for contrast_id, contrast_val in enumerate(contrasts_sl):
    print('Results for med versus sport group')
    
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    nib.save(z_map, f'{out_dir}/voxel/grey_matter_mask/post_1vs1/{contrast_val}_post_zmap_{contrast}.nii.gz')
    
    #uncorrecter p<0.001
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, 
                                                       height_control=None, 
                                                       cluster_threshold=2)
    s = plotting.plot_stat_map(thresholded_map1, 
                               threshold=threshold1, colorbar=True)
    save_path = f'{out_dir}/voxel/grey_matter_mask/post_1vs1/figures/uncorrected'
    if not os.path.exists(save_path):
        os.makedirs(save_path)   
    s.savefig(f'{save_path}/post_{contrast}_thresh_zmap_uncorr<0.001.png')
    for section in sections:
        s = plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False)
        s.savefig(f'{save_path}/post_{contrast}_thresh_zmap_uncorr<0.001_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map1, threshold=threshold1)
    nib.save(thresh_zmap, 
             f'{out_dir}/voxel/grey_matter_mask/post_1vs1/post_uncorr_thresh-001_{contrast}.nii.gz')
      
    table = get_clusters_table(z_map, stat_threshold=threshold1,
                               cluster_threshold=2, min_distance=0)
    
    save_path = f'{out_dir}/voxel/grey_matter_mask/post_1vs1/tables/uncorrected'
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    table.to_csv(f'{save_path}/post_clusters_table_{contrast}_uncorrected.csv')
    
    #fpr-thresholded map
    #level 0.005 means that there is 0.5% chance of declaring an inactive voxel, active
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=7)
    title = ('Thresholded z map, fpr <.001, clusters > 7 voxels \n'
             f'{contrast_val} against 2 other groups, post smoothing_fwhm=5.0')
    s = plotting.plot_stat_map(
        thresholded_map1, threshold=threshold1, colorbar=True,
        title=title)
    
    save_path = f'{out_dir}/voxel/grey_matter_mask/post_1vs1/figures/fpr'
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    s.savefig(f'{save_path}/post_{contrast}_thresh_zmap_fpr<0.005.png')
    for section in sections:
        s = plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{save_path}/post_{contrast}_thresh_zmap_fpr<0.005_{section}.png')
    plotting.show()

    thresh_zmap = image.threshold_img(thresholded_map1, threshold=threshold1)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/post_1vs1/post_fpr_thresh-001_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold1,
                               cluster_threshold=2, min_distance=0)
    
    save_path = f'{out_dir}/voxel/grey_matter_mask/post_1vs1/tables/fpr'
    if not os.path.exists(save_path):
        os.makedirs(save_path)     
    save_dir = f'{save_path}/post_clusters_table_{contrast}_fpr.csv'
    table.to_csv(save_dir)

    #fdr
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    title = (f'Thresholded z map, {contrast_val} against 2 other groups {contrast} expected fdr = .05')
    print('The FDR=.05 threshold is %.3g' % threshold2)  
    save_path = f'{out_dir}/voxel/grey_matter_mask/post_1vs1/figures/fdr'
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    for section in sections:
        s = plot_stat_map(thresholded_map2, 
                      threshold=threshold2,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{save_path}/post_{contrast}_thresh_zmap_fdr<0.05_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map2, threshold=threshold2)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/post_1vs1/post_fdr_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold2,
                               cluster_threshold=2, min_distance=0)
    save_path = f'{out_dir}/voxel/grey_matter_mask/post_1vs1/tables/fpr'
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    save_dir = f'{save_path}/post_clusters_table_{contrast}_fdr.csv'
    table.to_csv(save_dir)
    
    #Bonferroni (fwer) -thresholded map
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    title = ('Thresholded z map, expected fwer < .05 \n'
             f'{contrast_val} against 2 other groups, post, smoothing_fwhm=5.0')
    save_path = f'{out_dir}/voxel/grey_matter_mask/post_1vs1/figures/bonferroni'
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    for section in sections:
        s = plot_stat_map(thresholded_map3, 
                      threshold=threshold3,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{save_path}/post_{contrast}_thresh_zmap_bonferroni_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map3, threshold=threshold3)
    nib.save(thresh_zmap, 
             f'{out_dir}/voxel/grey_matter_mask/post_1vs1/post_bonferroni_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold3,
                               cluster_threshold=2, min_distance=0)
    save_path = f'{out_dir}/voxel/grey_matter_mask/post_1vs1/tables/bonferroni'
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    save_dir = f'{save_path}/post_clusters_table_{contrast}_bonferroni.csv'
    table.to_csv(save_dir)
    
    #% Computing the (corrected) negative log p-values with permutation test
    neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
                                  design_matrix=design_matrix,
                                  second_level_contrast=contrast_val,
                                  model_intercept=True, n_perm=10000,
                                  two_sided_test=False, mask=fmri_mask,
                                  random_state=rand_state,
                                  smoothing_fwhm=5.0, n_jobs=1)
    
    THRESH = -np.log10(0.05)  
    affine = neg_log_pvals_permuted_ols_unmasked.affine
    header = neg_log_pvals_permuted_ols_unmasked.header
    pvals_array = neg_log_pvals_permuted_ols_unmasked.get_fdata()
    z_map_array = z_map.get_fdata()
    z_map_array[pvals_array < THRESH] = 0
    z_map_thresh = nib.Nifti1Image(z_map_array, affine)
        
    #plot the (corrected) negative log p-values
    title = (f'Group-level ({contrast_val}) association between \n'
              'neg-log of non-parametric corrected p-values (FWER < 10%)')
    s = plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True,
                            )
    plotting.show()
    save_path = f'{out_dir}/voxel/grey_matter_mask/post_1vs1/figures/permut'
    if not os.path.exists(save_path):
        os.makedirs(save_path) 
    s.savefig(f'{save_path}/post_{contrast}_thresh_pmap_permut.png')
    
    #plot z-map
    s = plotting.plot_stat_map(z_map_thresh, colorbar=True)
    plotting.show()
    s.savefig(f'{save_path}/post_{contrast}_thresh_zmap_permut.png')
    
    for section in sections:
        s = plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      colorbar=True,
                      annotate=True)
        s.savefig(f'{save_path}/post_{contrast}_thresh_pmap_permut_{section}.png')
    plotting.show()
    
    for section in sections:
        s = plot_stat_map(z_map_thresh, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      colorbar=True,
                      annotate=True)
        s.savefig(f'{save_path}/post_{contrast}_thresh_zmap_permut_{section}.png')
    plotting.show()

    nib.save(neg_log_pvals_permuted_ols_unmasked, 
         f'{out_dir}/voxel/grey_matter_mask/post_1vs1/permuted_{contrast}_pmap.nii.gz')

    nib.save(z_map_thresh, 
         f'{out_dir}/voxel/grey_matter_mask/post_1vs1/permuted_{contrast}_thresh_zmap.nii.gz')

    save_path = f'{out_dir}/voxel/grey_matter_mask/post_1vs1/tables/permut'
    if not os.path.exists(save_path):
        os.makedirs(save_path)  
    table = get_clusters_table(z_map_thresh, stat_threshold=0,
                               cluster_threshold=2, min_distance=0)
    save_dir = f'{save_path}/post_clusters_table_{contrast}_neg_log_pvals_permuted_ols.csv'
    table.to_csv(save_dir)

    # The neg-log p-values obtained with nonparametric testing are capped at 3
    # since the number of permutations is 1e3. 




































