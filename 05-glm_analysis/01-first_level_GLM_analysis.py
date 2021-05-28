#First-level GLM analysis of working memory/stress fMRI data
#Supplementary analysis - standard GLM analysis of activation patterns changes between 0-, 1-, 2-, 4-back, and fixation.

#First level n-back task GLM analysis performed for each subject and sessions 1 and 3.

#contrasts of interest:
#included confounds regressors analogical to fc (aCompCor, 24 motion parameters, outlier scans)
#Outputs:

#z-maps for each subject and session

import sys
sys.path.append("..")

import pandas as pd
import numpy as np
import nibabel as nib

from nilearn import plotting
from nilearn import masking
from nilearn.image import mean_img, resample_to_img, concat_imgs
from nilearn.plotting import plot_stat_map, plot_anat, plot_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.datasets import fetch_icbm152_brain_gm_mask
from nilearn.glm.contrasts import compute_fixed_effects

from fctools import denoise

#%% compute_temporal_derivatives function
#The temporal derivatives are defined as the difference between consecutive timepoints
def compute_temporal_derivatives(confound_df):
    for col in confound_df.columns:
    
        #Example X --> X_dt
        new_name = '{}_dt'.format(col)
    
        #Compute differences for each pair of rows from start to end.
        new_col = confound_df[col].diff()
    
        #Make new column in our dataframe
        confound_df[new_name] = new_col
    
    confound_df.head()
    
    return confound_df

#%%Dummy TR Drop
#we are removing the first 4 timepoints from our functional image
#we’ll also have to do this for our first 4 confound timepoints

#First we'll load in our data and check the shape
raw_func_img = img.load_img(func)
raw_func_img.shape

#the fourth dimension represents frames/TRs(timepoints)
#We want to drop the first four timepoints entirely, to do so we use nibabel’s slicer feature.
#We’ll also drop the first 4 confound variable timepoints to match the functional scan

#Get all timepoints after the 4th
func_img = raw_func_img.slicer[:,:,:,5:]
func_img.shape

#Drop confound dummy TRs from the dataframe to match the size of our new func_img
drop_confound_df = confound_df.loc[5:]
print(drop_confound_df.shape) #number of rows should match that of the functional image
drop_confound_df.head()


#%%Setup
data_dir = '/home/alado/datasets/RBH/preprocessed/derivatives/fmriprep'
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/GLM/FLA/'
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
        events_dir = f'{top_dir}/Nifti/{sub}/{ses}/func/'
        
        sub_name = f'{sub}_{ses}_task-nback_run-1'
        fmri_img_path = f'{sub_dir}{sub_name}{suffix}'
        fmri_img_1 = nib.load(fmri_img_path)
        events_path = f'{events_dir}{sub_name}_events.tsv'
        events_1 = pd.read_csv(events_path, delimiter='\t')
        #events = events.loc[:, ['onset', 'duration', 'trial_type']]
        # create a "modulation" column with only ones (indicating no modulation)
        #events.loc[:, 'modulation'] = 1  # this automatically fills all rows with 1
        confounds_path = f'{sub_dir}{sub_name}_desc-confounds_regressors.tsv'
        confounds_1 = pd.read_csv(confounds_path, delimiter='\t')
        
        sub_name = f'{sub}_{ses}_task-nback_run-2'
        fmri_img_path = f'{sub_dir}{sub_name}{suffix}'
        fmri_img_2 = nib.load(fmri_img_path)
        events_path = f'{events_dir}{sub_name}_events.tsv'
        events_2 = pd.read_csv(events_path, delimiter='\t')
        confounds_path = f'{sub_dir}{sub_name}_desc-confounds_regressors.tsv'
        confounds_2 = pd.read_csv(confounds_path, delimiter='\t')
        
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
        confounds_motion_1 = confounds_1[confounds_1.filter(regex='trans_x|trans_y|trans_z|rot_x|rot_y|rot_z').columns]
        confounds_acompcor_1 = confounds_1[confounds_1.filter(regex='a_comp_cor').columns]
        confounds_acompcor_1 = confounds_acompcor_1.drop(confounds_acompcor_1.columns[3:], axis=1)
        #include either here or in the drift_model='cosine' in first level model
        #confounds_cosine = confounds[confounds.filter(regex='cosine').columns]
        confounds_scrub_1 = confounds_1[confounds_1.filter(regex='std_dvars|framewise_displacement').columns]
        # Detecting outliers 
        confounds_scrub_td_1 = denoise.temp_deriv(denoise.outliers_fd_dvars(confounds_scrub_1, fd=0.5, dvars=3), quadratic=False)
        # Concatenating columns
        confounds_clean_1 = pd.concat([confounds_motion_1, 
                                 confounds_acompcor_1,
                                 #confounds_cosine,
                                 confounds_scrub_td_1], 
                                 axis=1)
        confounds_2 = confounds_2.replace([np.inf, -np.inf], np.nan)
        confounds_2 = confounds_2.fillna(0)
        # Selecting columns of interest 
        confounds_motion_2 = confounds_2[confounds_2.filter(regex='trans_x|trans_y|trans_z|rot_x|rot_y|rot_z').columns]
        confounds_acompcor_2 = confounds_2[confounds_2.filter(regex='a_comp_cor').columns]
        confounds_acompcor_2 = confounds_acompcor_2.drop(confounds_acompcor_2.columns[3:], axis=1)
        #include either here or in the drift_model='cosine' in first level model
        #confounds_cosine = confounds[confounds.filter(regex='cosine').columns]
        confounds_scrub_2 = confounds_1[confounds_1.filter(regex='std_dvars|framewise_displacement').columns]
        # Detecting outliers 
        confounds_scrub_td_2 = denoise.temp_deriv(denoise.outliers_fd_dvars(confounds_scrub_2, fd=0.5, dvars=3), quadratic=False)
        
        # Concatenating columns
        confounds_clean_2 = pd.concat([confounds_motion_2, 
                                 confounds_acompcor_2,
                                 #confounds_cosine,
                                 confounds_scrub_td_2], 
                                 axis=1)
        
        #add separate signal cleaning?
        
        
        #Setup GLM model
        glm = FirstLevelModel(
            t_r=t_r,   
            slice_time_ref = 0.5, #bc fmriprep realings in time to the middle of each TR
            hrf_model='glover',
            drift_model='cosine',
            high_pass=0.008,
            mask_img = fmri_mask, #nifti image
            #smoothing_fwhm=3.5,   
            noise_model='ar1',
            #detrend=True #YEO analysis
            standardize=False, #better True for stat analysis of connectivity!
            minimize_memory=False)
  
        #masker = fmri_mask.NIftiMasker()
        
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
        
        glm_1 = glm.fit(fmri_img_1, events=events_1, confounds=confounds_clean_1)
        glm_2 = glm.fit(fmri_img_2, events=events_2, confounds=confounds_clean_2)
        
        # the design_matrices_ attribute is a list with, in our case, only a single element
        design_matrix_1 = glm_1.design_matrices_[0]
        design_matrix_2 = glm_2.design_matrices_[0]
        
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
        contrasts_1 = {'0back-1back': np.pad([1, -1, 0, 0, 0], (0,n_columns-5)),
                    '0back-2back': np.pad([1, 0, -1, 0, 0], (0,n_columns-5)),
                    '0back-4back': np.pad([1, 0, 0, -1, 0], (0,n_columns-5)),
                    '0back-fix': np.pad([1, 0, 0, 0, -1], (0,n_columns-5)),
                    '1back-0back': np.pad([-1, 1, 0, 0, 0], (0,n_columns-5)),
                    '1back-2back': np.pad([0, 1, -1, 0, 0], (0,n_columns-5)),
                    '1back-4back': np.pad([0, 1, 0, -1, 0], (0,n_columns-5)),
                    '1back-fix': np.pad([0, 1, 0, 0, -1], (0,n_columns-5)),
                    '2back-0back': np.pad([-1, 0, 1, 0, 0], (0,n_columns-5)),
                    '2back-1back': np.pad([0, -1, 1, 0, 0], (0,n_columns-5)),
                    '2back-4back': np.pad([0, 0, 1, -1, 0], (0,n_columns-5)),
                    '2back-fix': np.pad([0, 0, 1, 0, -1], (0,n_columns-5)),
                    '4back-0back': np.pad([-1, 0, 0, 1, 0], (0,n_columns-5)),
                    '4back-1back': np.pad([0, -1, 0, 1, 0], (0,n_columns-5)),
                    '4back-2back': np.pad([0, 0, -1, 1, 0], (0,n_columns-5)), 
                    '4back-fix': np.pad([0, 0, 0, 1, -1], (0,n_columns-5)),
                    '0back': np.pad([1, 0, 0, 0, 0], (0,n_columns-5)),
                    '1back': np.pad([0, 1, 0, 0, 0], (0,n_columns-5)),
                    '2back': np.pad([0, 0, 1, 0, 0], (0,n_columns-5)),
                    '4back': np.pad([0, 0, 0, 1, 0], (0,n_columns-5)),                  
                    'fix': np.pad([0, 0, 0, 0, 1], (0,n_columns-5)),
                    'f-contrast': np.eye(n_columns)[:5]
                    }
        
        n_columns = len(design_matrix_2.columns)
        contrasts_2 = {'0back-1back': np.pad([1, -1, 0, 0, 0], (0,n_columns-5)),
                    '0back-2back': np.pad([1, 0, -1, 0, 0], (0,n_columns-5)),
                    '0back-4back': np.pad([1, 0, 0, -1, 0], (0,n_columns-5)),
                    '0back-fix': np.pad([1, 0, 0, 0, -1], (0,n_columns-5)),
                    '1back-0back': np.pad([-1, 1, 0, 0, 0], (0,n_columns-5)),
                    '1back-2back': np.pad([0, 1, -1, 0, 0], (0,n_columns-5)),
                    '1back-4back': np.pad([0, 1, 0, -1, 0], (0,n_columns-5)),
                    '1back-fix': np.pad([0, 1, 0, 0, -1], (0,n_columns-5)),
                    '2back-0back': np.pad([-1, 0, 1, 0, 0], (0,n_columns-5)),
                    '2back-1back': np.pad([0, -1, 1, 0, 0], (0,n_columns-5)),
                    '2back-4back': np.pad([0, 0, 1, -1, 0], (0,n_columns-5)),
                    '2back-fix': np.pad([0, 0, 1, 0, -1], (0,n_columns-5)),
                    '4back-0back': np.pad([-1, 0, 0, 1, 0], (0,n_columns-5)),
                    '4back-1back': np.pad([0, -1, 0, 1, 0], (0,n_columns-5)),
                    '4back-2back': np.pad([0, 0, -1, 1, 0], (0,n_columns-5)), 
                    '4back-fix': np.pad([0, 0, 0, 1, -1], (0,n_columns-5)),
                    '0back': np.pad([1, 0, 0, 0, 0], (0,n_columns-5)),
                    '1back': np.pad([0, 1, 0, 0, 0], (0,n_columns-5)),
                    '2back': np.pad([0, 0, 1, 0, 0], (0,n_columns-5)),
                    '4back': np.pad([0, 0, 0, 1, 0], (0,n_columns-5)),                  
                    'fix': np.pad([0, 0, 0, 0, 1], (0,n_columns-5)),
                    'f-contrast': np.eye(n_columns)[:5]
                    }

        
        print('Computing contrasts...')
        for index, (contrast_id, contrast_val) in enumerate(contrasts_1.items()):
            print('Contrast % 2i out of %i: %s' % (index + 1, len(contrasts_1), contrast_id))
            # estimate the contasts
            # note that the model implictly computes a fixed effect across the two sessions
            #summary_stats = glm.compute_contrast(contrast_val, output_type='all')
            summary_statistics_run1 = glm_1.compute_contrast(contrast_val, output_type='all')
            summary_statistics_run2 = glm_2.compute_contrast(contrast_val, output_type='all')
            
            contrast_imgs = [summary_statistics_run1['effect_size'], summary_statistics_run2['effect_size']]
            variance_imgs = [summary_statistics_run1['effect_variance'], summary_statistics_run2['effect_variance']]
            
            fixed_fx_contrast, fixed_fx_variance, fixed_fx_stat = compute_fixed_effects(contrast_imgs, variance_imgs, fmri_mask)
            #plotting.plot_stat_map(fixed_fx_stat, bg_img=mean_img_, threshold=3.0, cut_coords=cut_coords,
            #    title='{0}, fixed effects'.format(contrast_id))
            
            #Not unexpectedly, the fixed effects version displays higher peaks than the input sessions.
            #Computing fixed effects enhances the signal-to-noise ratio of the resulting brain maps.
            #Note however that, technically, the output maps of the fixed effects map is a t statistic
            #(not a z statistic)
            #t to z conversion
            
            #z_map = glm.compute_contrast(contrast_val, output_type='z_score') if there is one run
            #plotting.plot_stat_map(z_map, bg_img=mean_img_, threshold=2.0,title='%s' % contrast_id)
            #timeseries[i,j,:] = masker.fit_transform(z_map)

            # Saving z_maps
            #or here we save the beta estimates of the fixed effects for the 2nd level
            nib.save(fixed_fx_contrast, f'{out_dir}/voxel/grey_matter_mask/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast_id}.nii.gz')
            
            #Saving all outputs
            #np.save(summary_stats, f'{out_dir}/voxel/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast_id}_summary_stats.npy')
            
            #g = plot_glass_brain(z_map, colorbar=True, plot_abs=False, title=f'{sub}, {ses}')
            #g.savefig(f'{out_dir}/voxel/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast_id}_glass_brain.png')
        
            #s = plot_stat_map(z_map, threshold=3, title=f'{sub}, {ses}')
            #s.savefig(f'{out_dir}/voxel/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast_id}_slices.png')
        
        #np.save(f'{out_dir}GLM_power_0b_minus_1b_zmap_timeseries.npy', timeseries)
        
            #save resampled mask
            nib.save(fmri_mask, f'/home/alado/datasets/RBH/GLM/resampled_masks/{sub}_{ses}_resampled_icbm_mask.nii.gz')

