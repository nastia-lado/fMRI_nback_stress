#FLA from BIDS

import sys
sys.path.append("..")
from os import path

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nibabel as nib

from nilearn import input_data, datasets
from nilearn import plotting
from nilearn.plotting import plot_stat_map, plot_anat, plot_img, show, plot_glass_brain
from nistats.first_level_model import FirstLevelModel
from nistats.reporting import plot_design_matrix, plot_contrast_matrix
from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import first_level_models_from_bids

from fctools import denoise

#%%Setup
data_dir = '/home/alado/datasets/RBH/preprocessed/'
raw_data_dir = '/home/alado/datasets/RBH/Nifti'
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/GLM/FLA/'
suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
mask_suffix = '_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz'

groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])
subjects_data_clean = groups[groups['group'].isin(['sport', 'med', 'control'])].reset_index()
subjects = subjects_data_clean['sub']

sess = ['ses-1', 'ses-3']
task = 'nback'
t_r = 3.0

#%%
sub = 'sub-02'
ses = 'ses-1'

sub_dir = f'{top_dir}/preprocessed/derivatives/fmriprep/{sub}/{ses}/func/'
sub_name = f'{sub}_{ses}_task-nback_run-1' 
events_dir = f'{top_dir}/Nifti/{sub}/{ses}/func/'
fmri_img_path = f'{sub_dir}{sub_name}{suffix}'
fmri_mask_path = f'{sub_dir}{sub_name}{mask_suffix}'

fmri_img = nib.load(fmri_img_path)
fmri_mask = nib.load(fmri_mask_path)
from nilearn.image import mean_img
mean_img_ = mean_img(fmri_img)


#%%
task_label = 'nback'
space_label = 'MNI152NLin2009cAsym'
derivatives_folder = 'derivatives/fmriprep'
models, models_run_imgs, models_events, models_confounds = first_level_models_from_bids(
        data_dir, task_label, space_label, 
        #smoothing_fwhm=5.0, 
        derivatives_folder=derivatives_folder
        )

#%%get rid of inf and nan
for i in range(len(models_confounds)):
    for j in range(2):
        models_confounds[i][j] = models_confounds[i][j].replace([np.inf, -np.inf], np.nan)
        models_confounds[i][j] = models_confounds[i][j].fillna(0)

#%%
import os
print([os.path.basename(run) for run in models_run_imgs[0]])

print(models_confounds[0][0].columns)

print(models_events[0][0]['trial_type'].value_counts())
#%%
#do somwthing about the columns in the design matrix


#%% First level model estimation
#Set the threshold as the z-variate with an uncorrected p-value of 0.001.

from scipy.stats import norm
p001_unc = norm.isf(0.001)

from nilearn import plotting
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(8, 4.5))
model_and_args = zip(models, models_run_imgs, models_events, models_confounds)
for midx, (model, imgs, events, confounds) in enumerate(model_and_args):
    # fit the GLM
    model.fit(imgs, events, confounds)
    design_matrix = model.design_matrices_[0]
    design_matrix2 = model.design_matrices_[1]
    n_columns = len(design_matrix2.columns)
    # contrasts = {'0back-1back': np.pad([1, -1, 0, 0, 0], (0,n_columns-5)),
    #               '0back-2back': np.pad([1, 0, -1, 0, 0], (0,n_columns-5)),
    #               '0back-3back': np.pad([1, 0, 0, -1, 0], (0,n_columns-5)),
    #               '0back-fix': np.pad([1, 0, 0, 0, -1], (0,n_columns-5)),
    #               '1back-2back': np.pad([0, 1, -1, 0, 0], (0,n_columns-5)),
    #               '1back-4back': np.pad([0, 1, 0, -1, 0], (0,n_columns-5)),
    #               '1back-fix': np.pad([0, 1, 0, 0, -1], (0,n_columns-5)),
    #               '2back-4back': np.pad([0, 0, 1, -1, 0], (0,n_columns-5)),
    #               '2back-fix': np.pad([0, 0, 1, 0, -1], (0,n_columns-5)),
    #               '4back-fix': np.pad([0, 0, 0, 1, -1], (0,n_columns-5))
    #               }
    
    # compute the contrast of interest
    zmap = model.compute_contrast(np.pad([1, -1, 0, 0, 0], (0,n_columns-5)))
    plotting.plot_glass_brain(zmap, colorbar=False, threshold=p001_unc,
                              title=('sub-' + model.subject_label),
                              axes=axes[int(midx / 5), int(midx % 5)],
                              plot_abs=False, display_mode='x')
fig.suptitle('subjects z_map (unc p<0.001)')
plotting.show()


#%%
model_1sub = models[0]

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(8, 4.5))
model_and_args = zip(models, models_run_imgs, models_events, models_confounds)
for midx, (model, imgs, events, confounds) in enumerate(model_and_args):
    for i in range(2):
    # fit the GLM
        fmri_img = imgs[i]
        event = events[i]
        confs = confounds[i]
        #load mask into model here
        model.mask_img = nib.load(fmri_mask)
        
        model.fit(fmri_img, event, confs)
        design_matrix = model.design_matrices_[0]
        n_columns = len(design_matrix.columns)
        contrasts = {'0back-1back': np.pad([1, -1, 0, 0, 0], (0,n_columns-5)),
                  '0back-2back': np.pad([1, 0, -1, 0, 0], (0,n_columns-5)),
                  '0back-3back': np.pad([1, 0, 0, -1, 0], (0,n_columns-5)),
                  '0back-fix': np.pad([1, 0, 0, 0, -1], (0,n_columns-5)),
                  '1back-2back': np.pad([0, 1, -1, 0, 0], (0,n_columns-5)),
                  '1back-4back': np.pad([0, 1, 0, -1, 0], (0,n_columns-5)),
                  '1back-fix': np.pad([0, 1, 0, 0, -1], (0,n_columns-5)),
                  '2back-4back': np.pad([0, 0, 1, -1, 0], (0,n_columns-5)),
                  '2back-fix': np.pad([0, 0, 1, 0, -1], (0,n_columns-5)),
                  '4back-fix': np.pad([0, 0, 0, 1, -1], (0,n_columns-5))
                  }
        
        print('Computing contrasts...')
        for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
            print('Contrast % 2i out of %i: %s' % (index + 1, len(contrasts), contrast_id))
            # estimate the contasts
            # note that the model implictly computes a fixed effect across the two sessions
            #or here z_map = fmri_glm
            z_map = model_1sub.compute_contrast(contrast_val, output_type='z_score')
            
            plotting.plot_stat_map(z_map)
            plotting.show()
            
            #summary_statistics_session1 = fmri_glm.compute_contrast(contrast_val, output_type='all')    
            plotting.plot_stat_map(
            z_map, bg_img=mean_img_, 
            threshold=2.0,
            #cut_coords=cut_coords,
            title='%s, first session' % contrast_id)
        
            # write the resulting stat images to file
            z_image_path = path.join(out_dir, '%s_z_map.nii.gz' % contrast_id)
            z_map.to_filename(z_image_path)
        
            # Saving z_maps
            nib.save(z_map, f'{out_dir}/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast_id}.nii.gz')
        
            g = plot_glass_brain(z_map, colorbar=True, plot_abs=False, title=f'{sub}, {ses}')
            g.savefig(f'{out_dir}/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast_id}_glass_brain.png')
        
            s = plot_stat_map(z_map, threshold=3, title=f'{sub}, {ses}')
            s.savefig(f'{out_dir}/{sub}_{ses}_space-MNI152NLin2009cAsym_0b_minus_1b_zmap_slices.png')

#%%
design_matrix = model_1sub.design_matrices_[0]
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)

#glm = glm.fit(fmri_img, design_matrix)

#r`2: returns a list, take the first element, because we only have one run
r2_img = model_1sub.r_square[0]
plotting.plot_stat_map(r2_img, threshold=0.2)
plotting.show()

resids = model_1sub.residuals[0]
n_columns = len(design_matrix.columns)

#'Effects_of_interest': np.eye(n_columns)[:5]
#10 contrasts
contrasts = {'0back-1back': np.pad([1, -1, 0, 0, 0], (0,n_columns-5)),
                  '0back-2back': np.pad([1, 0, -1, 0, 0], (0,n_columns-5)),
                  '0back-3back': np.pad([1, 0, 0, -1, 0], (0,n_columns-5)),
                  '0back-fix': np.pad([1, 0, 0, 0, -1], (0,n_columns-5)),
                  '1back-2back': np.pad([0, 1, -1, 0, 0], (0,n_columns-5)),
                  '1back-4back': np.pad([0, 1, 0, -1, 0], (0,n_columns-5)),
                  '1back-fix': np.pad([0, 1, 0, 0, -1], (0,n_columns-5)),
                  '2back-4back': np.pad([0, 0, 1, -1, 0], (0,n_columns-5)),
                  '2back-fix': np.pad([0, 0, 1, 0, -1], (0,n_columns-5)),
                  '4back-fix': np.pad([0, 0, 0, 1, -1], (0,n_columns-5))
                  }

# contrasts = {'0back': np.pad([1, 0, 0, 0, 0], (0,n_columns-5)),
#                   '2back': np.pad([0, 1, 0, 0, 0], (0,n_columns-5)),
#                   '3back': np.pad([0, 0, 1, 0, 0], (0,n_columns-5)),
#                   '4back': np.pad([0, 0, 0, 1, 0], (0,n_columns-5)),                  
#                   'fix': np.pad([0, 0, 0, 0, 1], (0,n_columns-5))
#                   }

print('Computing contrasts...')
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('Contrast % 2i out of %i: %s' % (index + 1, len(contrasts), contrast_id))
    # estimate the contasts
    # note that the model implictly computes a fixed effect across the two sessions
    #or here z_map = fmri_glm
    z_map = model_1sub.compute_contrast(contrast_val, output_type='z_score')
    
    plotting.plot_stat_map(z_map)
    plotting.show()
    
    #summary_statistics_session1 = fmri_glm.compute_contrast(contrast_val, output_type='all')    
    plotting.plot_stat_map(
    z_map, bg_img=mean_img_, 
    threshold=2.0,
    #cut_coords=cut_coords,
    title='%s, first session' % contrast_id)

    # write the resulting stat images to file
    z_image_path = path.join(out_dir, '%s_z_map.nii.gz' % contrast_id)
    z_map.to_filename(z_image_path)

    # Saving z_maps
    nib.save(z_map, f'{out_dir}/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast_id}.nii.gz')

    g = plot_glass_brain(z_map, colorbar=True, plot_abs=False, title=f'{sub}, {ses}')
    g.savefig(f'{out_dir}/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast_id}_glass_brain.png')

    s = plot_stat_map(z_map, threshold=3, title=f'{sub}, {ses}')
    s.savefig(f'{out_dir}/{sub}_{ses}_space-MNI152NLin2009cAsym_0b_minus_1b_zmap_slices.png')

#np.save(f'{out_dir}GLM_power_0b_minus_1b_zmap_timeseries.npy', timeseries)


























