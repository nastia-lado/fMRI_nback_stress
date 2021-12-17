#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 14:01:30 2021

@author: alado

Second-level only permutations
"""
"""
Created on Wed Dec  9 17:39:41 2020

@author: alado
"""

#Working memory training: Second level GLM analysis of working memory training fMRI data
#Second-level GLM analysis of activation patterns

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
    
    #% Computing the (corrected) negative log p-values with permutation test
    neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
                                  design_matrix=design_matrix,
                                  second_level_contrast=contrast_val,
                                  model_intercept=True, n_perm=10000,
                                  two_sided_test=False, mask=fmri_mask,
                                  smoothing_fwhm=5.0, n_jobs=1)
        
    #plot the (corrected) negative log p-values
    title = ('Group-level association between \n'
              'neg-log of non-parametric corrected p-values (FWER < 10%)')
    s = plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True, vmax=3,
                            title=title)
    plotting.show()
    save_path = f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/permut'
    if not os.path.exists(save_path):
        os.makedirs(save_path)       
    s.savefig(f'{save_path}/sport_{contrast}_thresh_zmap_permut.png')
    
    for section in sections:
        plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{save_path}/sport_{contrast}_thresh_zmap_permut_{section}.png')
    plotting.show()

    nib.save(neg_log_pvals_permuted_ols_unmasked, f'{out_dir}/voxel/grey_matter_mask/pre_post/sport_pre_post_permuted_{contrast}.nii.gz')
    
    # Since we are plotting negative log p-values and using a threshold equal to 0.5,
    # it corresponds to corrected p-values lower than 5%, meaning that there
    # is less than 5% probability to make a single false discovery
    # (95% chance that we make no false discovery at all).
    table = get_clusters_table(neg_log_pvals_permuted_ols_unmasked, stat_threshold=0.5,
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
    
    #% Computing the (corrected) negative log p-values with permutation test
    neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
                                  design_matrix=design_matrix,
                                  second_level_contrast=contrast_val,
                                  model_intercept=True, n_perm=10000,
                                  two_sided_test=False, mask=fmri_mask,
                                  smoothing_fwhm=5.0, n_jobs=1)
        
    #plot the (corrected) negative log p-values
    title = ('Group-level association between \n'
              'neg-log of non-parametric corrected p-values (FWER < 10%)')
    s = plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True, vmax=3,
                            title=title)
    plotting.show()
    s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/permut/med_{contrast}_thresh_zmap_permut.png')
    
    for section in sections:
        plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/figures/permut/med_{contrast}_thresh_zmap_permut_{section}.png')
    plotting.show()

    nib.save(neg_log_pvals_permuted_ols_unmasked, f'{out_dir}/voxel/grey_matter_mask/pre_post/med_pre_post_permuted_{contrast}.nii.gz')
     
    table = get_clusters_table(neg_log_pvals_permuted_ols_unmasked, stat_threshold=0.5,
                               cluster_threshold=2, min_distance=0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/tables/permut/med_clusters_table_{contrast}_neg_log_pvals_permuted_ols.csv'
    table.to_csv(save_dir)
    
    # The neg-log p-values obtained with nonparametric testing are capped at 3
    # since the number of permutations is 1e3. 

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
    
        #% Computing the (corrected) negative log p-values with permutation test
        neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
                                      design_matrix=design_matrix,
                                      second_level_contrast=contrast_val,
                                      model_intercept=True, n_perm=10000,
                                      two_sided_test=False, mask=fmri_mask,
                                      smoothing_fwhm=5.0, n_jobs=1)
            
        #plot the (corrected) negative log p-values
        title = (f'Group-level ({contrast_val}) association between \n'
                  'neg-log of non-parametric corrected p-values (FWER < 10%)')
        s = plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True, vmax=3,
                                title=title)
        plotting.show()
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/post/figures/permut/{contrast_val}_{contrast}_thresh_zmap_permut.png')
        
        for section in sections:
            plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                          display_mode=section, 
                          cut_coords=8, 
                          black_bg=False,
                          title=title)
            s.savefig(f'{out_dir}/voxel/grey_matter_mask/post/figures/permut/{contrast_val}_{contrast}_thresh_zmap_permut_{section}.png')
        plotting.show()
    
        nib.save(neg_log_pvals_permuted_ols_unmasked, f'{out_dir}/voxel/grey_matter_mask/post/{contrast_val}_post_permuted_{contrast}.nii.gz')
         
        table = get_clusters_table(neg_log_pvals_permuted_ols_unmasked, stat_threshold=0.5,
                                   cluster_threshold=2, min_distance=0)
        save_dir = f'{out_dir}/voxel/grey_matter_mask/post/tables/permut/{contrast_val}_clusters_table_{contrast}_neg_log_pvals_permuted_ols.csv'
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

    #% Computing the (corrected) negative log p-values with permutation test
    neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
                                  design_matrix=design_matrix,
                                  second_level_contrast=contrast_val,
                                  model_intercept=True, n_perm=10000,
                                  two_sided_test=False, mask=fmri_mask,
                                  smoothing_fwhm=5.0, n_jobs=1)
        
    #plot the (corrected) negative log p-values
    title = (f'Group-level ({contrast_val}) association between \n'
              'neg-log of non-parametric corrected p-values (FWER < 10%)')
    s = plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True, vmax=3,
                            title=title)
    plotting.show()
    s.savefig(f'{out_dir}/voxel/grey_matter_mask/post_1vs1/figures/permut/{contrast_val}_{contrast}_thresh_zmap_permut.png')
    
    for section in sections:
        plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/post_1vs1/figures/permut/{contrast_val}_{contrast}_thresh_zmap_permut_{section}.png')
    plotting.show()

    nib.save(neg_log_pvals_permuted_ols_unmasked, f'{out_dir}/voxel/grey_matter_mask/post_1vs1/{contrast_val}_post_permuted_{contrast}.nii.gz')
     
    table = get_clusters_table(neg_log_pvals_permuted_ols_unmasked, stat_threshold=0.5,
                               cluster_threshold=2, min_distance=0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/post_1vs1/tables/permut/{contrast_val}_clusters_table_{contrast}_neg_log_pvals_permuted_ols.csv'
    table.to_csv(save_dir)
    
    # The neg-log p-values obtained with nonparametric testing are capped at 3
    # since the number of permutations is 1e3. 


























