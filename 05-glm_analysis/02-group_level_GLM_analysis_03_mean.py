#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:39:41 2020

@author: alado
"""

#Working memory training: Second level GLM analysis of working memory training fMRI data
#Second-level GLM analysis of activation patterns

import sys
sys.path.append("..")
import itertools
import pandas as pd
import numpy as np
import nibabel as nib
import matplotlib as mpl
from nilearn import plotting
from nilearn.plotting import plot_stat_map, plot_anat, plot_img, show, plot_glass_brain, plot_connectome
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.input_data import NiftiMasker
from nilearn.glm import threshold_stats_img, cluster_level_inference
from nilearn.image import get_data, math_img
from nilearn.reporting import make_glm_report
from scipy.stats import norm
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
#increased cognitive load, like 4back-1 and 2back-1 and vice versa

groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])
#add subjects with high motion in at least one session
#high_motion = ['sub-']
#high_motion_filter = ~subjects_data_clean['sub'].isin(high_motion).values
# Removing high-motion subjects from dataset
#subjects_data_clean_lm = subjects_data_clean[~subjects_data_clean['sub'].isin(high_motion)].reset_index()

#%% Loading zmaps for contrasts defined like [1,-1,0,0,0]
#We are not interested in all contrasts, we are mainly interested in contrasts corresponding to the
#increased cognitive load, like 4back-1 and 2 back-1 and vice versa
contr_fl_interest = ['4back-1back']
zmaps = [] 
for k, contrast in enumerate(contr_fl_interest):
    for i, sub in enumerate(subs):
        zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_mean_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
            
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
design_matrix = pd.concat([design_matrix]*len_contr, ignore_index=True)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)

#%Step 3: Second-level GLM

second_level_model = SecondLevelModel(smoothing_fwhm=5.0)
second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)

contrasts_sl = ['control', 'sport', 'med']     

#%
for contrast_id, contrast_val in enumerate(contrasts_sl):
    print('Results for %s versus two remaining groups' % contrast_val)
    
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    
    #Raw z-map
    title = ('Raw z map, threshold=2, mean, %s' % contrast_val)
    display = plotting.plot_stat_map(z_map, threshold=2, title=title)
    
    for section in sections:
        plot_stat_map(z_map, 
                      threshold=2,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)

    #generate report html
    #make_glm_report(second_level_model, contrast_val).save_as_html('/home/alado/datasets/RBH/GLM/SLA/voxel/report7.html')
    
    #fpr-thresholded map
    #level 0.005 means that there is 0.5% chance of declaring an inactive voxel, active
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.005, height_control='fpr', cluster_threshold=7)
    title = ('Thresholded z map, fpr <.005, clusters > 7 voxels \n'
             'mean, smoothing_fwhm=5.0, %s' % contrast_val)
    plotting.plot_stat_map(
        thresholded_map1, threshold=threshold1,
        title=title)
    
    for section in sections:
        plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
    
    #fdr-thresholded map
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    print('The FDR=.05 threshold is %.3g' % threshold2)
    title = ('Thresholded z map, expected fdr = .05 \n'
         'mean, smoothing_fwhm=5.0, %s' % contrast_val)
    plotting.plot_stat_map(thresholded_map2, colorbar=True,
                           title='Thresholded z map, expected fdr = .05',
                           threshold=threshold2)
    plotting.show()
    
    #Bonferroni (fwer) -thresholded map
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    title = ('Thresholded z map, expected fwer < .05 \n'
             'mean, smoothing_fwhm=5.0, %s' % contrast_val)
    plotting.plot_stat_map(thresholded_map3,
                           title=title, colorbar=True,
                           threshold=threshold3)
    plotting.show()
    
    #Report the proportion of active voxels for all clusters defined by the input threshold
    proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)
    plotting.plot_stat_map(
        proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
        title='proportion true positives', vmax=1)
    
    #%Computing the (corrected) negative log p-values with permutation test
    
    neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
                                  design_matrix=design_matrix,
                                  second_level_contrast=contrast_val,
                                  model_intercept=True, n_perm=1000,
                                  two_sided_test=False, mask=None,
                                  smoothing_fwhm=5.0, n_jobs=1)
        
    #plot the (corrected) negative log p-values
    title = ('Group-level association between \n'
              'neg-log of non-parametric corrected p-values (FWER < 10%)')
    plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True, vmax=3,
                            title=title)
    plotting.show()
    
    for section in sections:
        plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
    
    # The neg-log p-values obtained with non parametric testing are capped at 3
    # since the number of permutations is 1e3.        
            
#%% Loading zmaps for contrasts defined like [1,-1,0,0,0]
#We are not interested in all contrasts, we are mainly interested in contrasts corresponding to the
#increased cognitive load, like 4back-1 and 2 back-1 and vice versa
contr_fl_interest = ['2back-1back']
zmaps = [] 
for k, contrast in enumerate(contr_fl_interest):
    for i, sub in enumerate(subs):
        zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_mean_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
            
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
design_matrix = pd.concat([design_matrix]*len_contr, ignore_index=True)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)

#%Step 3: Second-level GLM

second_level_model = SecondLevelModel(smoothing_fwhm=5.0)
second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)

contrasts_sl = ['control', 'sport', 'med']     

#%
for contrast_id, contrast_val in enumerate(contrasts_sl):
    print('Results for %s versus two remaining groups' % contrast_val)
    
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    
    #Raw z-map
    title = ('Raw z map, threshold=2, mean, %s' % contrast_val)
    display = plotting.plot_stat_map(z_map, threshold=2, title=title)
    
    for section in sections:
        plot_stat_map(z_map, 
                      threshold=2,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)

    #generate report html
    #make_glm_report(second_level_model, contrast_val).save_as_html('/home/alado/datasets/RBH/GLM/SLA/voxel/report7.html')
    
    #fpr-thresholded map
    #level 0.005 means that there is 0.5% chance of declaring an inactive voxel, active
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.005, height_control='fpr', cluster_threshold=7)
    title = ('Thresholded z map, fpr <.005, clusters > 7 voxels \n'
             'mean, smoothing_fwhm=5.0, %s' % contrast_val)
    plotting.plot_stat_map(
        thresholded_map1, threshold=threshold1,
        title=title)
    
    for section in sections:
        plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
    
    #fdr-thresholded map
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    print('The FDR=.05 threshold is %.3g' % threshold2)
    title = ('Thresholded z map, expected fdr = .05 \n'
         'mean, smoothing_fwhm=5.0, %s' % contrast_val)
    plotting.plot_stat_map(thresholded_map2, colorbar=True,
                           title='Thresholded z map, expected fdr = .05',
                           threshold=threshold2)
    plotting.show()
    
    #Bonferroni (fwer) -thresholded map
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    title = ('Thresholded z map, expected fwer < .05 \n'
             'mean, smoothing_fwhm=5.0, %s' % contrast_val)
    plotting.plot_stat_map(thresholded_map3,
                           title=title, colorbar=True,
                           threshold=threshold3)
    plotting.show()
    
    #Report the proportion of active voxels for all clusters defined by the input threshold
    proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)
    plotting.plot_stat_map(
        proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
        title='proportion true positives', vmax=1)
    
    #%Computing the (corrected) negative log p-values with permutation test
    
    neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
                                  design_matrix=design_matrix,
                                  second_level_contrast=contrast_val,
                                  model_intercept=True, n_perm=1000,
                                  two_sided_test=False, mask=None,
                                  smoothing_fwhm=5.0, n_jobs=1)
        
    #plot the (corrected) negative log p-values
    title = ('Group-level association between \n'
              'neg-log of non-parametric corrected p-values (FWER < 10%)')
    plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True, vmax=3,
                            title=title)
    plotting.show()
    
    for section in sections:
        plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
    
    # The neg-log p-values obtained with non parametric testing are capped at 3
    # since the number of permutations is 1e3.        
    
#%% Loading zmaps for contrast 'effects_of_interest'
#We are not interested in all contrasts, we are mainly interested in contrasts corresponding to the
#increased cognitive load, like 4back-1 and 2 back-1 and vice versa
contr_fl_interest = ['effects_of_interest']
zmaps = [] 
for k, contrast in enumerate(contr_fl_interest):
    for i, sub in enumerate(subs):
        zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_mean_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
            
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
design_matrix = pd.concat([design_matrix]*len_contr, ignore_index=True)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)

#%Step 3: Second-level GLM

second_level_model = SecondLevelModel(smoothing_fwhm=5.0)
second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)

contrasts_sl = ['control', 'sport', 'med']     

#%
for contrast_id, contrast_val in enumerate(contrasts_sl):
    print('Results for %s versus two remaining groups' % contrast_val)
    
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    
    #Raw z-map
    title = ('Raw z map, threshold=2, mean, %s' % contrast_val)
    display = plotting.plot_stat_map(z_map, threshold=2, title=title)
    
    for section in sections:
        plot_stat_map(z_map, 
                      threshold=2,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)

    #generate report html
    #make_glm_report(second_level_model, contrast_val).save_as_html('/home/alado/datasets/RBH/GLM/SLA/voxel/report7.html')
    
    #fpr-thresholded map
    #level 0.005 means that there is 0.5% chance of declaring an inactive voxel, active
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.005, height_control='fpr', cluster_threshold=7)
    title = ('Thresholded z map, fpr <.005, clusters > 7 voxels \n'
             'mean, smoothing_fwhm=5.0, %s' % contrast_val)
    plotting.plot_stat_map(
        thresholded_map1, threshold=threshold1,
        title=title)
    
    for section in sections:
        plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
    
    #fdr-thresholded map
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    print('The FDR=.05 threshold is %.3g' % threshold2)
    title = ('Thresholded z map, expected fdr = .05 \n'
         'mean, smoothing_fwhm=5.0, %s' % contrast_val)
    plotting.plot_stat_map(thresholded_map2, colorbar=True,
                           title='Thresholded z map, expected fdr = .05',
                           threshold=threshold2)
    plotting.show()
    
    #Bonferroni (fwer) -thresholded map
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    title = ('Thresholded z map, expected fwer < .05 \n'
             'mean, smoothing_fwhm=5.0, %s' % contrast_val)
    plotting.plot_stat_map(thresholded_map3,
                           title=title, colorbar=True,
                           threshold=threshold3)
    plotting.show()
    
    #Report the proportion of active voxels for all clusters defined by the input threshold
    proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)
    plotting.plot_stat_map(
        proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
        title='proportion true positives', vmax=1)
    
    #%Computing the (corrected) negative log p-values with permutation test
    
    neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
                                  design_matrix=design_matrix,
                                  second_level_contrast=contrast_val,
                                  model_intercept=True, n_perm=1000,
                                  two_sided_test=False, mask=None,
                                  smoothing_fwhm=5.0, n_jobs=1)
        
    #plot the (corrected) negative log p-values
    title = ('Group-level association between \n'
              'neg-log of non-parametric corrected p-values (FWER < 10%)')
    plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True, vmax=3,
                            title=title)
    plotting.show()

for section in sections:
    plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                  display_mode=section, 
                  cut_coords=8, 
                  black_bg=False,
                  title=title)
    
    # The neg-log p-values obtained with non parametric testing are capped at 3
    # since the number of permutations is 1e3.        
            