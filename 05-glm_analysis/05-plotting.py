#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:29:17 2021

@author: alado
"""

#Working memory training: Plot p-value maps

import sys
import os
sys.path.append("..")
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import plotting
from nilearn import image
from nilearn.plotting import plot_stat_map, plot_anat, plot_img, show
from nilearn.plotting import plot_glass_brain, plot_connectome
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

fmri_mask_path = '{top_dir}/GLM/resampled_icbm_mask.nii.gz'
fmri_mask = nib.load(fmri_mask_path)

#%%
print('Pre/post t-test smoothing_fwhm=5.0 for sport group with second-level model')
for k, contrast in enumerate(contr_fl_interest):
    print(contrast)
    
    load_dir = f'{top_dir}/voxel/grey_matter_mask/pre_post/'
    fmri_image = image.load_img(f'{load_dir}/sport_pre_post_permuted', wildcards=True, dtype=None)
    
#%%
load_dir = f'{top_dir}/GLM/SLA/voxel/grey_matter_mask/pre_post/'
fmri_image = image.load_img(
        f'{load_dir}sport_pre_post_permuted_4back-0back_thresh_zmap.nii.gz', 
        wildcards=True, dtype=None)

s = plotting.plot_stat_map(fmri_image, colorbar=True, annotate=True,
                           draw_cross=False)


fmri_image = image.load_img(
        f'{load_dir}sport_pre_post_permuted_4back-0back_pmap.nii.gz', 
        wildcards=True, dtype=None)

THRESH = -np.log10(0.05)
s = plotting.plot_stat_map(fmri_image, colorbar=True, annotate=True,
                           draw_cross=False, threshold = THRESH)













