"""
Created 06.02.2021

@author: alado
"""

#Working memory training: Second level GLM analysis of working memory training fMRI data
#Second-level GLM analysis of activation patterns

import sys
sys.path.append("..")
import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import plotting
from nilearn import image
from nilearn.plotting import plot_stat_map, plot_anat, plot_img, show
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.input_data import NiftiMasker
from nilearn.glm import threshold_stats_img, cluster_level_inference
from nilearn.image import get_data, math_img
from nilearn.reporting import make_glm_report
from scipy.stats import norm
from scipy.stats import ttest_rel
from nilearn.reporting import get_clusters_table
from nilearn.datasets import fetch_icbm152_brain_gm_mask

#%%Step 1: Loading data
data_dir = '/home/alado/datasets/RBH/preprocessed/derivatives/fmriprep'
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/GLM/SLA/'
#mask
fmri_mask = nib.load('/home/alado/datasets/RBH/GLM/resampled_icbm_mask.nii.gz')
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
                     '2back-1back', '2back-0back', '2back-fix', '0back','1back','2back','4back','fix']

groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])
#add subjects with high motion in at least one session
#high_motion = ['sub-']
#high_motion_filter = ~subjects_data_clean['sub'].isin(high_motion).values
# Removing high-motion subjects from dataset
#subjects_data_clean = subjects_data_clean[~subjects_data_clean['sub'].isin(high_motion)].reset_index()

#%%One-sample t-test from whole brain zmaps with GM mask
print('One-sample t-test for pre- zmaps')

nsubs = len(subs)
design_matrix = pd.DataFrame([1] * nsubs, columns=['intercept'])
fmri_mask_path = '/home/alado/datasets/RBH/GLM/resampled_icbm_mask.nii.gz'
fmri_mask = nib.load(fmri_mask_path)

for k, contrast in enumerate(contr_fl_interest):
    print(f'Contrast {contrast}')
    zmaps_s1 = [] 
    for i, sub in enumerate(subs):
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/{sub}_ses-1_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps_s1.append(zmap)
        
    second_level_model = SecondLevelModel(mask_img=fmri_mask, smoothing_fwhm=5.0)
    second_level_model = second_level_model.fit(zmaps_s1, design_matrix=design_matrix)
    
    z_map = second_level_model.compute_contrast('intercept', output_type='z_score')
    nib.save(z_map, f'{out_dir}/voxel/pre/pre_zmap_{contrast}.nii.gz')
    
    #threshold the second level contrast at uncorrected p < 0.001 and plot it
    p_val = 0.001
    p001_unc = norm.isf(p_val)
    title=f'group {contrast} (unc p<0.001)'    
    for section in sections:
       s = plot_stat_map(z_map, 
                      threshold=p001_unc,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
       s.savefig(f'{out_dir}/voxel/pre/figures/unc/{contrast}_uncorrected_p<0.001_{section}.png')
    plotting.show()
    
    table = get_clusters_table(z_map, stat_threshold=p001_unc,
                               cluster_threshold=3, min_distance=1.0)
    save_dir = f'{out_dir}/voxel/pre/tables/unc/clusters_table_{contrast}_uncorrected.csv'
    table.to_csv(save_dir)
    
    #fpr-thresholded map
    #level 0.005 means that there is 0.5% chance of declaring an inactive voxel, active
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=7)
    title = ('Thresholded z map, fpr <.001, clusters > 7 voxels \n'
             f'{contrast} smoothing_fwhm=5.0')
    s = plotting.plot_stat_map(
        thresholded_map1, threshold=threshold1, colorbar=True,
        title=title)
    s.savefig(f'{out_dir}/voxel/pre/figures/fpr/{contrast}_thresh_zmap_fpr<0.005.png')
    for section in sections:
        s = plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/pre/figures/fpr/{contrast}_thresh_zmap_fpr<0.005_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map1, threshold=threshold1)
    nib.save(thresh_zmap, f'{out_dir}/voxel/pre/pre_fpr_thresh-001_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold1,
                               cluster_threshold=3, min_distance=1.0)
    save_dir = f'{out_dir}/voxel/pre/tables/fpr/clusters_table_{contrast}_fpr.csv'
    table.to_csv(save_dir)
    
    #fdr
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    title = (f'Thresholded z map, group {contrast} expected fdr = .05')
    print('The FDR=.05 threshold is %.3g' % threshold2)    
    for section in sections:
        s = plot_stat_map(thresholded_map2, 
                      threshold=threshold2,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/pre/figures/fdr/{contrast}_thresh_zmap_fdr<0.05_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map2, threshold=threshold2)
    nib.save(thresh_zmap, f'{out_dir}/voxel/pre/pre_fdr_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold2,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/pre/tables/fdr/clusters_table_{contrast}_fdr.csv'
    table.to_csv(save_dir)
    
    #Bonferroni (fwer) -thresholded map
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    title = (f'Thresholded z map, group {contrast}\n'
             'expected fwer < .05, smoothing_fwhm=5.0')
    for section in sections:
        s = plot_stat_map(thresholded_map3, 
                      threshold=threshold3,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/pre/figures/bonferroni/{contrast}_thresh_zmap_bonferroni_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map3, threshold=threshold3)
    nib.save(thresh_zmap, f'{out_dir}/voxel/pre/pre_bonferroni_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold3,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/pre/tables/bonferroni/clusters_table_{contrast}_bonferroni.csv'
    table.to_csv(save_dir)
    
    #% Computing the (corrected) negative log p-values with permutation test
    neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps_s1,
                                  design_matrix=design_matrix,
                                  second_level_contrast='intercept',
                                  model_intercept=True, n_perm=10000,
                                  two_sided_test=False, mask=fmri_mask,
                                  smoothing_fwhm=5.0, n_jobs=1)
        
    #plot the (corrected) negative log p-values
    title = ('Group-level association between \n'
              'neg-log of non-parametric corrected p-values (FWER < 10%)')
    s = plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True, vmax=3,
                            title=title)
    plotting.show()
    s.savefig(f'{out_dir}/voxel/pre/figures/permut/{contrast}_thresh_zmap_permut.png')
    
    for section in sections:
        plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/pre/figures/permut/{contrast}_thresh_zmap_permut_{section}.png')
    plotting.show()

    nib.save(neg_log_pvals_permuted_ols_unmasked, f'{out_dir}/voxel/pre/pre_permuted_{contrast}.nii.gz')
     
    table = get_clusters_table(neg_log_pvals_permuted_ols_unmasked, stat_threshold=0,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/pre/tables/permut/clusters_table_{contrast}_neg_log_pvals_permuted_ols.csv'
    table.to_csv(save_dir)
    
    # The neg-log p-values obtained with nonparametric testing are capped at 3
    # since the number of permutations is 1e3. 
   
#%%remove high motion subjects
#add subjects with high motion in at least one session
#high_motion = ['sub-']
#high_motion_filter = ~subjects_data_clean['sub'].isin(high_motion).values
# Removing high-motion subjects from dataset
#subjects_data_clean = subjects_data_clean[~subjects_data_clean['sub'].isin(high_motion)].reset_index()

print('One-sample t-test for pre- zmaps')

nsubs = len(subs)
design_matrix = pd.DataFrame([1] * nsubs, columns=['intercept'])
icbm_mask = fetch_icbm152_brain_gm_mask()

for k, contrast in enumerate(contr_fl_interest):
    print(f'Contrast {contrast}')
    zmaps_s1 = [] 
    for i, sub in enumerate(subs):
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/{sub}_ses-1_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps_s1.append(zmap)
        
    second_level_model = SecondLevelModel(mask_img=fmri_mask, smoothing_fwhm=5.0)
    second_level_model = second_level_model.fit(zmaps_s1, design_matrix=design_matrix)
    
    z_map = second_level_model.compute_contrast('intercept', output_type='z_score')
    nib.save(z_map, f'{out_dir}/voxel/pre/pre_zmap_{contrast}.nii.gz')
    
    #threshold the second level contrast at uncorrected p < 0.001 and plot it
    p_val = 0.001
    p001_unc = norm.isf(p_val)
    title=f'group {contrast} (unc p<0.001)'    
    for section in sections:
       s = plot_stat_map(z_map, 
                      threshold=p001_unc,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
       s.savefig(f'{out_dir}/voxel/pre/figures/unc/{contrast}_uncorrected_p<0.001_{section}.png')
    plotting.show()
    
    table = get_clusters_table(z_map, stat_threshold=p001_unc,
                               cluster_threshold=3, min_distance=1.0)
    save_dir = f'{out_dir}/voxel/pre/tables/unc/clusters_table_{contrast}_uncorrected.csv'
    table.to_csv(save_dir)
    
    #fpr-thresholded map
    #level 0.005 means that there is 0.5% chance of declaring an inactive voxel, active
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=7)
    title = ('Thresholded z map, fpr <.001, clusters > 7 voxels \n'
             f'{contrast} smoothing_fwhm=5.0')
    s = plotting.plot_stat_map(
        thresholded_map1, threshold=threshold1, colorbar=True,
        title=title)
    s.savefig(f'{out_dir}/voxel/pre/figures/fpr/{contrast}_thresh_zmap_fpr<0.005.png')
    for section in sections:
        s = plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/pre/figures/fpr/{contrast}_thresh_zmap_fpr<0.005_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map1, threshold=threshold1)
    nib.save(thresh_zmap, f'{out_dir}/voxel/pre/pre_fpr_thresh-001_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold1,
                               cluster_threshold=3, min_distance=1.0)
    save_dir = f'{out_dir}/voxel/pre/tables/fpr/clusters_table_{contrast}_fpr.csv'
    table.to_csv(save_dir)
    
    #fdr
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    title = (f'Thresholded z map, group {contrast} expected fdr = .05')
    print('The FDR=.05 threshold is %.3g' % threshold2)    
    for section in sections:
        s = plot_stat_map(thresholded_map2, 
                      threshold=threshold2,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/pre/figures/fdr/{contrast}_thresh_zmap_fdr<0.05_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map2, threshold=threshold2)
    nib.save(thresh_zmap, f'{out_dir}/voxel/pre/pre_fdr_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold2,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/pre/tables/fdr/clusters_table_{contrast}_fdr.csv'
    table.to_csv(save_dir)
    
    #Bonferroni (fwer) -thresholded map
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    title = (f'Thresholded z map, group {contrast}\n'
             'expected fwer < .05, smoothing_fwhm=5.0')
    for section in sections:
        s = plot_stat_map(thresholded_map3, 
                      threshold=threshold3,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/pre/figures/bonferroni/{contrast}_thresh_zmap_bonferroni_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map3, threshold=threshold3)
    nib.save(thresh_zmap, f'{out_dir}/voxel/pre/pre_bonferroni_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold3,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/pre/tables/bonferroni/clusters_table_{contrast}_bonferroni.csv'
    table.to_csv(save_dir)
    
    #% Computing the (corrected) negative log p-values with permutation test
    neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps_s1,
                                  design_matrix=design_matrix,
                                  second_level_contrast='intercept',
                                  model_intercept=True, n_perm=10000,
                                  two_sided_test=False, mask=fmri_mask,
                                  smoothing_fwhm=5.0, n_jobs=1)
        
    #plot the (corrected) negative log p-values
    title = ('Group-level association between \n'
              'neg-log of non-parametric corrected p-values (FWER < 10%)')
    s = plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True, vmax=3,
                            title=title)
    plotting.show()
    s.savefig(f'{out_dir}/voxel/pre/figures/permut/{contrast}_thresh_zmap_permut.png')
    
    for section in sections:
        plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/pre/figures/permut/{contrast}_thresh_zmap_permut_{section}.png')
    plotting.show()

    nib.save(neg_log_pvals_permuted_ols_unmasked, f'{out_dir}/voxel/pre/pre_permuted_{contrast}.nii.gz')
     
    table = get_clusters_table(neg_log_pvals_permuted_ols_unmasked, stat_threshold=0,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/pre/tables/permut/clusters_table_{contrast}_neg_log_pvals_permuted_ols.csv'
    table.to_csv(save_dir)
    
    # The neg-log p-values obtained with nonparametric testing are capped at 3
    # since the number of permutations is 1e3. 






























