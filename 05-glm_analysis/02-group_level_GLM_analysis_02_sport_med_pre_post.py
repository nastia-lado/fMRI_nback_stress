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
rand_state = 42
nperms = 10000

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


























