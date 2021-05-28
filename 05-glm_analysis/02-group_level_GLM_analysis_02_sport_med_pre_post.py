"""
Created on Wed Dec  9 17:39:41 2020

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
from nilearn.plotting import plot_stat_map, plot_anat, plot_img, show, plot_glass_brain, plot_connectome
from nilearn.glm.second_level import SecondLevelModel, non_parametric_inference
from nilearn.plotting import plot_design_matrix, plot_contrast_matrix
from nilearn.input_data import NiftiMasker
from nilearn.glm import threshold_stats_img, cluster_level_inference
from nilearn.image import get_data, math_img
from nilearn.reporting import make_glm_report
from scipy.stats import norm
from scipy.stats import ttest_rel
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
                     '2back-1back', '2back-0back', '2back-fix', '0back','1back','2back','4back','fix']

groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])
#add subjects with high motion in at least one session
#high_motion = ['sub-']
#high_motion_filter = ~subjects_data_clean['sub'].isin(high_motion).values
# Removing high-motion subjects from dataset
#subjects_data_clean_lm = subjects_data_clean[~subjects_data_clean['sub'].isin(high_motion)].reset_index()

fmri_mask_path = '/home/alado/datasets/RBH/GLM/resampled_icbm_mask.nii.gz'
fmri_mask = nib.load(fmri_mask_path)
    
#%%Pre/post (ind) t-test smoothing_fwhm=5.0 for sport group with second-level model

print('Pre/post t-test smoothing_fwhm=5.0 for sport group with second-level model')

subs_sport = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport'])])
nsubs = len(subs_sport)

for k, contrast in enumerate(contr_fl_interest):
    print(contrast)
    zmaps_s1 = [] 
    for i, sub in enumerate(subs_sport):
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/grey_matter_mask/{sub}_ses-1_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps_s1.append(zmap)
            
    zmaps_s3 = [] 
    for i, sub in enumerate(subs_sport):
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/grey_matter_mask/{sub}_ses-3_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps_s3.append(zmap)
    
    zmaps = zmaps_s1 + zmaps_s3
    
    design_matrix = pd.DataFrame()
    #design_matrix['constant'] = [1] * nsubs * 2
    #no need to include an intercept as it is simply a linear combination of the other two regressors
    design_matrix['pre'] = [1] * nsubs + [0] * nsubs
    design_matrix['post'] = [0] * nsubs + [1] * nsubs

    contrast_val = 'post'
    
    second_level_model = SecondLevelModel(mask_img=fmri_mask,smoothing_fwhm=5.0)
    second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)
    
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    nib.save(z_map, f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/sport_pre_post_zmap_{contrast}.nii.gz')
    
    #fpr-thresholded map
    #level 0.005 means that there is 0.5% chance of declaring an inactive voxel, active
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=7)
    title = ('Thresholded z map, fpr <.001, clusters > 7 voxels \n'
             f'sport {contrast} smoothing_fwhm=5.0')
    s = plotting.plot_stat_map(
        thresholded_map1, threshold=threshold1, colorbar=True,
        title=title)
    s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/figures/fpr/sport_{contrast}_thresh_zmap_fpr<0.005.png')
    for section in sections:
        s = plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/figures/fpr/sport_{contrast}_thresh_zmap_fpr<0.005_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map1, threshold=threshold1)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/sport_pre_post_fpr_thresh-001_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold1,
                               cluster_threshold=3, min_distance=1.0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/tables/fpr/sport_clusters_table_{contrast}_fpr.csv'
    table.to_csv(save_dir)
    
    #fdr
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    title = (f'Thresholded z map, sport {contrast} expected fdr = .05')
    print('The FDR=.05 threshold is %.3g' % threshold2)    
    for section in sections:
        s = plot_stat_map(thresholded_map2, 
                      threshold=threshold2,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/figures/fdr/sport_{contrast}_thresh_zmap_fdr<0.05_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map2, threshold=threshold2)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/sport_pre_fdr_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold2,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/tables/fdr/sport_clusters_table_{contrast}_fdr.csv'
    table.to_csv(save_dir)
    
    #Bonferroni (fwer) -thresholded map
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    title = (f'Thresholded z map, sport {contrast}\n'
             'expected fwer < .05, smoothing_fwhm=5.0')
    for section in sections:
        s = plot_stat_map(thresholded_map3, 
                      threshold=threshold3,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/figures/bonferroni/sport_{contrast}_thresh_zmap_bonferroni_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map3, threshold=threshold3)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/sport_pre_post_bonferroni_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold3,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/tables/bonferroni/sport_clusters_table_{contrast}_bonferroni.csv'
    table.to_csv(save_dir)
    
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
    s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/figures/permut/sport_{contrast}_thresh_zmap_permut.png')
    
    for section in sections:
        plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/figures/permut/sport_{contrast}_thresh_zmap_permut_{section}.png')
    plotting.show()

    nib.save(neg_log_pvals_permuted_ols_unmasked, f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/sport_pre_post_permuted_{contrast}.nii.gz')
     
    table = get_clusters_table(neg_log_pvals_permuted_ols_unmasked, stat_threshold=0,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/tables/permut/sport_clusters_table_{contrast}_neg_log_pvals_permuted_ols.csv'
    table.to_csv(save_dir)
    
    # The neg-log p-values obtained with nonparametric testing are capped at 3
    # since the number of permutations is 1e3. 
    
#%%Pre/post (ind) t-test smoothing_fwhm=5.0 for meditation group with second-level model

print('Pre/post t-test smoothing_fwhm=5.0 for meditation group with second-level model')

subs_med = pd.Series.tolist(groups['sub'][groups['group'].isin(['med'])])
nsubs = len(subs_med)

for k, contrast in enumerate(contr_fl_interest):
    print(contrast)
    zmaps_s1 = [] 
    for i, sub in enumerate(subs_med):
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/grey_matter_mask/{sub}_ses-1_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps_s1.append(zmap)
            
    zmaps_s3 = [] 
    for i, sub in enumerate(subs_med):
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/grey_matter_mask/{sub}_ses-3_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps_s3.append(zmap)
    
    zmaps = zmaps_s1 + zmaps_s3
    
    design_matrix = pd.DataFrame()
    design_matrix['pre'] = [1] * nsubs + [0] * nsubs
    design_matrix['post'] = [0] * nsubs + [1] * nsubs
    
    contrast_val = 'post'

    second_level_model = SecondLevelModel(mask_img=fmri_mask,smoothing_fwhm=5.0)
    second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)
    
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    nib.save(z_map, f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/med_pre_post_zmap_{contrast}.nii.gz')
    
    #fpr-thresholded map
    #level 0.005 means that there is 0.5% chance of declaring an inactive voxel, active
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=7)
    title = ('Thresholded z map, fpr <.001, clusters > 7 voxels \n'
             f'med {contrast} smoothing_fwhm=5.0')
    s = plotting.plot_stat_map(
        thresholded_map1, threshold=threshold1, colorbar=True,
        title=title)
    s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/figures/fpr/med_{contrast}_thresh_zmap_fpr<0.005.png')
    for section in sections:
        s = plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/figures/fpr/med_{contrast}_thresh_zmap_fpr<0.005_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map1, threshold=threshold1)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/med_pre_post_fpr_thresh-001_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold1,
                               cluster_threshold=3, min_distance=1.0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/tables/fpr/med_clusters_table_{contrast}_fpr.csv'
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
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/figures/fdr/med_{contrast}_thresh_zmap_fdr<0.05_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map2, threshold=threshold2)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/med_pre_fdr_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold2,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/tables/fdr/med_clusters_table_{contrast}_fdr.csv'
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
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/figures/bonferroni/med_{contrast}_thresh_zmap_bonferroni_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map3, threshold=threshold3)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/med_pre_post_bonferroni_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold3,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/tables/bonferroni/med_clusters_table_{contrast}_bonferroni.csv'
    table.to_csv(save_dir)
    
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
    s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/figures/permut/med_{contrast}_thresh_zmap_permut.png')
    
    for section in sections:
        plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/figures/permut/med_{contrast}_thresh_zmap_permut_{section}.png')
    plotting.show()

    nib.save(neg_log_pvals_permuted_ols_unmasked, f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/med_pre_post_permuted_{contrast}.nii.gz')
     
    table = get_clusters_table(neg_log_pvals_permuted_ols_unmasked, stat_threshold=0,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/tables/permut/med_clusters_table_{contrast}_neg_log_pvals_permuted_ols.csv'
    table.to_csv(save_dir)
    
    # The neg-log p-values obtained with nonparametric testing are capped at 3
    # since the number of permutations is 1e3. 
    
#%%Pre/post (related) t-test smoothing_fwhm=5.0 for sport group with second-level model

print('Pre/post t-test smoothing_fwhm=5.0 for sport group with second-level model')

subs_sport = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport'])])
nsubs = len(subs_sport)

for k, contrast in enumerate(contr_fl_interest):
    print(contrast)
    zmaps_s1 = [] 
    for i, sub in enumerate(subs_sport):
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/grey_matter_mask/{sub}_ses-1_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps_s1.append(zmap)
            
    zmaps_s3 = [] 
    for i, sub in enumerate(subs_sport):
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/grey_matter_mask/{sub}_ses-3_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps_s3.append(zmap)
    
    zmaps = zmaps_s1 + zmaps_s3
    
    condition_effect = np.hstack(([1] * nsubs, [- 1] * nsubs))
    subject_effect = np.vstack((np.eye(nsubs), np.eye(nsubs)))
    subjects = ['S%02d' % i for i in range(1, nsubs + 1)]
    
    design_matrix = pd.DataFrame(
        np.hstack((condition_effect[:, np.newaxis], subject_effect)),
        columns=['pre_vs_post'] + subjects)

    contrast_val = 'pre_vs_post'
    
    second_level_model = SecondLevelModel(mask_img=fmri_mask,smoothing_fwhm=5.0)
    second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)
    
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    nib.save(z_map, f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/sport_pre_post_zmap_{contrast}.nii.gz')
    
    #fpr-thresholded map
    #level 0.005 means that there is 0.5% chance of declaring an inactive voxel, active
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=7)
    title = ('Thresholded z map, fpr <.001, clusters > 7 voxels \n'
             f'sport {contrast} smoothing_fwhm=5.0')
    s = plotting.plot_stat_map(
        thresholded_map1, threshold=threshold1, colorbar=True,
        title=title)
    s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/figures/fpr/sport_{contrast}_thresh_zmap_fpr<0.005.png')
    for section in sections:
        s = plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/figures/fpr/sport_{contrast}_thresh_zmap_fpr<0.005_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map1, threshold=threshold1)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/sport_pre_post_fpr_thresh-001_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold1,
                               cluster_threshold=3, min_distance=1.0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/tables/fpr/sport_clusters_table_{contrast}_fpr.csv'
    table.to_csv(save_dir)
    
    #fdr
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    title = (f'Thresholded z map, sport {contrast} expected fdr = .05')
    print('The FDR=.05 threshold is %.3g' % threshold2)    
    for section in sections:
        s = plot_stat_map(thresholded_map2, 
                      threshold=threshold2,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/figures/fdr/sport_{contrast}_thresh_zmap_fdr<0.05_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map2, threshold=threshold2)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/sport_pre_fdr_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold2,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/tables/fdr/sport_clusters_table_{contrast}_fdr.csv'
    table.to_csv(save_dir)
    
    #Bonferroni (fwer) -thresholded map
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    title = (f'Thresholded z map, sport {contrast}\n'
             'expected fwer < .05, smoothing_fwhm=5.0')
    for section in sections:
        s = plot_stat_map(thresholded_map3, 
                      threshold=threshold3,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/figures/bonferroni/sport_{contrast}_thresh_zmap_bonferroni_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map3, threshold=threshold3)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/sport_pre_post_bonferroni_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold3,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/tables/bonferroni/sport_clusters_table_{contrast}_bonferroni.csv'
    table.to_csv(save_dir)
    
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
    s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/figures/permut/sport_{contrast}_thresh_zmap_permut.png')
    
    for section in sections:
        plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/figures/permut/sport_{contrast}_thresh_zmap_permut_{section}.png')
    plotting.show()

    nib.save(neg_log_pvals_permuted_ols_unmasked, f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/sport_pre_post_permuted_{contrast}.nii.gz')
     
    table = get_clusters_table(neg_log_pvals_permuted_ols_unmasked, stat_threshold=0,
                               cluster_threshold=3, min_distance=2.0)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/tables/permut/sport_clusters_table_{contrast}_neg_log_pvals_permuted_ols.csv'
    table.to_csv(save_dir)
    
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
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/grey_matter_mask/{sub}_ses-1_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps_s1.append(zmap)
            
    zmaps_s3 = [] 
    for i, sub in enumerate(subs_med):
        zmap = nib.load(f'{top_dir}/GLM/FLA/voxel/grey_matter_mask/{sub}_ses-3_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
        zmaps_s3.append(zmap)
    
    zmaps = zmaps_s1 + zmaps_s3
    
    condition_effect = np.hstack(([1] * nsubs, [- 1] * nsubs))
    subject_effect = np.vstack((np.eye(nsubs), np.eye(nsubs)))
    subjects = ['S%02d' % i for i in range(1, nsubs + 1)]
    
    design_matrix = pd.DataFrame(
        np.hstack((condition_effect[:, np.newaxis], subject_effect)),
        columns=['pre_vs_post'] + subjects)

    contrast_val = 'pre_vs_post'

    second_level_model = SecondLevelModel(mask_img=fmri_mask,smoothing_fwhm=5.0)
    second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)
    
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    nib.save(z_map, f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/med_pre_post_zmap_{contrast}.nii.gz')
    
    #fpr-thresholded map
    #level 0.005 means that there is 0.5% chance of declaring an inactive voxel, active
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=7)
    title = ('Thresholded z map, fpr <.001, clusters > 7 voxels \n'
             f'med {contrast} smoothing_fwhm=5.0')
    s = plotting.plot_stat_map(
        thresholded_map1, threshold=threshold1, colorbar=True,
        title=title)
    s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/figures/fpr/med_{contrast}_thresh_zmap_fpr<0.005.png')
    for section in sections:
        s = plot_stat_map(thresholded_map1, 
                      threshold=threshold1,
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/figures/fpr/med_{contrast}_thresh_zmap_fpr<0.005_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map1, threshold=threshold1)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/med_pre_post_fpr_thresh-001_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold1,
                               cluster_threshold=2, min_distance=0.5)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/tables/fpr/med_clusters_table_{contrast}_fpr.csv'
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
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/figures/fdr/med_{contrast}_thresh_zmap_fdr<0.05_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map2, threshold=threshold2)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/med_pre_fdr_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold2,
                               cluster_threshold=2, min_distance=0.5)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/tables/fdr/med_clusters_table_{contrast}_fdr.csv'
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
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/figures/bonferroni/med_{contrast}_thresh_zmap_bonferroni_{section}.png')
    plotting.show()
    
    thresh_zmap = image.threshold_img(thresholded_map3, threshold=threshold3)
    nib.save(thresh_zmap, f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/med_pre_post_bonferroni_thresh-05_{contrast}.nii.gz')
    
    table = get_clusters_table(z_map, stat_threshold=threshold3,
                               cluster_threshold=2, min_distance=0.5)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/tables/bonferroni/med_clusters_table_{contrast}_bonferroni.csv'
    table.to_csv(save_dir)
    
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
    s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/figures/permut/med_{contrast}_thresh_zmap_permut.png')
    
    for section in sections:
        plot_stat_map(neg_log_pvals_permuted_ols_unmasked, 
                      display_mode=section, 
                      cut_coords=8, 
                      black_bg=False,
                      title=title)
        s.savefig(f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/figures/permut/med_{contrast}_thresh_zmap_permut_{section}.png')
    plotting.show()

    nib.save(neg_log_pvals_permuted_ols_unmasked, f'{out_dir}/voxel/grey_matter_mask/pre_post/rel_test/med_pre_post_permuted_{contrast}.nii.gz')
     
    table = get_clusters_table(neg_log_pvals_permuted_ols_unmasked, stat_threshold=0,
                               cluster_threshold=2, min_distance=0.5)
    save_dir = f'{out_dir}/voxel/grey_matter_mask/pre_post/ind_test/tables/permut/med_clusters_table_{contrast}_neg_log_pvals_permuted_ols.csv'
    table.to_csv(save_dir)
    
    # The neg-log p-values obtained with nonparametric testing are capped at 3
    # since the number of permutations is 1e3. 


























