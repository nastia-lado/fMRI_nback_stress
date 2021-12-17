#Working memory training: Second level GLM analysis of working memory training fMRI data
#Second-level GLM analysis of activation patterns

import sys
sys.path.append("..")
import os
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import nibabel as nib
import matplotlib as mpl
from nilearn import plotting
from nilearn import input_data, datasets
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

contrasts_fl = ['0back-1back','0back-2back','0back-4back','0back-fix',
             '1back-2back','1back-4back','1back-fix','2back-4back',
             '2back-fix','4back-fix','0back','1back','2back','4back','fix']

groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])
#add subjects with high motion in at least one session
#high_motion = ['sub-']
#high_motion_filter = ~subjects_data_clean['sub'].isin(high_motion).values
# Removing high-motion subjects from dataset
#subjects_data_clean_lm = subjects_data_clean[~subjects_data_clean['sub'].isin(high_motion)].reset_index()

#%% Loading zmaps for contrasts defined like [1,-1,0,0,0]
zmaps = [] 
for k, contrast in enumerate(contrasts_fl[0:10]):
    for j, ses in enumerate(sess):
        for i, sub in enumerate(subs):
            zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast}.nii.gz')

#%Step 2: Create second-level design matrix (520)
n_sub = len(subs)
design_matrix = pd.DataFrame()
design_matrix['constant'] = [1] * n_sub * len(sess)
ses_arr = np.asarray([1,2])
ses_val = (ses_arr - np.mean(ses_arr))/np.std(ses_arr)
design_matrix['ses_pre'] = list(itertools.chain.from_iterable([[i] * n_sub for i in ses_val]))
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(1,2)
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(-1,1)
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(2,0)
design_matrix['ses_post'] = list(itertools.chain.from_iterable([[i] * n_sub for i in ses_val]))
design_matrix['ses_post'] = design_matrix['ses_post'].replace(-1,0)
design_matrix['control'] = list(itertools.chain.from_iterable([(groups['group']=='control').values.astype(int)]*2))
design_matrix['sport'] = list(itertools.chain.from_iterable([(groups['group']=='sport').values.astype(int)]*2))
design_matrix['med'] = list(itertools.chain.from_iterable([(groups['group']=='med').values.astype(int)]*2))
ds_sub = np.eye(len(groups['sub']))
ds_subs = pd.DataFrame(np.concatenate([ds_sub, ds_sub]), columns = groups['sub'])
design_matrix = pd.concat((design_matrix, ds_subs), axis=1)
design_matrix = pd.concat([design_matrix]*10, ignore_index=True)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)

#%Step 3: Second level GLM analysis
#no smoothing on a previous step
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)

#% Step 4: estimate the contrast (e.g. use the column name of the design matrix)
n_columns = len(design_matrix.columns)

contrast = np.zeros((6, n_columns))
contrast[0:6, 1:6] =  [[-1, 1, 1, -1, 0],
                        [-1, 1, 1, 0, -1],
                        [-1, 1, 0, 1, -1],
                        [-1, 1, 0, -1, 1],
                        [-1, 1, -1, 1, 0],
                        [-1, 1, -1, 0, 1]]

z_map = second_level_model.compute_contrast(contrast, output_type='z_score')

#generate report html
make_glm_report(second_level_model, contrast).save_as_html(f'{top_dir}/GLM/SLA/voxel/report1_pos_and_neg_fla.html')

#%Step 5: FDR-thresholded result
#Compute the required threshold level and return the thresholded map (map, threshold)
#add cluster_threshold
_, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr')

#plot the second level contrast at the computed thresholds
#bg_img default
plotting.plot_stat_map(
    z_map, threshold=threshold, colorbar=True,
    title='Group-level \n'
    '(fdr=0.01)')
plotting.show()

#%Computing corrected p-values with parametric test to compare with non parametric test
#gives RuntimeWarning: divide by zero encountered in log10, but it's ok
p_val = second_level_model.compute_contrast(contrast, output_type='p_value')
n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
# Correcting the p-values for multiple testing and taking negative logarithm
neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))".format(str(n_voxels)),img=p_val)

#plot the (corrected) negative log p-values for the parametric test

#cut_coords = [50, -17, -3]
# Since we are plotting negative log p-values and using a threshold equal to 1,
# it corresponds to corrected p-values lower than 10%, meaning that there
# is less than 10% probability to make a single false discovery
# (90% chance that we make no false discoveries at all).
# This threshold is much more conservative than the previous one.
threshold = 1
title = ('Group-level association between \n'
         'neg-log of parametric corrected p-values (FWER < 10%)')
plotting.plot_stat_map(neg_log_pval, colorbar=True, 
                       threshold=threshold, title=title)
plotting.show()

#%threshold the second level contrast at uncorrected p < 0.001 and plot
p_val = 0.001
p001_uncorrected = norm.isf(p_val) #Inverse survival function

proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)

plotting.plot_stat_map(
    proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
    title='group, proportion true positives', vmax=1)

plotting.plot_stat_map(
    z_map, threshold=p001_uncorrected, colorbar=True, display_mode='z',
    title='group (uncorrected p < 0.001)')
plotting.show()


#%threshold the second level contrast and plot it

threshold = 3.1  # correponds to  p < .001, uncorrected
display = plotting.plot_glass_brain(
    z_map, threshold=threshold, colorbar=True, plot_abs=False,
    title='vertical vs horizontal checkerboard (unc p<0.001')
plotting.show()

#%Threshold the resulting map: false positive rate < .001, cluster size > 10 voxels

thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=10)

#p<.001 uncorrected-thresholded map (with only clusters > 10 voxels).

plotting.plot_stat_map(
    z_map,
    #thresholded_map1, 
    #cut_coords=display.cut_coords, 
    threshold=threshold1,
    title='Thresholded z map, fpr <.001, clusters > 10 voxels')
plotting.show()

#%FDR <.05 (False Discovery Rate) and no cluster-level threshold.
#the fdr-thresholded map
thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
print('The FDR=.05 threshold is %.3g' % threshold2)
plotting.plot_stat_map(thresholded_map2,
                       #cut_coords=display.cut_coords,
                       title='Thresholded z map, expected fdr = .05',
                       threshold=threshold2)
plotting.show()

#%use FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold.
#If the data has not been intensively smoothed, we can use a simple Bonferroni correction.
#the Bonferroni-thresholded map
thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
plotting.plot_stat_map(thresholded_map3,
                       title='Thresholded z map, expected fwer < .05',
                       threshold=threshold3)
plotting.show()


#%Computing the (corrected) negative log p-values with permutation test

# neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
#                              design_matrix=design_matrix,
#                              second_level_contrast=contrast,
#                              model_intercept=True, n_perm=1000,
#                              two_sided_test=False, mask=None,
#                              smoothing_fwhm=5.0, n_jobs=1)
    
#plot the (corrected) negative log p-values

# title = ('Group-level association between \n'
#          'neg-log of non-parametric corrected p-values (FWER < 10%)')
# plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True,
#                        #cut_coords=cut_coords,
#                        threshold=threshold,
#                        title=title)
# plotting.show()

# The neg-log p-values obtained with non parametric testing are capped at 3
# since the number of permutations is 1e3.
# The non parametric test yields a few more discoveries
# and is then more powerful than the usual parametric procedure.

#%%Loading zmaos for contrasts defined like [1,0,0,0,0]
zmaps = []
for k, contrast in enumerate(contrasts_fl[10:]):
    for j, ses in enumerate(sess):
        for i, sub in enumerate(subs):
            zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
                
#%Step 2: Create second-level design matrix (260)
n_sub = len(subs)
design_matrix = pd.DataFrame()
design_matrix['constant'] = [1] * n_sub * len(sess)
ses_arr = np.asarray([1,2])
ses_val = (ses_arr - np.mean(ses_arr))/np.std(ses_arr)
design_matrix['ses_pre'] = list(itertools.chain.from_iterable([[i] * n_sub for i in ses_val]))
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(1,2)
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(-1,1)
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(2,0)
design_matrix['ses_post'] = list(itertools.chain.from_iterable([[i] * n_sub for i in ses_val]))
design_matrix['ses_post'] = design_matrix['ses_post'].replace(-1,0)
design_matrix['control'] = list(itertools.chain.from_iterable([(groups['group']=='control').values.astype(int)]*2))
design_matrix['sport'] = list(itertools.chain.from_iterable([(groups['group']=='sport').values.astype(int)]*2))
design_matrix['med'] = list(itertools.chain.from_iterable([(groups['group']=='med').values.astype(int)]*2))
ds_sub = np.eye(len(groups['sub']))
ds_subs = pd.DataFrame(np.concatenate([ds_sub, ds_sub]), columns = groups['sub'])
design_matrix = pd.concat((design_matrix, ds_subs), axis=1)
design_matrix = pd.concat([design_matrix]*5, ignore_index=True)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)

#%Step 3: Second level GLM analysis
#smoothing done on a previous step = 3.5
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)

#% Step 4: estimate the contrast (e.g. use the column name of the design matrix)
n_columns = len(design_matrix.columns)

contrast = np.zeros((6, n_columns))
contrast[0:6, 1:6] =  [[-1, 1, 1, -1, 0],
                        [-1, 1, 1, 0, -1],
                        [-1, 1, 0, 1, -1],
                        [-1, 1, 0, -1, 1],
                        [-1, 1, -1, 1, 0],
                        [-1, 1, -1, 0, 1]]

z_map = second_level_model.compute_contrast(contrast, output_type='z_score')

#Generate report html
make_glm_report(second_level_model, contrast).save_as_html(f'{top_dir}/GLM/SLA/voxel/report2.html')

#%Step 5: FDR-thresholded result
#Compute the required threshold level and return the thresholded map (map, threshold)
#add cluster_threshold
_, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr')

#plot the second level contrast at the computed thresholds
#bg_img default
plotting.plot_stat_map(
    z_map, threshold=threshold, colorbar=True,
    title='Group-level \n'
    '(fdr=0.01)')
plotting.show()

#%Computing corrected p-values with parametric test to compare with non parametric test
#gives RuntimeWarning: divide by zero encountered in log10, but it's ok
p_val = second_level_model.compute_contrast(contrast, output_type='p_value')
n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
# Correcting the p-values for multiple testing and taking negative logarithm
neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))".format(str(n_voxels)),img=p_val)

#plot the (corrected) negative log p-values for the parametric test

#cut_coords = [50, -17, -3]
# Since we are plotting negative log p-values and using a threshold equal to 1,
# it corresponds to corrected p-values lower than 10%, meaning that there
# is less than 10% probability to make a single false discovery
# (90% chance that we make no false discoveries at all).
# This threshold is much more conservative than the previous one.
threshold = 1
title = ('Group-level association between \n'
         'neg-log of parametric corrected p-values (FWER < 10%)')
plotting.plot_stat_map(neg_log_pval, colorbar=True, 
                       #cut_coords=cut_coords,
                       threshold=threshold, title=title)
plotting.show()

#%threshold the second level contrast at uncorrected p < 0.001 and plot
p_val = 0.001
p001_uncorrected = norm.isf(p_val) #Inverse survival function

proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)

plotting.plot_stat_map(
    proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
    title='group, proportion true positives', vmax=1)

plotting.plot_stat_map(
    z_map, threshold=p001_uncorrected, colorbar=True, display_mode='z',
    title='group (uncorrected p < 0.001)')
plotting.show()

#%threshold the second level contrast and plot it

threshold = 3.1  # correponds to  p < .001, uncorrected
display = plotting.plot_glass_brain(
    z_map, threshold=threshold, colorbar=True, plot_abs=False,
    title='vertical vs horizontal checkerboard (unc p<0.001')
plotting.show()

#%Threshold the resulting map: false positive rate < .001, cluster size > 10 voxels

thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=10)

#p<.001 uncorrected-thresholded map (with only clusters > 10 voxels).

plotting.plot_stat_map(
    z_map,
    #thresholded_map1, 
    #cut_coords=display.cut_coords, 
    threshold=threshold1,
    title='Thresholded z map, fpr <.001, clusters > 10 voxels')
plotting.show()

#%FDR <.05 (False Discovery Rate) and no cluster-level threshold.

thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
print('The FDR=.05 threshold is %.3g' % threshold2)

#the fdr-thresholded map
plotting.plot_stat_map(thresholded_map2,
                       #cut_coords=display.cut_coords,
                       title='Thresholded z map, expected fdr = .05',
                       threshold=threshold2
                       )
plotting.show()

#%FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold.
#If the data has not been intensively smoothed, we can use a simple Bonferroni correction.

thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)

#the Bonferroni-thresholded map

plotting.plot_stat_map(thresholded_map3,
                       #cut_coords=display.cut_coords,
                       title='Thresholded z map, expected fwer < .05',
                       threshold=threshold3)
plotting.show()

#%Computing the (corrected) negative log p-values with permutation test

# neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
#                              design_matrix=design_matrix,
#                              second_level_contrast=contrast,
#                              model_intercept=True, n_perm=1000,
#                              two_sided_test=False, mask=None,
#                              smoothing_fwhm=5.0, n_jobs=1)
    
# #plot the (corrected) negative log p-values

# title = ('Group-level association between \n'
#          'neg-log of non-parametric corrected p-values (FWER < 10%)')
# plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True,
#                        #cut_coords=cut_coords,
#                        threshold=threshold,
#                        title=title)
# plotting.show()

# The neg-log p-values obtained with non parametric testing are capped at 3
# since the number of permutations is 1e3.
# The non parametric test yields a few more discoveries
# and is then more powerful than the usual parametric procedure.

#%% Compute each contrast separately
#% Loading zmaps for contrasts defined like [1,-1,0,0,0]
zmaps = [] 
for k, contrast in enumerate(contrasts_fl[0:10]):
    for j, ses in enumerate(sess):
        for i, sub in enumerate(subs):
            zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast}.nii.gz')

#%Step 2: Create second-level design matrix (520)
n_sub = len(subs)
design_matrix = pd.DataFrame()
design_matrix['constant'] = [1] * n_sub * len(sess)
ses_arr = np.asarray([1,2])
ses_val = (ses_arr - np.mean(ses_arr))/np.std(ses_arr)
design_matrix['ses_pre'] = list(itertools.chain.from_iterable([[i] * n_sub for i in ses_val]))
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(1,2)
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(-1,1)
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(2,0)
design_matrix['ses_post'] = list(itertools.chain.from_iterable([[i] * n_sub for i in ses_val]))
design_matrix['ses_post'] = design_matrix['ses_post'].replace(-1,0)
design_matrix['control'] = list(itertools.chain.from_iterable([(groups['group']=='control').values.astype(int)]*2))
design_matrix['sport'] = list(itertools.chain.from_iterable([(groups['group']=='sport').values.astype(int)]*2))
design_matrix['med'] = list(itertools.chain.from_iterable([(groups['group']=='med').values.astype(int)]*2))
ds_sub = np.eye(len(groups['sub']))
ds_subs = pd.DataFrame(np.concatenate([ds_sub, ds_sub]), columns = groups['sub'])
design_matrix = pd.concat((design_matrix, ds_subs), axis=1)
design_matrix = pd.concat([design_matrix]*10, ignore_index=True)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)

#%Step 3: Second level GLM analysis
#no smoothing on a previous step
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)

#% Step 4: estimate the contrast (e.g. use the column name of the design matrix)
n_columns = len(design_matrix.columns)
contrasts_sl = {'ses+control-sport': np.pad([0, -1, 1, 1, -1, 0], (0,n_columns-6)),
                'ses-control+sport': np.pad([0, -1, 1, -1, 1, 0], (0,n_columns-6)),
                'ses+control-med': np.pad([0, -1, 1, 1, 0, -1], (0,n_columns-6)),
                'ses-control+med': np.pad([0, -1, 1, -1, 0, 1], (0,n_columns-6)),
                'ses+sport-med': np.pad([0, -1, 1, 0, 1, -1], (0,n_columns-6)),
                'ses-sport+med': np.pad([0, -1, 1, 0, -1, 1], (0,n_columns-6))}

for index, (contrast_id, contrast_val) in enumerate(contrasts_sl.items()):
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    plotting.plot_stat_map(z_map, threshold=2.0,title='%s' % contrast_id)
    
    #generate report html
    #make_glm_report(second_level_model, contrast_val)
    #Generate report html
    #make_glm_report(second_level_model, contrast).save_as_html(f'{top_dir}/RBH/GLM/SLA/voxel/report3_{}.html')

    #%Step 5: FDR-thresholded result
    #Compute the required threshold level and return the thresholded map (map, threshold)
    #add cluster_threshold
    _, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    
    #plot the second level contrast at the computed thresholds
    #bg_img default
    plotting.plot_stat_map(
        z_map, threshold=threshold, colorbar=True,
        title='Group-level %s\n'
        '(fdr=0.01)' % contrast_id)
    plotting.show()

    #%Computing corrected p-values with parametric test to compare with non parametric test
    #gives RuntimeWarning: divide by zero encountered in log10, but it's ok
    p_val = second_level_model.compute_contrast(contrast_val, output_type='p_value')
    n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
    # Correcting the p-values for multiple testing and taking negative logarithm
    neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))".format(str(n_voxels)),img=p_val)
    
    #plot the (corrected) negative log p-values for the parametric test
    
    #cut_coords = [50, -17, -3]
    # Since we are plotting negative log p-values and using a threshold equal to 1,
    # it corresponds to corrected p-values lower than 10%, meaning that there
    # is less than 10% probability to make a single false discovery
    # (90% chance that we make no false discoveries at all).
    # This threshold is much more conservative than the previous one.
    threshold = 1
    title = ('Group-level association between \n'
             'neg-log of parametric corrected p-values (FWER < 10%)')
    plotting.plot_stat_map(neg_log_pval, colorbar=True, 
                           #cut_coords=cut_coords,
                           threshold=threshold, title=title)
    plotting.show()

    #%threshold the second level contrast at uncorrected p < 0.001 and plot
    p_val = 0.001
    p001_uncorrected = norm.isf(p_val) #Inverse survival function
    
    proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)
    
    plotting.plot_stat_map(
        proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
        title='group, proportion true positives', vmax=1)
    
    plotting.plot_stat_map(
        z_map, threshold=p001_uncorrected, colorbar=True, display_mode='z',
        title='group (uncorrected p < 0.001)')
    plotting.show()

    
    #%threshold the second level contrast and plot it
    
    threshold = 3.1  # correponds to  p < .001, uncorrected
    display = plotting.plot_glass_brain(
        z_map, threshold=threshold, colorbar=True, plot_abs=False,
        title='vertical vs horizontal checkerboard (unc p<0.001')
    plotting.show()

    #%Threshold the resulting map: false positive rate < .001, cluster size > 10 voxels
    
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=10)
    
    #p<.001 uncorrected-thresholded map (with only clusters > 10 voxels).
    
    plotting.plot_stat_map(
        z_map,
        #thresholded_map1, 
        #cut_coords=display.cut_coords, 
        threshold=threshold1,
        title='Thresholded z map, fpr <.001, clusters > 10 voxels')
    plotting.show()
    
    #%FDR <.05 (False Discovery Rate) and no cluster-level threshold.
    
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    print('The FDR=.05 threshold is %.3g' % threshold2)
    
    #the fdr-thresholded map
    plotting.plot_stat_map(thresholded_map2,
                           #cut_coords=display.cut_coords,
                           title='Thresholded z map, expected fdr = .05',
                           threshold=threshold2
                           )
    plotting.show()

    #%Now use FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold.
    #If the data has not been intensively smoothed, we can use a simple Bonferroni correction.
    
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    
    #the Bonferroni-thresholded map
    
    plotting.plot_stat_map(thresholded_map3,
                           #cut_coords=display.cut_coords,
                           title='Thresholded z map, expected fwer < .05',
                           threshold=threshold3)
    plotting.show()
    
    #%Computing the (corrected) negative log p-values with permutation test
    
    # neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
    #                              design_matrix=design_matrix,
    #                              second_level_contrast=contrast_val,
    #                              model_intercept=True, n_perm=1000,
    #                              two_sided_test=False, mask=None,
    #                              smoothing_fwhm=5.0, n_jobs=1)
        
    #plot the (corrected) negative log p-values
    
    # title = ('Group-level association between \n'
    #          'neg-log of non-parametric corrected p-values (FWER < 10%)')
    # plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True,
    #                        #cut_coords=cut_coords,
    #                        threshold=threshold,
    #                        title=title)
    # plotting.show()
    
    # The neg-log p-values obtained with non parametric testing are capped at 3
    # since the number of permutations is 1e3.
    # The non parametric test yields a few more discoveries
    # and is then more powerful than the usual parametric procedure.

#%% Compute each contrast separately
#%Loading zmaps for contrasts defined like [1,0,0,0,0]
zmaps = []
for k, contrast in enumerate(contrasts_fl[10:]):
    for j, ses in enumerate(sess):
        for i, sub in enumerate(subs):
            zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
                
#%Step 2: Create second-level design matrix
n_sub = len(subs)
design_matrix = pd.DataFrame()
design_matrix['constant'] = [1] * n_sub * len(sess)
ses_arr = np.asarray([1,2])
ses_val = (ses_arr - np.mean(ses_arr))/np.std(ses_arr)
design_matrix['ses_pre'] = list(itertools.chain.from_iterable([[i] * n_sub for i in ses_val]))
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(1,2)
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(-1,1)
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(2,0)
design_matrix['ses_post'] = list(itertools.chain.from_iterable([[i] * n_sub for i in ses_val]))
design_matrix['ses_post'] = design_matrix['ses_post'].replace(-1,0)
design_matrix['control'] = list(itertools.chain.from_iterable([(groups['group']=='control').values.astype(int)]*2))
design_matrix['sport'] = list(itertools.chain.from_iterable([(groups['group']=='sport').values.astype(int)]*2))
design_matrix['med'] = list(itertools.chain.from_iterable([(groups['group']=='med').values.astype(int)]*2))
ds_sub = np.eye(len(groups['sub']))
ds_subs = pd.DataFrame(np.concatenate([ds_sub, ds_sub]), columns = groups['sub'])
design_matrix = pd.concat((design_matrix, ds_subs), axis=1)
design_matrix = pd.concat([design_matrix]*5, ignore_index=True)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)

#%Step 3: Second level GLM analysis
#no smoothing on a previous step
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)

#% Step 4: estimate the contrast (e.g. use the column name of the design matrix)
n_columns = len(design_matrix.columns)
contrasts_sl = {'ses+control-sport': np.pad([0, -1, 1, 1, -1, 0], (0,n_columns-6)),
                'ses-control+sport': np.pad([0, -1, 1, -1, 1, 0], (0,n_columns-6)),
                'ses+control-med': np.pad([0, -1, 1, 1, 0, -1], (0,n_columns-6)),
                'ses-control+med': np.pad([0, -1, 1, -1, 0, 1], (0,n_columns-6)),
                'ses+sport-med': np.pad([0, -1, 1, 0, 1, -1], (0,n_columns-6)),
                'ses-sport+med': np.pad([0, -1, 1, 0, -1, 1], (0,n_columns-6))}

for index, (contrast_id, contrast_val) in enumerate(contrasts_sl.items()):
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    plotting.plot_stat_map(z_map, threshold=2.0,title='%s' % contrast_id)
    
    #generate report html
    make_glm_report(second_level_model, contrast_val)

    #%Step 5: FDR-thresholded result
    #Compute the required threshold level and return the thresholded map (map, threshold)
    #add cluster_threshold
    _, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    
    #plot the second level contrast at the computed thresholds
    #bg_img default
    plotting.plot_stat_map(
        z_map, threshold=threshold, colorbar=True,
        title='Group-level %s\n'
        '(fdr=0.01)' % contrast_id)
    plotting.show()

    #%Computing corrected p-values with parametric test to compare with non parametric test
    #gives RuntimeWarning: divide by zero encountered in log10, but it's ok
    p_val = second_level_model.compute_contrast(contrast_val, output_type='p_value')
    n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
    # Correcting the p-values for multiple testing and taking negative logarithm
    neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))".format(str(n_voxels)),img=p_val)
    
    #plot the (corrected) negative log p-values for the parametric test
    
    #cut_coords = [50, -17, -3]
    # Since we are plotting negative log p-values and using a threshold equal to 1,
    # it corresponds to corrected p-values lower than 10%, meaning that there
    # is less than 10% probability to make a single false discovery
    # (90% chance that we make no false discoveries at all).
    # This threshold is much more conservative than the previous one.
    threshold = 1
    title = ('Group-level association between \n'
             'neg-log of parametric corrected p-values (FWER < 10%)')
    plotting.plot_stat_map(neg_log_pval, colorbar=True, 
                           #cut_coords=cut_coords,
                           threshold=threshold, title=title)
    plotting.show()

    #%threshold the second level contrast at uncorrected p < 0.001 and plot
    p_val = 0.001
    p001_uncorrected = norm.isf(p_val) #Inverse survival function
    
    proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)
    
    plotting.plot_stat_map(
        proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
        title='group, proportion true positives', vmax=1)
    
    plotting.plot_stat_map(
        z_map, threshold=p001_uncorrected, colorbar=True, display_mode='z',
        title='group (uncorrected p < 0.001)')
    plotting.show()

    
    #%threshold the second level contrast and plot it
    
    threshold = 3.1  # correponds to  p < .001, uncorrected
    display = plotting.plot_glass_brain(
        z_map, threshold=threshold, colorbar=True, plot_abs=False,
        title='vertical vs horizontal checkerboard (unc p<0.001')
    plotting.show()

    #%Threshold the resulting map: false positive rate < .001, cluster size > 10 voxels
    
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=10)
    
    #p<.001 uncorrected-thresholded map (with only clusters > 10 voxels).
    
    plotting.plot_stat_map(
        z_map,
        #thresholded_map1, 
        #cut_coords=display.cut_coords, 
        threshold=threshold1,
        title='Thresholded z map, fpr <.001, clusters > 10 voxels')
    plotting.show()
    
    #%FDR <.05 (False Discovery Rate) and no cluster-level threshold.
    
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    print('The FDR=.05 threshold is %.3g' % threshold2)
    
    #the fdr-thresholded map
    plotting.plot_stat_map(thresholded_map2,
                           #cut_coords=display.cut_coords,
                           title='Thresholded z map, expected fdr = .05',
                           threshold=threshold2
                           )
    plotting.show()

    #%Now use FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold.
    #If the data has not been intensively smoothed, we can use a simple Bonferroni correction.
    
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    
    #the Bonferroni-thresholded map
    
    plotting.plot_stat_map(thresholded_map3,
                           #cut_coords=display.cut_coords,
                           title='Thresholded z map, expected fwer < .05',
                           threshold=threshold3)
    plotting.show()
    
    #%Computing the (corrected) negative log p-values with permutation test
    
    # neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
    #                              design_matrix=design_matrix,
    #                              second_level_contrast=contrast_val,
    #                              model_intercept=True, n_perm=1000,
    #                              two_sided_test=False, mask=None,
    #                              smoothing_fwhm=5.0, n_jobs=1)
        
    #plot the (corrected) negative log p-values
    
    # title = ('Group-level association between \n'
    #          'neg-log of non-parametric corrected p-values (FWER < 10%)')
    # plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True,
    #                        #cut_coords=cut_coords,
    #                        threshold=threshold,
    #                        title=title)
    # plotting.show()
    
    # The neg-log p-values obtained with non parametric testing are capped at 3
    # since the number of permutations is 1e3.
    # The non parametric test yields a few more discoveries
    # and is then more powerful than the usual parametric procedure.

#%%each FL contrast separately
#% Loading zmaps for contrasts defined like [1,-1,0,0,0]
#%Step 2: Create second-level design matrix
n_sub = len(subs)
design_matrix = pd.DataFrame()
design_matrix['constant'] = [1] * n_sub * len(sess)
ses_arr = np.asarray([1,2])
ses_val = (ses_arr - np.mean(ses_arr))/np.std(ses_arr)
design_matrix['ses_pre'] = list(itertools.chain.from_iterable([[i] * n_sub for i in ses_val]))
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(1,2)
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(-1,1)
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(2,0)
design_matrix['ses_post'] = list(itertools.chain.from_iterable([[i] * n_sub for i in ses_val]))
design_matrix['ses_post'] = design_matrix['ses_post'].replace(-1,0)
design_matrix['control'] = list(itertools.chain.from_iterable([(groups['group']=='control').values.astype(int)]*2))
design_matrix['sport'] = list(itertools.chain.from_iterable([(groups['group']=='sport').values.astype(int)]*2))
design_matrix['med'] = list(itertools.chain.from_iterable([(groups['group']=='med').values.astype(int)]*2))
ds_sub = np.eye(len(groups['sub']))
ds_subs = pd.DataFrame(np.concatenate([ds_sub, ds_sub]), columns = groups['sub'])
design_matrix = pd.concat((design_matrix, ds_subs), axis=1)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)
    
n_columns = len(design_matrix.columns)
contrasts_sl = {'ses+control-sport': np.pad([0, -1, 1, 1, -1, 0], (0,n_columns-6)),
                'ses-control+sport': np.pad([0, -1, 1, -1, 1, 0], (0,n_columns-6)),
                'ses+control-med': np.pad([0, -1, 1, 1, 0, -1], (0,n_columns-6)),
                'ses-control+med': np.pad([0, -1, 1, -1, 0, 1], (0,n_columns-6)),
                'ses+sport-med': np.pad([0, -1, 1, 0, 1, -1], (0,n_columns-6)),
                'ses-sport+med': np.pad([0, -1, 1, 0, -1, 1], (0,n_columns-6))}

for k, contrast in enumerate(contrasts_fl[0:10]):
    zmaps = []
    print('Contrast % 2i out of %i: %s' % (k + 1, len(contrasts_fl[0:10]), k))
    for j, ses in enumerate(sess):
        for i, sub in enumerate(subs):
            zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
    
    #%Step 3: Second level GLM analysis 
    #no smoothing on a previous step
    second_level_model = SecondLevelModel()
    second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)
    
    #% Step 4: estimate the contrast (e.g. use the column name of the design matrix)
    for index, (contrast_id, contrast_val) in enumerate(contrasts_sl.items()):
        z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
        plotting.plot_stat_map(z_map, threshold=2.0, title='%s' % contrast_val)
        
        #generate report html
        make_glm_report(second_level_model, contrast_val)
    
        #%Step 5: FDR-thresholded result
        #Compute the required threshold level and return the thresholded map (map, threshold)
        #add cluster_threshold
        _, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
        
        #plot the second level contrast at the computed thresholds
        #bg_img default
        plotting.plot_stat_map(
            z_map, threshold=threshold, colorbar=True,
            title='Group-level %s\n'
            '(fdr=0.01)' % contrast_val)
        plotting.show()
    
        #%Computing corrected p-values with parametric test to compare with non parametric test
        #gives RuntimeWarning: divide by zero encountered in log10, but it's ok
        p_val = second_level_model.compute_contrast(contrast_val, output_type='p_value')
        n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
        # Correcting the p-values for multiple testing and taking negative logarithm
        neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))".format(str(n_voxels)),img=p_val)
        
        #plot the (corrected) negative log p-values for the parametric test
        
        #cut_coords = [50, -17, -3]
        # Since we are plotting negative log p-values and using a threshold equal to 1,
        # it corresponds to corrected p-values lower than 10%, meaning that there
        # is less than 10% probability to make a single false discovery
        # (90% chance that we make no false discoveries at all).
        # This threshold is much more conservative than the previous one.
        threshold = 1
        title = ('Group-level association between \n'
                 'neg-log of parametric corrected p-values (FWER < 10%)')
        plotting.plot_stat_map(neg_log_pval, colorbar=True, 
                               #cut_coords=cut_coords,
                               threshold=threshold, title=title)
        plotting.show()
    
        #%threshold the second level contrast at uncorrected p < 0.001 and plot
        p_val = 0.001
        p001_uncorrected = norm.isf(p_val) #Inverse survival function
        
        proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)
        
        plotting.plot_stat_map(
            proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
            title='group, proportion true positives', vmax=1)
        
        plotting.plot_stat_map(
            z_map, threshold=p001_uncorrected, colorbar=True, display_mode='z',
            title='group (uncorrected p < 0.001)')
        plotting.show()
    
        
        #%threshold the second level contrast and plot it
        
        threshold = 3.1  # correponds to  p < .001, uncorrected
        display = plotting.plot_glass_brain(
            z_map, threshold=threshold, colorbar=True, plot_abs=False,
            title='vertical vs horizontal checkerboard (unc p<0.001')
        plotting.show()
    
        #%Threshold the resulting map: false positive rate < .001, cluster size > 10 voxels
        
        thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=10)
        
        #p<.001 uncorrected-thresholded map (with only clusters > 10 voxels).
        
        plotting.plot_stat_map(
            z_map,
            #thresholded_map1, 
            #cut_coords=display.cut_coords, 
            threshold=threshold1,
            title='Thresholded z map, fpr <.001, clusters > 10 voxels')
        plotting.show()
        
        #%FDR <.05 (False Discovery Rate) and no cluster-level threshold.
        
        thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
        print('The FDR=.05 threshold is %.3g' % threshold2)
        
        #the fdr-thresholded map
        plotting.plot_stat_map(thresholded_map2,
                               #cut_coords=display.cut_coords,
                               title='Thresholded z map, expected fdr = .05',
                               threshold=threshold2
                               )
        plotting.show()
    
        #%Now use FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold.
        #If the data has not been intensively smoothed, we can use a simple Bonferroni correction.
        
        thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
        print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
        
        #the Bonferroni-thresholded map
        
        plotting.plot_stat_map(thresholded_map3,
                               #cut_coords=display.cut_coords,
                               title='Thresholded z map, expected fwer < .05',
                               threshold=threshold3)
        plotting.show()
        
        # contrast_val[contrast_val == -1] = 0
        
        # #%Computing the (corrected) negative log p-values with permutation test
        # neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
        #                               design_matrix=design_matrix,
        #                               second_level_contrast=contrast_val,
        #                               model_intercept=True, n_perm=1000,
        #                               two_sided_test=False, mask=None,
        #                               smoothing_fwhm=5.0, n_jobs=1)
            
        # #plot the (corrected) negative log p-values
        # title = ('Group-level association between \n'
        #           'neg-log of non-parametric corrected p-values (FWER < 10%)')
        # plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True,
        #                         #cut_coords=cut_coords,
        #                         threshold=threshold,
        #                         title=title)
        # plotting.show()
        
        # The neg-log p-values obtained with non parametric testing are capped at 3
        # since the number of permutations is 1e3.
        # The non parametric test yields a few more discoveries
        # and is then more powerful than the usual parametric procedure.

#%%each mean pre-post FL contrast separately
#%Loading zmaos for contrasts defined like [1,0,0,0,0]
#%Step 2: Create second-level design matrix
#%Step 2: Create second-level design matrix
n_sub = len(subs)
design_matrix = pd.DataFrame()
design_matrix['constant'] = [1] * n_sub * len(sess)
ses_arr = np.asarray([1,2])
ses_val = (ses_arr - np.mean(ses_arr))/np.std(ses_arr)
design_matrix['ses_pre'] = list(itertools.chain.from_iterable([[i] * n_sub for i in ses_val]))
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(1,2)
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(-1,1)
design_matrix['ses_pre'] = design_matrix['ses_pre'].replace(2,0)
design_matrix['ses_post'] = list(itertools.chain.from_iterable([[i] * n_sub for i in ses_val]))
design_matrix['ses_post'] = design_matrix['ses_post'].replace(-1,0)
design_matrix['control'] = list(itertools.chain.from_iterable([(groups['group']=='control').values.astype(int)]*2))
design_matrix['sport'] = list(itertools.chain.from_iterable([(groups['group']=='sport').values.astype(int)]*2))
design_matrix['med'] = list(itertools.chain.from_iterable([(groups['group']=='med').values.astype(int)]*2))
ds_sub = np.eye(len(groups['sub']))
ds_subs = pd.DataFrame(np.concatenate([ds_sub, ds_sub]), columns = groups['sub'])
design_matrix = pd.concat((design_matrix, ds_subs), axis=1)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)
    
n_columns = len(design_matrix.columns)
contrasts_sl = {'ses+control-sport': np.pad([0, -1, 1, 1, -1, 0], (0,n_columns-6)),
                'ses-control+sport': np.pad([0, -1, 1, -1, 1, 0], (0,n_columns-6)),
                'ses+control-med': np.pad([0, -1, 1, 1, 0, -1], (0,n_columns-6)),
                'ses-control+med': np.pad([0, -1, 1, -1, 0, 1], (0,n_columns-6)),
                'ses+sport-med': np.pad([0, -1, 1, 0, 1, -1], (0,n_columns-6)),
                'ses-sport+med': np.pad([0, -1, 1, 0, -1, 1], (0,n_columns-6))}

for k, contrast in enumerate(contrasts_fl[0:10]):
    zmaps = []
    print('Contrast % 2i out of %i: %s' % (k + 1, len(contrasts_fl[0:10]), k))
    for j, ses in enumerate(sess):
        for i, sub in enumerate(subs):
            zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_{ses}_space-MNI152NLin2009cAsym_{contrast}.nii.gz')

    #%Step 3: Second level GLM analysis
    #no smoothing on a previous step
    second_level_model = SecondLevelModel()
    second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)
    
    #% Step 4: estimate the contrast (e.g. use the column name of the design matrix)
    for index, (contrast_id, contrast_val) in enumerate(contrasts_sl.itmes()):
        z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
        plotting.plot_stat_map(z_map, threshold=2.0,title='%s' % contrast_id)
        
        #generate report html
        make_glm_report(second_level_model, contrast_val)
    
        #%Step 5: FDR-thresholded result
        #Compute the required threshold level and return the thresholded map (map, threshold)
        #add cluster_threshold
        _, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
        
        #plot the second level contrast at the computed thresholds
        #bg_img default
        plotting.plot_stat_map(
            z_map, threshold=threshold, colorbar=True,
            title='Group-level %s\n'
            '(fdr=0.01)' % contrast_id)
        plotting.show()
    
        #%Computing corrected p-values with parametric test to compare with non parametric test
        #gives RuntimeWarning: divide by zero encountered in log10, but it's ok
        p_val = second_level_model.compute_contrast(contrast_val, output_type='p_value')
        n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
        # Correcting the p-values for multiple testing and taking negative logarithm
        neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))".format(str(n_voxels)),img=p_val)
        
        #plot the (corrected) negative log p-values for the parametric test
        
        #cut_coords = [50, -17, -3]
        # Since we are plotting negative log p-values and using a threshold equal to 1,
        # it corresponds to corrected p-values lower than 10%, meaning that there
        # is less than 10% probability to make a single false discovery
        # (90% chance that we make no false discoveries at all).
        # This threshold is much more conservative than the previous one.
        threshold = 1
        title = ('Group-level association between \n'
                 'neg-log of parametric corrected p-values (FWER < 10%)')
        plotting.plot_stat_map(neg_log_pval, colorbar=True, 
                               #cut_coords=cut_coords,
                               threshold=threshold, title=title)
        plotting.show()
    
        #%threshold the second level contrast at uncorrected p < 0.001 and plot
        p_val = 0.001
        p001_uncorrected = norm.isf(p_val) #Inverse survival function
        
        proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)
        
        plotting.plot_stat_map(
            proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
            title='group, proportion true positives', vmax=1)
        
        plotting.plot_stat_map(
            z_map, threshold=p001_uncorrected, colorbar=True, display_mode='z',
            title='group (uncorrected p < 0.001)')
        plotting.show()
    
        
        #%threshold the second level contrast and plot it
        
        threshold = 3.1  # correponds to  p < .001, uncorrected
        display = plotting.plot_glass_brain(
            z_map, threshold=threshold, colorbar=True, plot_abs=False,
            title='vertical vs horizontal checkerboard (unc p<0.001')
        plotting.show()
    
        #%Threshold the resulting map: false positive rate < .001, cluster size > 10 voxels
        
        thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=10)
        
        #p<.001 uncorrected-thresholded map (with only clusters > 10 voxels).
        
        plotting.plot_stat_map(
            z_map,
            #thresholded_map1, 
            #cut_coords=display.cut_coords, 
            threshold=threshold1,
            title='Thresholded z map, fpr <.001, clusters > 10 voxels')
        plotting.show()
        
        #%FDR <.05 (False Discovery Rate) and no cluster-level threshold.
        
        thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
        print('The FDR=.05 threshold is %.3g' % threshold2)
        
        #the fdr-thresholded map
        plotting.plot_stat_map(thresholded_map2,
                               #cut_coords=display.cut_coords,
                               title='Thresholded z map, expected fdr = .05',
                               threshold=threshold2
                               )
        plotting.show()
    
        #%Now use FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold.
        #If the data has not been intensively smoothed, we can use a simple Bonferroni correction.
        
        thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
        print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
        
        #the Bonferroni-thresholded map
        
        plotting.plot_stat_map(thresholded_map3,
                               #cut_coords=display.cut_coords,
                               title='Thresholded z map, expected fwer < .05',
                               threshold=threshold3)
        plotting.show()
        
        # contrast_val[contrast_val == -1] = 0
        # #%Computing the (corrected) negative log p-values with permutation test
        # neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
        #                               design_matrix=design_matrix,
        #                               second_level_contrast=contrast_val,
        #                               model_intercept=True, n_perm=1000,
        #                               two_sided_test=False, mask=None,
        #                               smoothing_fwhm=5.0, n_jobs=1)
            
        # #plot the (corrected) negative log p-values
        # title = ('Group-level association between \n'
        #           'neg-log of non-parametric corrected p-values (FWER < 10%)')
        # plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True,
        #                         #cut_coords=cut_coords,
        #                         threshold=threshold,
        #                         title=title)
        # plotting.show()
        
        # The neg-log p-values obtained with non parametric testing are capped at 3
        # since the number of permutations is 1e3.
        # The non parametric test yields a few more discoveries
        # and is then more powerful than the usual parametric procedure.



#%%Compute from mean pre-post
#% Loading zmaps for contrasts defined like [1,-1,0,0,0]
zmaps = [] 
for k, contrast in enumerate(contrasts_fl[0:10]):
        for i, sub in enumerate(subs):
            zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_mean_space-MNI152NLin2009cAsym_{contrast}.nii.gz')

#%Step 2: Create second-level design matrix (520)
n_sub = len(subs)
design_matrix = pd.DataFrame()
design_matrix['constant'] = [1] * n_sub
design_matrix['control'] = list(itertools.chain.from_iterable([(groups['group']=='control').values.astype(int)]))
design_matrix['sport'] = list(itertools.chain.from_iterable([(groups['group']=='sport').values.astype(int)]))
design_matrix['med'] = list(itertools.chain.from_iterable([(groups['group']=='med').values.astype(int)]))
ds_sub = np.eye(len(groups['sub']))
ds_subs = pd.DataFrame(ds_sub, columns = groups['sub'])
design_matrix = pd.concat((design_matrix, ds_subs), axis=1)
design_matrix = pd.concat([design_matrix]*10, ignore_index=True)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)

#%Step 3: Second level GLM analysis
#no smoothing on a previous step
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)

#% Step 4: estimate the contrast (e.g. use the column name of the design matrix)
n_columns = len(design_matrix.columns)
contrasts_sl = {'control-sport': np.pad([0, 1, -1, 0], (0,n_columns-4)),
                'control+sport': np.pad([0, -1, 1, 0], (0,n_columns-4)),
                'control-med': np.pad([0, 1, 0, -1], (0,n_columns-4)),
                'control+med': np.pad([0, -1, 0, 1], (0,n_columns-4)),
                'sport-med': np.pad([0, 0, 1, -1], (0,n_columns-4)),
                'sport+med': np.pad([0, 0, -1, 1], (0,n_columns-4))}

for index, (contrast_id, contrast_val) in enumerate(contrasts_sl.items()):
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    plotting.plot_stat_map(z_map, threshold=2.0,title='%s' % contrast_id)
    
    #generate report html
    make_glm_report(second_level_model, contrast_val)

    #%Step 5: FDR-thresholded result
    #Compute the required threshold level and return the thresholded map (map, threshold)
    #add cluster_threshold
    _, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    
    #plot the second level contrast at the computed thresholds
    #bg_img default
    plotting.plot_stat_map(
        z_map, threshold=threshold, colorbar=True,
        title='Group-level %s\n'
        '(fdr=0.01)' % contrast_id)
    plotting.show()

    #%Computing corrected p-values with parametric test to compare with non parametric test
    #gives RuntimeWarning: divide by zero encountered in log10, but it's ok
    p_val = second_level_model.compute_contrast(contrast_val, output_type='p_value')
    n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
    # Correcting the p-values for multiple testing and taking negative logarithm
    neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))".format(str(n_voxels)),img=p_val)
    
    #plot the (corrected) negative log p-values for the parametric test
    
    #cut_coords = [50, -17, -3]
    # Since we are plotting negative log p-values and using a threshold equal to 1,
    # it corresponds to corrected p-values lower than 10%, meaning that there
    # is less than 10% probability to make a single false discovery
    # (90% chance that we make no false discoveries at all).
    # This threshold is much more conservative than the previous one.
    threshold = 1
    title = ('Group-level association between \n'
             'neg-log of parametric corrected p-values (FWER < 10%)')
    plotting.plot_stat_map(neg_log_pval, colorbar=True, 
                           #cut_coords=cut_coords,
                           threshold=threshold, title=title)
    plotting.show()

    #%threshold the second level contrast at uncorrected p < 0.001 and plot
    p_val = 0.001
    p001_uncorrected = norm.isf(p_val) #Inverse survival function
    
    proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)
    
    plotting.plot_stat_map(
        proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
        title='group, proportion true positives', vmax=1)
    
    plotting.plot_stat_map(
        z_map, threshold=p001_uncorrected, colorbar=True, display_mode='z',
        title='group (uncorrected p < 0.001)')
    plotting.show()

    
    #%threshold the second level contrast and plot it
    
    threshold = 3.1  # correponds to  p < .001, uncorrected
    display = plotting.plot_glass_brain(
        z_map, threshold=threshold, colorbar=True, plot_abs=False,
        title='group (unc p<0.001')
    plotting.show()

    #%Threshold the resulting map: false positive rate < .001, cluster size > 10 voxels
    
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=10)
    
    #p<.001 uncorrected-thresholded map (with only clusters > 10 voxels).
    
    plotting.plot_stat_map(
        z_map,
        #thresholded_map1, 
        #cut_coords=display.cut_coords, 
        threshold=threshold1,
        title='Thresholded z map, fpr <.001, clusters > 10 voxels')
    plotting.show()
    
    #%FDR <.05 (False Discovery Rate) and no cluster-level threshold.
    
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    print('The FDR=.05 threshold is %.3g' % threshold2)
    
    #the fdr-thresholded map
    plotting.plot_stat_map(thresholded_map2,
                           #cut_coords=display.cut_coords,
                           title='Thresholded z map, expected fdr = .05',
                           threshold=threshold2
                           )
    plotting.show()

    #%Now use FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold.
    #If the data has not been intensively smoothed, we can use a simple Bonferroni correction.
    
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    
    #the Bonferroni-thresholded map
    
    plotting.plot_stat_map(thresholded_map3,
                           #cut_coords=display.cut_coords,
                           title='Thresholded z map, expected fwer < .05',
                           threshold=threshold3)
    plotting.show()
    
    #%Computing the (corrected) negative log p-values with permutation test
    
    # neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
    #                              design_matrix=design_matrix,
    #                              second_level_contrast=contrast_val,
    #                              model_intercept=True, n_perm=1000,
    #                              two_sided_test=False, mask=None,
    #                              smoothing_fwhm=5.0, n_jobs=1)
        
    #plot the (corrected) negative log p-values
    
    # title = ('Group-level association between \n'
    #          'neg-log of non-parametric corrected p-values (FWER < 10%)')
    # plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True,
    #                        #cut_coords=cut_coords,
    #                        threshold=threshold,
    #                        title=title)
    # plotting.show()
    
    # The neg-log p-values obtained with non parametric testing are capped at 3
    # since the number of permutations is 1e3.
    # The non parametric te#%%Compute from mean pre-postst yields a few more discoveries
    # and is then more powerful than the usual parametric procedure.


#%%Compute from mean pre-post
#%Loading zmaos for contrasts defined like [1,0,0,0,0]
zmaps = []
for k, contrast in enumerate(contrasts_fl[10:]):
        for i, sub in enumerate(subs):
            zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_mean_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
                
#%Step 2: Create second-level design matrix
n_sub = len(subs)
design_matrix = pd.DataFrame()
design_matrix['constant'] = [1] * n_sub
design_matrix['control'] = list(itertools.chain.from_iterable([(groups['group']=='control').values.astype(int)]))
design_matrix['sport'] = list(itertools.chain.from_iterable([(groups['group']=='sport').values.astype(int)]))
design_matrix['med'] = list(itertools.chain.from_iterable([(groups['group']=='med').values.astype(int)]))
ds_sub = np.eye(len(groups['sub']))
ds_subs = pd.DataFrame(ds_sub, columns = groups['sub'])
design_matrix = pd.concat((design_matrix, ds_subs), axis=1)
design_matrix = pd.concat([design_matrix]*5, ignore_index=True)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)

#%Step 3: Second level GLM analysis
#no smoothing on a previous step
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)

#% Step 4: estimate the contrast (e.g. use the column name of the design matrix)
n_columns = len(design_matrix.columns)

contrasts_sl = {'control-sport': np.pad([0, 1, -1, 0], (0,n_columns-4)),
                'control+sport': np.pad([0, -1, 1, 0], (0,n_columns-4)),
                'control-med': np.pad([0, 1, 0, -1], (0,n_columns-4)),
                'control+med': np.pad([0, -1, 0, 1], (0,n_columns-4)),
                'sport-med': np.pad([0, 0, 1, -1], (0,n_columns-4)),
                'sport+med': np.pad([0, 0, -1, 1], (0,n_columns-4))}

for index, (contrast_id, contrast_val) in enumerate(contrasts_sl.items()):
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    plotting.plot_stat_map(z_map, threshold=2.0,title='%s' % contrast_id)
    
    #generate report html
    make_glm_report(second_level_model, contrast_val)

    #%Step 5: FDR-thresholded result
    #Compute the required threshold level and return the thresholded map (map, threshold)
    #add cluster_threshold
    _, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    
    #plot the second level contrast at the computed thresholds
    #bg_img default
    plotting.plot_stat_map(
        z_map, threshold=threshold, colorbar=True,
        title='Group-level %s\n'
        '(fdr=0.01)' % contrast_id)
    plotting.show()

    #%Computing corrected p-values with parametric test to compare with non parametric test
    #gives RuntimeWarning: divide by zero encountered in log10, but it's ok
    p_val = second_level_model.compute_contrast(contrast_val, output_type='p_value')
    n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
    # Correcting the p-values for multiple testing and taking negative logarithm
    neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))".format(str(n_voxels)),img=p_val)
    
    #plot the (corrected) negative log p-values for the parametric test
    
    #cut_coords = [50, -17, -3]
    # Since we are plotting negative log p-values and using a threshold equal to 1,
    # it corresponds to corrected p-values lower than 10%, meaning that there
    # is less than 10% probability to make a single false discovery
    # (90% chance that we make no false discoveries at all).
    # This threshold is much more conservative than the previous one.
    threshold = 1
    title = ('Group-level association between \n'
             'neg-log of parametric corrected p-values (FWER < 10%)')
    plotting.plot_stat_map(neg_log_pval, colorbar=True, 
                           #cut_coords=cut_coords,
                           threshold=threshold, title=title)
    plotting.show()

    #%threshold the second level contrast at uncorrected p < 0.001 and plot
    p_val = 0.001
    p001_uncorrected = norm.isf(p_val) #Inverse survival function
    
    proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)
    
    plotting.plot_stat_map(
        proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
        title='group, proportion true positives', vmax=1)
    
    plotting.plot_stat_map(
        z_map, threshold=p001_uncorrected, colorbar=True, display_mode='z',
        title='group (uncorrected p < 0.001)')
    plotting.show()

    
    #%threshold the second level contrast and plot it
    
    threshold = 3.1  # correponds to  p < .001, uncorrected
    display = plotting.plot_glass_brain(
        z_map, threshold=threshold, colorbar=True, plot_abs=False,
        title='vertical vs horizontal checkerboard (unc p<0.001')
    plotting.show()

    #%Threshold the resulting map: false positive rate < .001, cluster size > 10 voxels
    
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=10)
    
    #p<.001 uncorrected-thresholded map (with only clusters > 10 voxels).
    
    plotting.plot_stat_map(
        z_map,
        #thresholded_map1, 
        #cut_coords=display.cut_coords, 
        threshold=threshold1,
        title='Thresholded z map, fpr <.001, clusters > 10 voxels')
    plotting.show()
    
    #%FDR <.05 (False Discovery Rate) and no cluster-level threshold.
    
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    print('The FDR=.05 threshold is %.3g' % threshold2)
    
    #the fdr-thresholded map
    plotting.plot_stat_map(thresholded_map2,
                           #cut_coords=display.cut_coords,
                           title='Thresholded z map, expected fdr = .05',
                           threshold=threshold2
                           )
    plotting.show()

    #%Now use FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold.
    #If the data has not been intensively smoothed, we can use a simple Bonferroni correction.
    
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    
    #the Bonferroni-thresholded map
    
    plotting.plot_stat_map(thresholded_map3,
                           #cut_coords=display.cut_coords,
                           title='Thresholded z map, expected fwer < .05',
                           threshold=threshold3)
    plotting.show()
    
    #%Computing the (corrected) negative log p-values with permutation test
    
    # neg_log_pvals_permuted_ols_unmasked = non_parametric_inference(zmaps,
    #                              design_matrix=design_matrix,
    #                              second_level_contrast=contrast_val,
    #                              model_intercept=True, n_perm=1000,
    #                              two_sided_test=False, mask=None,
    #                              smoothing_fwhm=5.0, n_jobs=1)
        
    #plot the (corrected) negative log p-values
    
    # title = ('Group-level association between \n'
    #          'neg-log of non-parametric corrected p-values (FWER < 10%)')
    # plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True,
    #                        #cut_coords=cut_coords,
    #                        threshold=threshold,
    #                        title=title)
    # plotting.show()
    
    # The neg-log p-values obtained with non parametric testing are capped at 3
    # since the number of permutations is 1e3.
    # The non parametric test yields a few more discoveries
    # and is then more powerful than the usual parametric procedure.

#%%compute contrast as column in design matrix from pre-post files
#% Loading zmaps for contrasts defined like [1,-1,0,0,0]
zmaps = [] 
for k, contrast in enumerate(contrasts_fl[0:10]):
    for i, sub in enumerate(subs):
        zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_mean_space-MNI152NLin2009cAsym_{contrast}.nii.gz')

#%Step 2: Create second-level design matrix (520)
n_sub = len(subs)
design_matrix = pd.DataFrame()
design_matrix['constant'] = [1] * n_sub
design_matrix['control'] = list(itertools.chain.from_iterable([(groups['group']=='control').values.astype(int)]))
design_matrix['sport'] = list(itertools.chain.from_iterable([(groups['group']=='sport').values.astype(int)]))
design_matrix['med'] = list(itertools.chain.from_iterable([(groups['group']=='med').values.astype(int)]))
ds_sub = np.eye(len(groups['sub']))
ds_subs = pd.DataFrame(ds_sub, columns = groups['sub'])
design_matrix = pd.concat((design_matrix, ds_subs), axis=1)
design_matrix = pd.concat([design_matrix]*10, ignore_index=True)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)

#%Step 3: Second level GLM analysis
#no smoothing on a previous step
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)

#%Step 4: estimate the contrast (e.g. use the column name of the design matrix)
contrasts_sl = ['control', 'sport', 'med']

for index, contrast_val in enumerate(contrasts_sl):
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    plotting.plot_stat_map(z_map, threshold=2.0,title='%s' % contrast_id)
    
    #generate report html
    make_glm_report(second_level_model, contrast_val)

    #%Step 5: FDR-thresholded result
    #Compute the required threshold level and return the thresholded map (map, threshold)
    #add cluster_threshold
    _, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    
    #plot the second level contrast at the computed thresholds
    #bg_img default
    plotting.plot_stat_map(
        z_map, threshold=threshold, colorbar=True,
        title='Group-level %s\n'
        '(fdr=0.01)' % contrast_id)
    plotting.show()

    #%Computing corrected p-values with parametric test to compare with non parametric test
    #gives RuntimeWarning: divide by zero encountered in log10, but it's ok
    p_val = second_level_model.compute_contrast(contrast_val, output_type='p_value')
    n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
    # Correcting the p-values for multiple testing and taking negative logarithm
    neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))".format(str(n_voxels)),img=p_val)
    
    #plot the (corrected) negative log p-values for the parametric test
    
    #cut_coords = [50, -17, -3]
    # Since we are plotting negative log p-values and using a threshold equal to 1,
    # it corresponds to corrected p-values lower than 10%, meaning that there
    # is less than 10% probability to make a single false discovery
    # (90% chance that we make no false discoveries at all).
    # This threshold is much more conservative than the previous one.
    threshold = 1
    title = ('Group-level association between \n'
             'neg-log of parametric corrected p-values (FWER < 10%)')
    plotting.plot_stat_map(neg_log_pval, colorbar=True, 
                           #cut_coords=cut_coords,
                           threshold=threshold, title=title)
    plotting.show()

    #%threshold the second level contrast at uncorrected p < 0.001 and plot
    p_val = 0.001
    p001_uncorrected = norm.isf(p_val) #Inverse survival function
    
    proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)
    
    plotting.plot_stat_map(
        proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
        title='group, proportion true positives', vmax=1)
    
    plotting.plot_stat_map(
        z_map, threshold=p001_uncorrected, colorbar=True, display_mode='z',
        title='group (uncorrected p < 0.001)')
    plotting.show()

    
    #%threshold the second level contrast and plot it
    
    threshold = 3.1  # correponds to  p < .001, uncorrected
    display = plotting.plot_glass_brain(
        z_map, threshold=threshold, colorbar=True, plot_abs=False,
        title='vertical vs horizontal checkerboard (unc p<0.001')
    plotting.show()

    #%Threshold the resulting map: false positive rate < .001, cluster size > 10 voxels
    
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=10)
    
    #p<.001 uncorrected-thresholded map (with only clusters > 10 voxels).
    
    plotting.plot_stat_map(
        z_map,
        #thresholded_map1, 
        #cut_coords=display.cut_coords, 
        threshold=threshold1,
        title='Thresholded z map, fpr <.001, clusters > 10 voxels')
    plotting.show()
    
    #%FDR <.05 (False Discovery Rate) and no cluster-level threshold.
    
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    print('The FDR=.05 threshold is %.3g' % threshold2)
    
    #the fdr-thresholded map
    plotting.plot_stat_map(thresholded_map2,
                           #cut_coords=display.cut_coords,
                           title='Thresholded z map, expected fdr = .05',
                           threshold=threshold2
                           )
    plotting.show()

    #%Now use FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold.
    #If the data has not been intensively smoothed, we can use a simple Bonferroni correction.
    
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    
    #the Bonferroni-thresholded map
    
    plotting.plot_stat_map(thresholded_map3,
                           #cut_coords=display.cut_coords,
                           title='Thresholded z map, expected fwer < .05',
                           threshold=threshold3)
    plotting.show()
    
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
    plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True,
                            #cut_coords=cut_coords,
                            threshold=threshold,
                            title=title)
    plotting.show()
    
    # The neg-log p-values obtained with non parametric testing are capped at 3
    # since the number of permutations is 1e3.
    # The non parametric test yields a few more discoveries
    # and is then more powerful than the usual parametric procedure.


#%%compute contrast as column in design matrix from pre-post files
#%Loading zmaos for contrasts defined like [1,0,0,0,0]
zmaps = []
for k, contrast in enumerate(contrasts_fl[10:]):
    for i, sub in enumerate(subs):
        zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_mean_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
                
#%Step 2: Create second-level design matrix
n_sub = len(subs)
design_matrix = pd.DataFrame()
design_matrix['constant'] = [1] * n_sub
design_matrix['control'] = list(itertools.chain.from_iterable([(groups['group']=='control').values.astype(int)]))
design_matrix['sport'] = list(itertools.chain.from_iterable([(groups['group']=='sport').values.astype(int)]))
design_matrix['med'] = list(itertools.chain.from_iterable([(groups['group']=='med').values.astype(int)]))
ds_sub = np.eye(len(groups['sub']))
ds_subs = pd.DataFrame(ds_sub, columns = groups['sub'])
design_matrix = pd.concat((design_matrix, ds_subs), axis=1)
design_matrix = pd.concat([design_matrix]*5, ignore_index=True)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)

#%Step 3: Second level GLM analysis
#no smoothing on a previous step
second_level_model = SecondLevelModel()
second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)

#%Step 4: estimate the contrast (e.g. use the column name of the design matrix)
contrasts_sl = ['control', 'sport', 'med']

for index, contrast_val in enumerate(contrasts_sl):
    z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
    plotting.plot_stat_map(z_map, threshold=2.0,title='%s' % contrast_id)
    
    #generate report html
    make_glm_report(second_level_model, contrast_val)

    #%Step 5: FDR-thresholded result
    #Compute the required threshold level and return the thresholded map (map, threshold)
    #add cluster_threshold
    _, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    
    #plot the second level contrast at the computed thresholds
    #bg_img default
    plotting.plot_stat_map(
        z_map, threshold=threshold, colorbar=True,
        title='Group-level %s\n'
        '(fdr=0.01)' % contrast_id)
    plotting.show()

    #%Computing corrected p-values with parametric test to compare with non parametric test
    #gives RuntimeWarning: divide by zero encountered in log10, but it's ok
    p_val = second_level_model.compute_contrast(contrast_val, output_type='p_value')
    n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
    # Correcting the p-values for multiple testing and taking negative logarithm
    neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))".format(str(n_voxels)),img=p_val)
    
    #plot the (corrected) negative log p-values for the parametric test
    
    #cut_coords = [50, -17, -3]
    # Since we are plotting negative log p-values and using a threshold equal to 1,
    # it corresponds to corrected p-values lower than 10%, meaning that there
    # is less than 10% probability to make a single false discovery
    # (90% chance that we make no false discoveries at all).
    # This threshold is much more conservative than the previous one.
    threshold = 1
    title = ('Group-level association between \n'
             'neg-log of parametric corrected p-values (FWER < 10%)')
    plotting.plot_stat_map(neg_log_pval, colorbar=True, 
                           #cut_coords=cut_coords,
                           threshold=threshold, title=title)
    plotting.show()

    #%threshold the second level contrast at uncorrected p < 0.001 and plot
    p_val = 0.001
    p001_uncorrected = norm.isf(p_val) #Inverse survival function
    
    proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)
    
    plotting.plot_stat_map(
        proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
        title='group, proportion true positives', vmax=1)
    
    plotting.plot_stat_map(
        z_map, threshold=p001_uncorrected, colorbar=True, display_mode='z',
        title='group (uncorrected p < 0.001)')
    plotting.show()

    
    #%threshold the second level contrast and plot it
    
    threshold = 3.1  # correponds to  p < .001, uncorrected
    display = plotting.plot_glass_brain(
        z_map, threshold=threshold, colorbar=True, plot_abs=False,
        title='vertical vs horizontal checkerboard (unc p<0.001')
    plotting.show()

    #%Threshold the resulting map: false positive rate < .001, cluster size > 10 voxels
    
    thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=10)
    
    #p<.001 uncorrected-thresholded map (with only clusters > 10 voxels).
    
    plotting.plot_stat_map(
        z_map,
        #thresholded_map1, 
        #cut_coords=display.cut_coords, 
        threshold=threshold1,
        title='Thresholded z map, fpr <.001, clusters > 10 voxels')
    plotting.show()
    
    #%FDR <.05 (False Discovery Rate) and no cluster-level threshold.
    
    thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
    print('The FDR=.05 threshold is %.3g' % threshold2)
    
    #the fdr-thresholded map
    plotting.plot_stat_map(thresholded_map2,
                           #cut_coords=display.cut_coords,
                           title='Thresholded z map, expected fdr = .05',
                           threshold=threshold2
                           )
    plotting.show()

    #%Now use FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold.
    #If the data has not been intensively smoothed, we can use a simple Bonferroni correction.
    
    thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
    print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
    
    #the Bonferroni-thresholded map
    
    plotting.plot_stat_map(thresholded_map3,
                           #cut_coords=display.cut_coords,
                           title='Thresholded z map, expected fwer < .05',
                           threshold=threshold3)
    plotting.show()
    
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
    plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True,
                            #cut_coords=cut_coords,
                            threshold=threshold,
                            title=title)
    plotting.show()
    
    # The neg-log p-values obtained with non parametric testing are capped at 3
    # since the number of permutations is 1e3.
    # The non parametric test yields a few more discoveries
    # and is then more powerful than the usual parametric procedure.

#%%each mean pre-post FL contrast separately
#% Loading zmaps for contrasts defined like [1,-1,0,0,0]
#%Step 2: Create second-level design matrix
n_sub = len(subs)
design_matrix = pd.DataFrame()
design_matrix['constant'] = [1] * n_sub
design_matrix['control'] = list(itertools.chain.from_iterable([(groups['group']=='control').values.astype(int)]))
design_matrix['sport'] = list(itertools.chain.from_iterable([(groups['group']=='sport').values.astype(int)]))
design_matrix['med'] = list(itertools.chain.from_iterable([(groups['group']=='med').values.astype(int)]))
#ds_sub = np.eye(len(groups['sub']))
#ds_subs = pd.DataFrame(ds_sub, columns = groups['sub'])
#design_matrix = pd.concat((design_matrix, ds_subs), axis=1)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)
    
contrasts_sl = ['control', 'sport', 'med']
    
for k, contrast in enumerate(contrasts_fl[0:10]):
    zmaps = []
    print('Contrast % 2i out of %i: %s' % (k + 1, len(contrasts_fl[0:10]), k))
    for i, sub in enumerate(subs):
        zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_mean_space-MNI152NLin2009cAsym_{contrast}.nii.gz')
    
    #%Step 3: Second level GLM analysis 
    #no smoothing on a previous step
    second_level_model = SecondLevelModel()
    second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)
    
    #% Step 4: estimate the contrast (e.g. use the column name of the design matrix)
    for index, contrast_val in enumerate(contrasts_sl):
        z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
        plotting.plot_stat_map(z_map, threshold=2.0, title='%s' % contrast_val)
        
        #generate report html
        make_glm_report(second_level_model, contrast_val)
    
        #%Step 5: FDR-thresholded result
        #Compute the required threshold level and return the thresholded map (map, threshold)
        #add cluster_threshold
        _, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
        
        #plot the second level contrast at the computed thresholds
        #bg_img default
        plotting.plot_stat_map(
            z_map, threshold=threshold, colorbar=True,
            title='Group-level %s\n'
            '(fdr=0.01)' % contrast_val)
        plotting.show()
    
        #%Computing corrected p-values with parametric test to compare with non parametric test
        #gives RuntimeWarning: divide by zero encountered in log10, but it's ok
        p_val = second_level_model.compute_contrast(contrast_val, output_type='p_value')
        n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
        # Correcting the p-values for multiple testing and taking negative logarithm
        neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))".format(str(n_voxels)),img=p_val)
        
        #plot the (corrected) negative log p-values for the parametric test
        
        #cut_coords = [50, -17, -3]
        # Since we are plotting negative log p-values and using a threshold equal to 1,
        # it corresponds to corrected p-values lower than 10%, meaning that there
        # is less than 10% probability to make a single false discovery
        # (90% chance that we make no false discoveries at all).
        # This threshold is much more conservative than the previous one.
        threshold = 1
        title = ('Group-level association between \n'
                 'neg-log of parametric corrected p-values (FWER < 10%)')
        plotting.plot_stat_map(neg_log_pval, colorbar=True, 
                               #cut_coords=cut_coords,
                               threshold=threshold, title=title)
        plotting.show()
    
        #%threshold the second level contrast at uncorrected p < 0.001 and plot
        p_val = 0.001
        p001_uncorrected = norm.isf(p_val) #Inverse survival function
        
        proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)
        
        plotting.plot_stat_map(
            proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
            title='group, proportion true positives', vmax=1)
        
        plotting.plot_stat_map(
            z_map, threshold=p001_uncorrected, colorbar=True, display_mode='z',
            title='group (uncorrected p < 0.001)')
        plotting.show()
    
        
        #%threshold the second level contrast and plot it
        
        threshold = 3.1  # correponds to  p < .001, uncorrected
        display = plotting.plot_glass_brain(
            z_map, threshold=threshold, colorbar=True, plot_abs=False,
            title='vertical vs horizontal checkerboard (unc p<0.001')
        plotting.show()
    
        #%Threshold the resulting map: false positive rate < .001, cluster size > 10 voxels
        
        thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=10)
        
        #p<.001 uncorrected-thresholded map (with only clusters > 10 voxels).
        
        plotting.plot_stat_map(
            z_map,
            #thresholded_map1, 
            #cut_coords=display.cut_coords, 
            threshold=threshold1,
            title='Thresholded z map, fpr <.001, clusters > 10 voxels')
        plotting.show()
        
        #%FDR <.05 (False Discovery Rate) and no cluster-level threshold.
        
        thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
        print('The FDR=.05 threshold is %.3g' % threshold2)
        
        #the fdr-thresholded map
        plotting.plot_stat_map(thresholded_map2,
                               #cut_coords=display.cut_coords,
                               title='Thresholded z map, expected fdr = .05',
                               threshold=threshold2
                               )
        plotting.show()
    
        #%Now use FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold.
        #If the data has not been intensively smoothed, we can use a simple Bonferroni correction.
        
        thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
        print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
        
        #the Bonferroni-thresholded map
        
        plotting.plot_stat_map(thresholded_map3,
                               #cut_coords=display.cut_coords,
                               title='Thresholded z map, expected fwer < .05',
                               threshold=threshold3)
        plotting.show()
        
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
        plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True,
                                #cut_coords=cut_coords,
                                threshold=threshold,
                                title=title)
        plotting.show()
        
        # The neg-log p-values obtained with non parametric testing are capped at 3
        # since the number of permutations is 1e3.
        # The non parametric test yields a few more discoveries
        # and is then more powerful than the usual parametric procedure.

#%%each mean pre-post FL contrast separately
#%Loading zmaos for contrasts defined like [1,0,0,0,0]
#%Step 2: Create second-level design matrix
n_sub = len(subs)
design_matrix = pd.DataFrame()
design_matrix['constant'] = [1] * n_sub
design_matrix['control'] = list(itertools.chain.from_iterable([(groups['group']=='control').values.astype(int)]))
design_matrix['sport'] = list(itertools.chain.from_iterable([(groups['group']=='sport').values.astype(int)]))
design_matrix['med'] = list(itertools.chain.from_iterable([(groups['group']=='med').values.astype(int)]))
#ds_sub = np.eye(len(groups['sub']))
#ds_subs = pd.DataFrame(np.concatenate([ds_sub, ds_sub]), columns = groups['sub'])
#design_matrix = pd.concat((design_matrix, ds_subs), axis=1)
#plot
ax = plot_design_matrix(design_matrix)
ax.get_images()[0].set_clim(0, 0.2)

contrasts_sl = ['control', 'sport', 'med']

for k, contrast in enumerate(contrasts_fl[10:]):
    zmaps = []
    for i, sub in enumerate(subs):
        zmaps.append(f'{top_dir}/GLM/FLA/voxel/{sub}_mean_space-MNI152NLin2009cAsym_{contrast}.nii.gz')

    #%Step 3: Second level GLM analysis
    #no smoothing on a previous step
    second_level_model = SecondLevelModel()
    second_level_model = second_level_model.fit(zmaps, design_matrix=design_matrix)
    
    #% Step 4: estimate the contrast (e.g. use the column name of the design matrix)
    for index, contrast_val in enumerate(contrasts_sl):
        z_map = second_level_model.compute_contrast(contrast_val, output_type='z_score')
        plotting.plot_stat_map(z_map, threshold=2.0,title='%s' % contrast_id)
        
        #generate report html
        make_glm_report(second_level_model, contrast_val)
    
        #%Step 5: FDR-thresholded result
        #Compute the required threshold level and return the thresholded map (map, threshold)
        #add cluster_threshold
        _, threshold = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
        
        #plot the second level contrast at the computed thresholds
        #bg_img default
        plotting.plot_stat_map(
            z_map, threshold=threshold, colorbar=True,
            title='Group-level %s\n'
            '(fdr=0.01)' % contrast_id)
        plotting.show()
    
        #%Computing corrected p-values with parametric test to compare with non parametric test
        #gives RuntimeWarning: divide by zero encountered in log10, but it's ok
        p_val = second_level_model.compute_contrast(contrast_val, output_type='p_value')
        n_voxels = np.sum(get_data(second_level_model.masker_.mask_img_))
        # Correcting the p-values for multiple testing and taking negative logarithm
        neg_log_pval = math_img("-np.log10(np.minimum(1, img * {}))".format(str(n_voxels)),img=p_val)
        
        #plot the (corrected) negative log p-values for the parametric test
        
        #cut_coords = [50, -17, -3]
        # Since we are plotting negative log p-values and using a threshold equal to 1,
        # it corresponds to corrected p-values lower than 10%, meaning that there
        # is less than 10% probability to make a single false discovery
        # (90% chance that we make no false discoveries at all).
        # This threshold is much more conservative than the previous one.
        threshold = 1
        title = ('Group-level association between \n'
                 'neg-log of parametric corrected p-values (FWER < 10%)')
        plotting.plot_stat_map(neg_log_pval, colorbar=True, 
                               #cut_coords=cut_coords,
                               threshold=threshold, title=title)
        plotting.show()
    
        #%threshold the second level contrast at uncorrected p < 0.001 and plot
        p_val = 0.001
        p001_uncorrected = norm.isf(p_val) #Inverse survival function
        
        proportion_true_discoveries_img = cluster_level_inference(z_map, threshold=[3, 4, 5], alpha=.05)
        
        plotting.plot_stat_map(
            proportion_true_discoveries_img, threshold=0., colorbar=True, display_mode='z',
            title='group, proportion true positives', vmax=1)
        
        plotting.plot_stat_map(
            z_map, threshold=p001_uncorrected, colorbar=True, display_mode='z',
            title='group (uncorrected p < 0.001)')
        plotting.show()
    
        
        #%threshold the second level contrast and plot it
        
        threshold = 3.1  # correponds to  p < .001, uncorrected
        display = plotting.plot_glass_brain(
            z_map, threshold=threshold, colorbar=True, plot_abs=False,
            title='vertical vs horizontal checkerboard (unc p<0.001')
        plotting.show()
    
        #%Threshold the resulting map: false positive rate < .001, cluster size > 10 voxels
        
        thresholded_map1, threshold1 = threshold_stats_img(z_map, alpha=.001, height_control='fpr', cluster_threshold=10)
        
        #p<.001 uncorrected-thresholded map (with only clusters > 10 voxels).
        
        plotting.plot_stat_map(
            z_map,
            #thresholded_map1, 
            #cut_coords=display.cut_coords, 
            threshold=threshold1,
            title='Thresholded z map, fpr <.001, clusters > 10 voxels')
        plotting.show()
        
        #%FDR <.05 (False Discovery Rate) and no cluster-level threshold.
        
        thresholded_map2, threshold2 = threshold_stats_img(z_map, alpha=.05, height_control='fdr')
        print('The FDR=.05 threshold is %.3g' % threshold2)
        
        #the fdr-thresholded map
        plotting.plot_stat_map(thresholded_map2,
                               #cut_coords=display.cut_coords,
                               title='Thresholded z map, expected fdr = .05',
                               threshold=threshold2
                               )
        plotting.show()
    
        #%Now use FWER <.05 (Family-Wise Error Rate) and no cluster-level threshold.
        #If the data has not been intensively smoothed, we can use a simple Bonferroni correction.
        
        thresholded_map3, threshold3 = threshold_stats_img(z_map, alpha=.05, height_control='bonferroni')
        print('The p<.05 Bonferroni-corrected threshold is %.3g' % threshold3)
        
        #the Bonferroni-thresholded map
        
        plotting.plot_stat_map(thresholded_map3,
                               #cut_coords=display.cut_coords,
                               title='Thresholded z map, expected fwer < .05',
                               threshold=threshold3)
        plotting.show()
        
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
        plotting.plot_stat_map(neg_log_pvals_permuted_ols_unmasked, colorbar=True,
                                #cut_coords=cut_coords,
                                threshold=threshold,
                                title=title)
        plotting.show()
        
        # The neg-log p-values obtained with non parametric testing are capped at 3
        # since the number of permutations is 1e3.
        # The non parametric test yields a few more discoveries
        # and is then more powerful than the usual parametric procedure.


















#%%Step 2: Loading Power pacellation
power264 = datasets.fetch_coords_power_2011()
power264_coords = np.vstack((power264.rois['x'], power264.rois['y'], power264.rois['z'])).T

# Creating masker file
power264_maps = input_data.NiftiSpheresMasker(
    seeds = power264_coords, 
    radius = 5,
)

power264_networks = np.asarray(pd.read_csv(f'{top_dir}/GLM/modules.txt', header=None)[0])
power264_colors = {'AU':'#d182c6',
                   'CER':'#9fc5e8',
                   'CO':'#7d009d', 
                   'DA':'#75df33', 
                   'DM':'#ed1126', 
                   'FP':'#f6e838', 
                   'MEM':'#bebab5',
                   'SAL':'#2a2a2a',
                   'SOM':'#6ccadf',
                   'SUB':'#980000',
                   'UNC':'#f58c00',
                   'VA':'#00a074', 
                   'VIS':'#5131ac'}

power_palette = (sns.color_palette(power264_colors.values()))
sns.palplot(power_palette)

#%%Step 5: Contrast analysis
group_zmap = second_level_model.compute_contrast(contrast, output_type='z_score')

# Extracting z-score values for each Power parcellation node
power_zmap = power264_maps.fit_transform(group_zmap)

# Plotting activations on glass brain
glass_brain = plot_glass_brain(group_zmap, colorbar=True, display_mode='ortho', plot_abs=False, threshold=8, cmap='RdBu_r')
plt.savefig(f'{out_dir}figures/GLM_zmap_glass_brain_session.png', bbox_inches="tight", dpi=300)

#%%
# Visualization on Power ROIs coordinates
norm = plt.Normalize(vmin=-14, vmax=14)
colors = np.squeeze(plt.cm.RdBu_r(norm(power_zmap)))
plot_connectome(np.zeros((264,264)), power264_coords, node_color=colors)

plt.savefig(f'{out_dir}figures/GLM_Power_nodes_glass_brain.pdf', bbox_inches="tight", dpi=300)

#%%
fig, ax = plt.subplots(figsize=(0.2, 3))
cb1 = mpl.colorbar.ColorbarBase(ax, cmap='RdBu_r',
                                norm=norm,
                                orientation='vertical')
plt.savefig(f'{out_dir}figures/GLM_colorbar.pdf', bbox_inches="tight", dpi=300)


#%%





#%%Step 6: Distribution of activations over large-scale systems

timeseries = np.load(f'{data_dir}GLM_power_2b_minus_1b_zmap_timeseries.npy')
subjects_data_clean_lm = subjects_data_clean

networks_binary = pd.get_dummies(power264_networks)
mean_zscore_group = pd.DataFrame()

mean_zscore_group_np = np.empty((subjects_data_clean.shape[0], 4, 13))

for i, sub in enumerate(subjects_data_clean_lm['sub']):
    for j, ses in enumerate(sessions):
        for n, net in enumerate(networks_binary.columns):
            
            mean_zscore = timeseries[i, j, networks_binary[net].astype('bool')].mean()
            mean_zscore_group_np[i, j, n] = mean_zscore
            mean_zscore_group = pd.concat([mean_zscore_group, pd.DataFrame({
                "Subject":sub,
                "Session": ses,
                "Group": subjects_data_clean_lm['group'][i],
                "Network": net,
                "Activation": mean_zscore}, index=[0])],   axis=0)
                     
mean_zscore_group.to_csv('/home/finc/Dropbox/Projects/LearningBrain/data/neuroimaging/04-glm/glm_results_all.csv', index=False)
np.save('/home/finc/Dropbox/Projects/LearningBrain/data/neuroimaging/04-glm/glm_results_all.npy', mean_zscore_group_np)

#%%
mean_zscore_group_clean = mean_zscore_group[~mean_zscore_group['Subject'].isin(high_motion)]
sort_result = mean_zscore_group.groupby(["Network"])["Activation"].aggregate(np.mean).reset_index().sort_values("Activation", 
                                                                                                                ascending=False)
power_palette_sorted = [power_palette[index] for index in sort_result.index.to_list()]

plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'Helvetica'

small = 20
medium = 20
bigger = 20

plt.rc('font', size=small)          # controls default text sizes
plt.rc('axes', titlesize=small)     # fontsize of the axes title
plt.rc('axes', linewidth=2.2)
plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=small)    # legend fontsize
plt.rc('figure', titlesize=bigger)  # fontsize of the figure title
plt.rc('lines', linewidth=2.2, color='gray')

sns.catplot(x='Network', y='Activation', data=mean_zscore_group_clean, kind="bar", palette=power_palette_sorted, height=5, aspect=2.5, order=sort_result['Network'])
plt.ylabel("Mean z-score (2-back vs. 1-back)")
plt.xlabel("System")
plt.savefig('../figures/GLM_power_networks.pdf', bbox_inches="tight", dpi=300)

#%%
import seaborn as sns

sns.catplot(x='Session', y='Activation', hue='Group', row='Network', data=mean_zscore_group, kind='box')

#%%
groups = np.unique(subjects_data_clean_lm['group'])

control_filter = (subjects_data_clean_lm['group']=='Control').values
experimental_filter = (subjects_data_clean_lm['group']=='Experimental').values

norm = plt.Normalize(vmin=-5, vmax=5)

for group in groups:
    for s, ses in enumerate(sessions):
        colors = plt.cm.RdBu_r(norm(timeseries_clean[(subjects_data_clean_lm['group']==group).values, s, :].mean(axis=0)))
        plot_connectome(np.zeros((264,264)), power264_coords, node_color=colors, title=f"Group: {group}, Session: {ses}")

#%%
import matplotlib as mpl
fig, ax = plt.subplots(figsize=(0.2, 4))
cb1 = mpl.colorbar.ColorbarBase(ax, cmap='RdBu_r',
                                norm=norm,
                                orientation='vertical')












