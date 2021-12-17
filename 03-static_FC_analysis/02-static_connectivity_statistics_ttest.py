#Static connectivity t-test
#Last edited: 20-01-2021

import sys
sys.path.append("..")
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import nilearn
import nilearn.connectome

from nilearn import datasets, plotting, input_data
from nilearn.connectome import sym_matrix_to_vec
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.stats import multitest
from scipy import stats
from mne import viz

import seaborn as sns
sns.reset_orig()

#%%Step 1: Loading data
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/func_connect/05-fc_statistics/'
suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
denoising = '_denoised_acompcor_task_effects.nii.gz'

groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])
sess = ['ses-1', 'ses-3']
cond = ['1', '2', '3', '4', '5']
tasks = ['nback']
task = 'nback'
n_scans_ses_1_run_1 = pd.Series.tolist(groups['n_scans_ses-1_run-1'])
n_scans_ses_3_run_1 = pd.Series.tolist(groups['n_scans_ses-3_run-1'])
n_scans_ses_1_run_2 = pd.Series.tolist(groups['n_scans_ses-1_run-2'])
n_scans_ses_3_run_2 = pd.Series.tolist(groups['n_scans_ses-3_run-2'])
t_r = 3
alpha = 0.05 
subs_sport = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport'])])
subs_med = pd.Series.tolist(groups['sub'][groups['group'].isin(['med'])])
subs_control = pd.Series.tolist(groups['sub'][groups['group'].isin(['control'])])
#here [0] is to convert from tuple to array
idx_subs_sport = np.array(np.where(np.isin(subs,subs_sport)))[0]
idx_subs_med = np.array(np.where(np.isin(subs,subs_med)))[0]
idx_subs_control = np.array(np.where(np.isin(subs,subs_control)))[0]
n_subs_sport = len(subs_sport)
n_subs_med = len(subs_med)
n_subs_control = len(subs_control)

idx_subs_grouped = [idx_subs_sport, idx_subs_med, idx_subs_control]
n_subs_grouped = [n_subs_sport, n_subs_med, n_subs_control]
subs_labels_grouped = ['sport', 'med', 'control']

multitest_methods_names = ['bonferroni','holm','holm-sidak','simes-hochberg',
                           'fdr_bh', 'fdr_by','fdr_tsbh','fdr_tsbky','fdr_gbs']
                           

#%%Step 2: loading atlases for labels on matrices
# Loading Power ROIs coordinates
power = datasets.fetch_coords_power_2011()
power_coords = np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T
#doesn't work
power_labels = power_coords

#Harvard-Oxford 2 atlases (48+21+2=71)
ho_cort = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
ho_sub = datasets.fetch_atlas_harvard_oxford('sub-maxprob-thr25-2mm')
ho_cort_labels = ho_cort.labels
ho_sub_labels = ho_sub.labels
ho_labels = ho_cort_labels + ho_sub_labels
#drop bg at 0 and 49
ho_labels.remove('Background')
ho_labels.remove('Background')
#drop 59-61: right cerebral white matter, cerebral cortex, lateral ventrical, 
#drop 48-50: left cerebral white matter, cerebral cortex, lateral ventrical, 
del ho_labels[59:62]
del ho_labels[48:51]

#Talairach (72)
talairach = datasets.fetch_atlas_talairach('ba')
talairach_filename = talairach.maps
talairach_labels = talairach.labels
#drop bg at 0
talairach_labels.remove('Background')

#Craddock
craddock = datasets.fetch_atlas_craddock_2012()
craddock_filename = craddock.scorr_mean
craddock_masker = input_data.NiftiMapsMasker(maps_img=craddock_filename, 
                        standardize=True, memory='nilearn_cache', verbose=5)

aal = datasets.fetch_atlas_aal()
aal_filename = aal.maps
aal_labels = aal.labels

#nback_power = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_power_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)
#nback_craddock = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_craddock_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)
nback_ho_cort = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_harvard-oxford_cort_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)
nback_ho_sub = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_harvard-oxford_sub_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)
nback_ho = np.concatenate((nback_ho_cort, nback_ho_sub), axis=3)
nback_talairach = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_talairach_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)
nback_aal = np.load(f'{top_dir}/func_connect/02-extracted_timeseries/nback/LB_nback_timeseries_aal_denoised_acompcor_no_smooth_run-1.npy', allow_pickle=True)

nback = np.asarray([[nback_ho, 'nback_harvard-oxford'],
                    [nback_talairach, 'nback_talairach'],
                    [nback_aal, 'nback_aal'],
                    #[nback_power, 'nback_power'], 
                    #[nback_craddock, 'nback_craddock']
                    ])

atlas_labels =  np.asarray([[ho_labels, 'harvard-oxford'],
                    [talairach_labels, 'nback_talairach'],
                    [aal_labels, 'nback_aal'],
                    #[power_labels, 'power'],
                    #[craddock_filename, 'craddock']
                    ])

experim_conditions = ['0back', '1back', '2back', '4back', 'fix']

#%%Step 3: Load correlation matrices
#atlas:
#-power
#-Harvard-Oxford cortical-subcortical atlases (Makris 2006)
#-Talairach atlas
#-Craddock atlas

#correlation matrices weighted atlas-based 
correl_matrices_wei_z = []
for p, (i, atlas) in enumerate(nback):
    correlation_matrices_wei_z = np.load(f'{top_dir}/func_connect/04-correlation_matrices/static/LB_{nback[p,1]}_static_correlation_matrices_wei_z.npy')
    if atlas == 'nback_harvard-oxford':
        #correlation_matrices_wei_z = correlation_matrices_wei_z[:,:,:,[0:48,51:59,62:-1],[0:48,51:59,62:-1]]
        correlation_matrices_wei_z = np.delete(correlation_matrices_wei_z,[48, 49, 50, 59, 60, 61],3)
        correlation_matrices_wei_z = np.delete(correlation_matrices_wei_z,[48, 49, 50, 59, 60, 61],4)
    correl_matrices_wei_z.append(correlation_matrices_wei_z)

#correlation matrices atlas-based non-weighted
# correl_matrices_z = []
# for p, (i, atlas) in enumerate(nback):
#     correlation_matrices_z = np.load(f'{top_dir}/func_connect/04-correlation_matrices/static/LB_{nback[p,1]}_static_correlation_matrices_z.npy')
#     correl_matrices_z.append(correlation_matrices_z)

#%%visualize weighted atlas-based correlations
for p, atlas in enumerate(nback):
    for ses in range(len(sess)):
        for con in range(len(cond)):    
            correl_matrix_wei_z = correl_matrices_wei_z[p]
            correl_matrix_wei_z = correl_matrix_wei_z[:,ses,con,:,:]        
            plotting.plot_matrix(np.mean(correl_matrix_wei_z, axis=0), vmin=-1., vmax=1., colorbar=True,
                  title=f'{nback[p,1]} ses-{ses} condition-{con} correlation matrix weighted')

# for p, atlas in enumerate(nback):
#     for ses in range(len(sess)):
#         for con in range(len(cond)):    
#             correl_matrix_z = correl_matrices_z[p]
#             correl_matrix_z = correl_matrix_z[:,ses,:,:]
#             plotting.plot_matrix(np.mean(correl_matrix_z, axis=0), vmin=-1., vmax=1., colorbar=True,
#                   title=f'{nback[p,1]} ses-{ses} condition-{con} correlation matrix non-weighted')

#%%
#check
#https://seaborn.pydata.org/examples/structured_heatmap.html
p = 0
atlas = 'harvard-oxford'
ses = 'ses-1'
for con in range(len(cond)): 
    correl_matrix_wei_z = correl_matrices_wei_z[p]
    correl_matrix_wei_z = correl_matrix_wei_z[:,0,con,:,:]       
    title=f'{atlas} atlas {ses} condition-{experim_conditions[con]} weighted correlation matrix ' 
    fig = plt.figure(figsize=(20, 20))
    #fig.suptitle(title)
    ax = fig.add_subplot(111)
    im = plt.imshow(np.mean(correl_matrix_wei_z, axis=0), cmap=plt.cm.RdBu_r, vmin=-1., vmax=1.)
    ax.set_title(title)
    ax.set_xticks(range(len(ho_labels)))
    ax.set_xticklabels(ho_labels, rotation='vertical')
    ax.set_yticks(range(len(ho_labels)))
    ax.set_yticklabels(ho_labels, rotation='horizontal')    
    fig.colorbar(im)
    fig.show()
    fig.savefig(f'{out_dir}/{title}')
    
    fig = plt.figure(figsize=(20, 20), facecolor='white')
    #ax = fig.add_subplot(111)
    viz.plot_connectivity_circle(np.mean(correl_matrix_wei_z, axis=0),ho_labels,title=title,facecolor='white', textcolor='black',fig=fig, show=True) #colormap='RdBu_r', vmin=-1, vmax=1,
    #plt.show()
    fig.savefig(f'{out_dir}/{title}_circle')

#%%pre/post t-test for each group
list_strings = []

for group in range(len(subs_labels_grouped)):
    idx_subs = idx_subs_grouped[group]
    n_subs = n_subs_grouped[group]
    string = f'results for {subs_labels_grouped[group]} group'
    list_strings.append(string)
    print(f'results for {subs_labels_grouped[group]} group')
    
    for i, (j, atlas) in enumerate(nback):
        for con in range(len(cond)):    
            corr_matr_s0 = correl_matrices_wei_z[i]
            corr_matr_s0 = corr_matr_s0[idx_subs,0,con,:,:]
                    
            corr_matr_s1 = correl_matrices_wei_z[i]
            corr_matr_s1 = corr_matr_s1[idx_subs,1,con,:,:]
            
            #take only lower triangle
            corr_vec_s0 = []
            corr_vec_s1 = []
            for k in range(n_subs):
                matrix = corr_matr_s0[k,:,:]
                corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
                corr_vec_s0.append(corr_vec)
                matrix = corr_matr_s1[k,:,:]
                corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
                corr_vec_s1.append(corr_vec)
            corr_vec_s0 = np.array(corr_vec_s0)
            corr_vec_s1 = np.array(corr_vec_s1) 
            
            t_val, p_val = ttest_rel(corr_vec_s0, corr_vec_s1, 0)
    
            n_samples, n_tests = corr_vec_s0.shape
            threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)
            for method in multitest_methods_names:
                output = multitest.multipletests(p_val, alpha=alpha, method=method)
                reject = output[0]
                pval_corrected = output[1]
                #threshold_bonferroni = stats.t.ppf(1.0 - alpha / n_tests, n_samples - 1)
                if True in reject:
                    string = f'{method}-corrected significant results {nback[i,1]} condition-{experim_conditions[con]}'
                    list_strings.append(string)
                    print(f'{method}-corrected significant results {nback[i,1]} condition-{experim_conditions[con]}')
                    correlation_matrices_wei_z = correl_matrices_wei_z[i]
                    diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                    p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                    
                    #get locations of the significant results to get labels
                    a1,a2 = np.where(p_vals_matrix <= 0.05)
                    sign_labels1 = []
                    sign_labels2 = []
                    atlas_lab = atlas_labels[i,0]
                    for k in range(len(a1)):
                        sign_labels1.append(atlas_lab[a1[k]])
                        sign_labels2.append(atlas_lab[a2[k]])
                    
                    string = f'atlas: {atlas_labels[i,1]}, condition-{experim_conditions[con]}'
                    list_strings.append(string)
                    print(string)
                    string = f'The connectivity strength differs significantly between pre and post for {subs_labels_grouped[group]} group'
                    list_strings.append(string)
                    print(string)
                    for k in range(int(len(sign_labels1)/2)):
                        string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                        list_strings.append(string)
                        print(string)
                else:
                    string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                    list_strings.append(string)
                    print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
        
with open(f'{out_dir}static/all_ttests_pre-post_matrices_wei.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item)
            
#%%post sport/control
for i, (j, atlas) in enumerate(nback):
    for con in range(len(cond)):    
        corr_matr_s0 = correl_matrices_wei_z[i]
        corr_matr_s0 = corr_matr_s0[idx_subs_sport,0,con,:,:]
                
        corr_matr_s1 = correl_matrices_wei_z[i]
        corr_matr_s1 = corr_matr_s1[idx_subs_control,1,con,:,:]
        
        #take only lower triangle
        corr_vec_s0 = []
        corr_vec_s1 = []
        for k in range(n_subs_sport):
            matrix = corr_matr_s0[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s0.append(corr_vec)
        for k in range(n_subs_control):
            matrix = corr_matr_s1[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s1.append(corr_vec)
        corr_vec_s0 = np.array(corr_vec_s0)
        corr_vec_s1 = np.array(corr_vec_s1) 
        
        t_val, p_val = ttest_ind(corr_vec_s0, corr_vec_s1, 0)
         
        n_samples, n_tests = corr_vec_s0.shape
        threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)
        
        for method in multitest_methods_names:
            output = multitest.multipletests(p_val, alpha=alpha, method=method)
            reject = output[0]
            pval_corrected = output[1]

            if True in reject:
                string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                list_strings.append(string)
                print(string)
                correlation_matrices_wei_z = correl_matrices_wei_z[i]
                diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                
                #get locations of the significant results to get labels
                a1,a2 = np.where(p_vals_matrix <= 0.05)
                sign_labels1 = []
                sign_labels2 = []
                atlas_lab = atlas_labels[i,0]
                for k in range(len(a1)):
                    sign_labels1.append(atlas_lab[a1[k]])
                    sign_labels2.append(atlas_lab[a2[k]])
                
                string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                list_strings.append(string)
                print(string)
                string = f'The connectivity strength differs significantly between sport and control groups post'
                list_strings.append(string)
                print(string)
                for k in range(int(len(sign_labels1)/2)):
                    string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                    list_strings.append(string)
                    print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
            else:
                string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                list_strings.append(string)
                print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
        
with open(f'{out_dir}static/ttests_post_sport-control_matrices_wei.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item)

#%%post sport/med
for i, (j, atlas) in enumerate(nback):
    for con in range(len(cond)):    
        corr_matr_s0 = correl_matrices_wei_z[i]
        corr_matr_s0 = corr_matr_s0[idx_subs_sport,0,con,:,:]
                
        corr_matr_s1 = correl_matrices_wei_z[i]
        corr_matr_s1 = corr_matr_s1[idx_subs_med,1,con,:,:]
        
        #take only lower triangle
        corr_vec_s0 = []
        corr_vec_s1 = []
        for k in range(n_subs_sport):
            matrix = corr_matr_s0[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s0.append(corr_vec)
        for k in range(n_subs_med):
            matrix = corr_matr_s1[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s1.append(corr_vec)
        corr_vec_s0 = np.array(corr_vec_s0)
        corr_vec_s1 = np.array(corr_vec_s1) 
        
        t_val, p_val = ttest_ind(corr_vec_s0, corr_vec_s1, 0)
         
        n_samples, n_tests = corr_vec_s0.shape
        threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)
        
        for method in multitest_methods_names:
            output = multitest.multipletests(p_val, alpha=alpha, method=method)
            reject = output[0]
            pval_corrected = output[1]

            if True in reject:
                string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                list_strings.append(string)
                print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
                correlation_matrices_wei_z = correl_matrices_wei_z[i]
                diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
                p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                
                #get locations of the significant results to get labels
                a1,a2 = np.where(p_vals_matrix <= 0.05)
                sign_labels1 = []
                sign_labels2 = []
                atlas_lab = atlas_labels[i,0]
                for k in range(len(a1)):
                    sign_labels1.append(atlas_lab[a1[k]])
                    sign_labels2.append(atlas_lab[a2[k]])
                
                string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                list_strings.append(string)
                print(string)
                string = f'The connectivity strength differs significantly between sport and meditation groups post'
                list_strings.append(string)
                print(string)
                for k in range(int(len(sign_labels1)/2)):
                    string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                    list_strings.append(string)
                    print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
            else:
                string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                list_strings.append(string)
                print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
        
with open(f'{out_dir}static/ttests_post_sport-meditation_matrices_wei.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item)
#%%post med/control
for i, (j, atlas) in enumerate(nback):
    for con in range(len(cond)):
     corr_matr_s0 = correl_matrices_wei_z[i]
     corr_matr_s0 = corr_matr_s0[idx_subs_med,0,con,:,:]
             
     corr_matr_s1 = correl_matrices_wei_z[i]
     corr_matr_s1 = corr_matr_s1[idx_subs_control,1,con,:,:]
     
     #take only lower triangle
     corr_vec_s0 = []
     corr_vec_s1 = []
     for k in range(n_subs_med):
         matrix = corr_matr_s0[k,:,:]
         corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
         corr_vec_s0.append(corr_vec)
     for k in range(n_subs_control):
         matrix = corr_matr_s1[k,:,:]
         corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
         corr_vec_s1.append(corr_vec)
     corr_vec_s0 = np.array(corr_vec_s0)
     corr_vec_s1 = np.array(corr_vec_s1) 
     
     t_val, p_val = ttest_ind(corr_vec_s0, corr_vec_s1, 0)
      
     n_samples, n_tests = corr_vec_s0.shape
     threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)
     for method in multitest_methods_names:
        output = multitest.multipletests(p_val, alpha=alpha, method=method)
        reject = output[0]
        pval_corrected = output[1]

        if True in reject:
            string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
            list_strings.append(string)
            print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
            correlation_matrices_wei_z = correl_matrices_wei_z[i]
            diagonal = np.ones(len(correlation_matrices_wei_z[0,0,0,:,0]))
            p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
            
            #get locations of the significant results to get labels
            a1,a2 = np.where(p_vals_matrix <= 0.05)
            sign_labels1 = []
            sign_labels2 = []
            atlas_lab = atlas_labels[i,0]
            for k in range(len(a1)):
                sign_labels1.append(atlas_lab[a1[k]])
                sign_labels2.append(atlas_lab[a2[k]])
            
            string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
            list_strings.append(string)
            print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
            string = f'The connectivity strength differs significantly between meditation and control groups post'
            list_strings.append(string)
            print(f'The connectivity strength differs significantly between meditation and control groups post')
            for k in range(int(len(sign_labels1)/2)):
                string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                list_strings.append(string)
                print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
        else:
            string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
            list_strings.append(string)
            print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
        
with open(f'{out_dir}static/ttests_post_meditation-control_matrices_wei.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item)
        
#%%pre/post t-test for each group non wei
list_strings = []

for group in range(len(subs_labels_grouped)):
    idx_subs = idx_subs_grouped[group]
    n_subs = n_subs_grouped[group]
    string = f'results for {subs_labels_grouped[group]} group'
    list_strings.append(string)
    print(f'results for {subs_labels_grouped[group]} group')
    
    for i, (j, atlas) in enumerate(nback):
        for con in range(len(cond)):    
            corr_matr_s0 = correl_matrices_z[i]
            corr_matr_s0 = corr_matr_s0[idx_subs,0,:,:]
                    
            corr_matr_s1 = correl_matrices_z[i]
            corr_matr_s1 = corr_matr_s1[idx_subs,1,:,:]
            
            #take only lower triangle
            corr_vec_s0 = []
            corr_vec_s1 = []
            for k in range(n_subs):
                matrix = corr_matr_s0[k,:,:]
                corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
                corr_vec_s0.append(corr_vec)
                matrix = corr_matr_s1[k,:,:]
                corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
                corr_vec_s1.append(corr_vec)
            corr_vec_s0 = np.array(corr_vec_s0)
            corr_vec_s1 = np.array(corr_vec_s1) 
            
            t_val, p_val = ttest_rel(corr_vec_s0, corr_vec_s1, 0)
    
            n_samples, n_tests = corr_vec_s0.shape
            threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)
            for method in multitest_methods_names:
                output = multitest.multipletests(p_val, alpha=alpha, method=method)
                reject = output[0]
                pval_corrected = output[1]
                #threshold_bonferroni = stats.t.ppf(1.0 - alpha / n_tests, n_samples - 1)
                if True in reject:
                    string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                    list_strings.append(string)
                    print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
                    correlation_matrices_z = correl_matrices_z[i]
                    diagonal = np.ones(len(correlation_matrices_z[0,0,:,0]))
                    p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                    
                    #get locations of the significant results to get labels
                    a1,a2 = np.where(p_vals_matrix <= 0.05)
                    sign_labels1 = []
                    sign_labels2 = []
                    atlas_lab = atlas_labels[i,0]
                    for k in range(len(a1)):
                        sign_labels1.append(atlas_lab[a1[k]])
                        sign_labels2.append(atlas_lab[a2[k]])
                    
                    string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                    list_strings.append(string)
                    print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
                    string = f'The connectivity strength differs significantly between pre and post for {subs_labels_grouped[group]} group'
                    list_strings.append(string)
                    print(f'The connectivity strength differs significantly between pre and post for {subs_labels_grouped[group]} group')
                    for k in range(int(len(sign_labels1)/2)):
                        string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                        list_strings.append(string)
                        print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
                else:
                    string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                    list_strings.append(string)
                    print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
        
with open(f'{out_dir}static/all_ttests_pre-post_matrices_non-wei.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item)
        
#%%post sport/control
for i, (j, atlas) in enumerate(nback):
    for con in range(len(cond)):    
        corr_matr_s0 = correl_matrices_z[i]
        corr_matr_s0 = corr_matr_s0[idx_subs_sport,0,:,:]
                
        corr_matr_s1 = correl_matrices_z[i]
        corr_matr_s1 = corr_matr_s1[idx_subs_control,1,:,:]
        
        #take only lower triangle
        corr_vec_s0 = []
        corr_vec_s1 = []
        for k in range(n_subs_sport):
            matrix = corr_matr_s0[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s0.append(corr_vec)
        for k in range(n_subs_control):
            matrix = corr_matr_s1[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s1.append(corr_vec)
        corr_vec_s0 = np.array(corr_vec_s0)
        corr_vec_s1 = np.array(corr_vec_s1) 
        
        t_val, p_val = ttest_ind(corr_vec_s0, corr_vec_s1, 0)
         
        n_samples, n_tests = corr_vec_s0.shape
        threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)
        
        for method in multitest_methods_names:
            output = multitest.multipletests(p_val, alpha=alpha, method=method)
            reject = output[0]
            pval_corrected = output[1]

            if True in reject:
                string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                list_strings.append(string)
                print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
                correlation_matrices_z = correl_matrices_z[i]
                diagonal = np.ones(len(correlation_matrices_z[0,0,:,0]))
                p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                
                #get locations of the significant results to get labels
                a1,a2 = np.where(p_vals_matrix <= 0.05)
                sign_labels1 = []
                sign_labels2 = []
                atlas_lab = atlas_labels[i,0]
                for k in range(len(a1)):
                    sign_labels1.append(atlas_lab[a1[k]])
                    sign_labels2.append(atlas_lab[a2[k]])
                
                string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                list_strings.append(string)
                print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
                string = f'The connectivity strength differs significantly between sport and control groups post'
                list_strings.append(string)
                print(f'The connectivity strength differs significantly between sport and control groups post')
                for k in range(int(len(sign_labels1)/2)):
                    string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                    list_strings.append(string)
                    print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
            else:
                string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                list_strings.append(string)
                print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
        
with open(f'{out_dir}static/ttests_post_sport-control_matrices_non-wei.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item)

#%%post sport/med
for i, (j, atlas) in enumerate(nback):
    for con in range(len(cond)):    
        corr_matr_s0 = correl_matrices_z[i]
        corr_matr_s0 = corr_matr_s0[idx_subs_sport,0,:,:]
                
        corr_matr_s1 = correl_matrices_z[i]
        corr_matr_s1 = corr_matr_s1[idx_subs_med,1,:,:]
        
        #take only lower triangle
        corr_vec_s0 = []
        corr_vec_s1 = []
        for k in range(n_subs_sport):
            matrix = corr_matr_s0[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s0.append(corr_vec)
        for k in range(n_subs_med):
            matrix = corr_matr_s1[k,:,:]
            corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
            corr_vec_s1.append(corr_vec)
        corr_vec_s0 = np.array(corr_vec_s0)
        corr_vec_s1 = np.array(corr_vec_s1) 
        
        t_val, p_val = ttest_ind(corr_vec_s0, corr_vec_s1, 0)
         
        n_samples, n_tests = corr_vec_s0.shape
        threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)
        
        for method in multitest_methods_names:
            output = multitest.multipletests(p_val, alpha=alpha, method=method)
            reject = output[0]
            pval_corrected = output[1]

            if True in reject:
                string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
                list_strings.append(string)
                print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
                correlation_matrices_z = correl_matrices_z[i]
                diagonal = np.ones(len(correlation_matrices_z[0,0,:,0]))
                p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
                
                #get locations of the significant results to get labels
                a1,a2 = np.where(p_vals_matrix <= 0.05)
                sign_labels1 = []
                sign_labels2 = []
                atlas_lab = atlas_labels[i,0]
                for k in range(len(a1)):
                    sign_labels1.append(atlas_lab[a1[k]])
                    sign_labels2.append(atlas_lab[a2[k]])
                
                string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
                list_strings.append(string)
                print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
                string = f'The connectivity strength differs significantly between sport and meditation groups post'
                list_strings.append(string)
                print(f'The connectivity strength differs significantly between sport and meditation groups post')
                for k in range(int(len(sign_labels1)/2)):
                    string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                    list_strings.append(string)
                    print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
            else:
                string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
                list_strings.append(string)
                print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
        
with open(f'{out_dir}static/ttests_post_sport-meditation_matrices_non-wei.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item)
#%%post med/control
for i, (j, atlas) in enumerate(nback):
    for con in range(len(cond)):
     corr_matr_s0 = correl_matrices_z[i]
     corr_matr_s0 = corr_matr_s0[idx_subs_med,0,:,:]
             
     corr_matr_s1 = correl_matrices_z[i]
     corr_matr_s1 = corr_matr_s1[idx_subs_control,1,:,:]
     
     #take only lower triangle
     corr_vec_s0 = []
     corr_vec_s1 = []
     for k in range(n_subs_med):
         matrix = corr_matr_s0[k,:,:]
         corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
         corr_vec_s0.append(corr_vec)
     for k in range(n_subs_control):
         matrix = corr_matr_s1[k,:,:]
         corr_vec = sym_matrix_to_vec(matrix, discard_diagonal=True)
         corr_vec_s1.append(corr_vec)
     corr_vec_s0 = np.array(corr_vec_s0)
     corr_vec_s1 = np.array(corr_vec_s1) 
     
     t_val, p_val = ttest_ind(corr_vec_s0, corr_vec_s1, 0)
      
     n_samples, n_tests = corr_vec_s0.shape
     threshold_uncorrected = stats.t.ppf(1.0 - alpha, n_samples - 1)
     for method in multitest_methods_names:
        output = multitest.multipletests(p_val, alpha=alpha, method=method)
        reject = output[0]
        pval_corrected = output[1]

        if True in reject:
            string = f'{method}-corrected significant results {nback[i,1]} condition-{con}'
            list_strings.append(string)
            print(f'{method}-corrected significant results {nback[i,1]} condition-{con}')
            correlation_matrices_z = correl_matrices_z[i]
            diagonal = np.ones(len(correlation_matrices_z[0,0,:,0]))
            p_vals_matrix = nilearn.connectome.vec_to_sym_matrix(pval_corrected, diagonal)
            
            #get locations of the significant results to get labels
            a1,a2 = np.where(p_vals_matrix <= 0.05)
            sign_labels1 = []
            sign_labels2 = []
            atlas_lab = atlas_labels[i,0]
            for k in range(len(a1)):
                sign_labels1.append(atlas_lab[a1[k]])
                sign_labels2.append(atlas_lab[a2[k]])
            
            string = f'atlas: {atlas_labels[i,1]}, condition-{con}'
            list_strings.append(string)
            print(f'atlas: {atlas_labels[i,1]}, condition-{con}')
            string = f'The connectivity strength differs significantly between meditation and control groups post'
            list_strings.append(string)
            print(f'The connectivity strength differs significantly between meditation and control groups post')
            for k in range(int(len(sign_labels1)/2)):
                string = f'{k}. {sign_labels1[k]} and {sign_labels2[k]}'
                list_strings.append(string)
                print(f'{k}. {sign_labels1[k]} and {sign_labels2[k]}')
        else:
            string = f"no {method}-corrected significant results {nback[i,1]} condition-{con}"
            list_strings.append(string)
            print(f"no {method}-corrected significant results {nback[i,1]} condition-{con}")
        
with open(f'{out_dir}static/ttests_post_meditation-control_matrices_non-wei.txt', 'w') as f:
    for item in list_strings:
        f.write("%s\n" % item)