"""
Seed-based correlation nltools

Created on Mon May  3 22:30:55 2021

@author: alado
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltools.data import Brain_Data, Design_Matrix, Adjacency
from nltools.mask import expand_mask, roi_to_brain
from nltools.stats import zscore, fdr, one_sample_permutation
from nltools.file_reader import onsets_to_dm
from nltools.plotting import component_viewer
from scipy.stats import binom, ttest_1samp
from sklearn.metrics import pairwise_distances
from copy import deepcopy
import networkx as nx
from nilearn.plotting import plot_stat_map, view_img_on_surf
from bids import BIDSLayout, BIDSValidator
import nibabel as nib

#%%
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/func_connect/04-correlation_matrices/'
data_dir = f'{top_dir}/Nifti'
#Layout class representing an entire BIDS dataset
layout = BIDSLayout(data_dir, derivatives=True)
subs = layout.get_subjects()

sub = 'sub-02'
fwhm=5
t_r = 3
#%%
data = Brain_Data(layout.get(subject=subs[0], task='nback', scope='derivatives', suffix='bold', extension='nii.gz', return_type='file')[0])
smoothed = data.smooth(fwhm=fwhm)

#%%mask
#http://cosanlab.com/static/papers/delaVega_2016_JNeuro.pdf
#mask = Brain_Data('../masks/k50_2mm.nii.gz')

#UPSAMPLE MASK???
mask = Brain_Data('https://neurovault.org/media/images/8423/k50_2mm.nii.gz')

mask.plot()

mask_x = expand_mask(mask)

f = mask_x[0:5].plot()
#%%
vmpfc = smoothed.extract_roi(mask=mask_x[32])

plt.figure(figsize=(15,5))
plt.plot(vmpfc, linewidth=3)
plt.ylabel('Mean Intensitiy', fontsize=18)
plt.xlabel('Time (TRs)', fontsize=18)

#%%
tr = layout.get_tr()
fwhm = 5
n_tr = len(data)

def make_motion_covariates(mc, tr):
    z_mc = zscore(mc)
    all_mc = pd.concat([z_mc, z_mc**2, z_mc.diff(), z_mc.diff()**2], axis=1)
    all_mc.fillna(value=0, inplace=True)
    return Design_Matrix(all_mc, sampling_freq=1/tr)


vmpfc = zscore(pd.DataFrame(vmpfc, columns=['vmpfc']))

csf_mask = Brain_Data(f'{data_dir}/derivatives/fmriprep/sub-{subs[0]}/anat/sub-{subs[0]}_space-MNI152NLin2009cAsym_label-CSF_probseg.nii.gz')
csf_mask = csf_mask.threshold(upper=0.7, binarize=True)
csf_mask.plot()
csf = zscore(pd.DataFrame(smoothed.extract_roi(mask=csf_mask).T, columns=['csf']))

#%%
spikes = smoothed.find_spikes(global_spike_cutoff=3, diff_spike_cutoff=3)
covariates = pd.read_csv(layout.get(subject=subs[0], scope='derivatives', extension='.tsv')[0].path, sep='\t')
mc = covariates[['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z']]
mc_cov = make_motion_covariates(mc, tr)
dm = Design_Matrix(pd.concat([vmpfc, csf, mc_cov, spikes.drop(labels='TR', axis=1)], axis=1), sampling_freq=1/tr)
dm = dm.add_poly(order=2, include_lower=True)

smoothed.X = dm
stats = smoothed.regress()

vmpfc_conn = stats['beta'][0]

#%%
vmpfc_conn.threshold(upper=5, lower=-5).plot()























