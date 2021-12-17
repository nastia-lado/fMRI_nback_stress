# File structure for Lipsia
import shutil, os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nibabel as nib

data_dir = '/home/alado/datasets/RBH/preprocessed/derivatives/fmriprep'
top_dir = '/home/alado/datasets/RBH'
sess = ['ses-1', 'ses-3']
groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs_sport = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport'])])
subs_med = pd.Series.tolist(groups['sub'][groups['group'].isin(['med'])])
subs_control = pd.Series.tolist(groups['sub'][groups['group'].isin(['control'])])

#%%
files = []
for i, sub in enumerate(subs_sport):
    file = (f'{top_dir}/Lipsia/{sub}/ses-1/func/{sub}_ses-1_task-nback_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.v')
    files.append(file)
for f in files:
    shutil.copy(f, f'{top_dir}/Lipsia/pre_sport')

files = []
for i, sub in enumerate(subs_sport):
    file = (f'{top_dir}/Lipsia/{sub}/ses-3/func/{sub}_ses-3_task-nback_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.v')
    files.append(file)
for f in files:
    shutil.copy(f, f'{top_dir}/Lipsia/post_sport')   

files = []
for i, sub in enumerate(subs_med):
    file = (f'{top_dir}/Lipsia/{sub}/ses-1/func/{sub}_ses-1_task-nback_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.v')
    files.append(file)
for f in files:
    shutil.copy(f, f'{top_dir}/Lipsia/pre_med')
    
files = []
for i, sub in enumerate(subs_med):
    file = (f'{top_dir}/Lipsia/{sub}/ses-3/func/{sub}_ses-3_task-nback_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.v')
    files.append(file)
for f in files:
    shutil.copy(f, f'{top_dir}/Lipsia/post_med')

files = []
for i, sub in enumerate(subs_control):
    file = (f'{top_dir}/Lipsia/{sub}/ses-1/func/{sub}_ses-1_task-nback_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.v')
    files.append(file)
for f in files:
    shutil.copy(f, f'{top_dir}/Lipsia/pre_control')
    
files = []
for i, sub in enumerate(subs_control):
    file = (f'{top_dir}/Lipsia/{sub}/ses-3/func/{sub}_ses-3_task-nback_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.v')
    files.append(file)
for f in files:
    shutil.copy(f, f'{top_dir}/Lipsia/post_control')
    
#%%
files = []
for i, sub in enumerate(subs_sport):
    file = (f'{top_dir}/Lipsia/{sub}/ses-1/func/{sub}_ses-1_task_nback_run-1_lisa-precoloring_4-3_result.v')
    files.append(file)
for f in files:
    shutil.copy(f, f'{top_dir}/Lipsia/pre_sport')

files = []
for i, sub in enumerate(subs_sport):
    file = (f'{top_dir}/Lipsia/{sub}/ses-3/func/{sub}_ses-3_task_nback_run-1_lisa-precoloring_4-3_result.v')
    files.append(file)
for f in files:
    shutil.copy(f, f'{top_dir}/Lipsia/post_sport')   

files = []
for i, sub in enumerate(subs_med):
    file = (f'{top_dir}/Lipsia/{sub}/ses-1/func/{sub}_ses-1_task_nback_run-1_lisa-precoloring_4-3_result.v')
    files.append(file)
for f in files:
    shutil.copy(f, f'{top_dir}/Lipsia/pre_med')
    
files = []
for i, sub in enumerate(subs_med):
    file = (f'{top_dir}/Lipsia/{sub}/ses-3/func/{sub}_ses-3_task_nback_run-1_lisa-precoloring_4-3_result.v')
    files.append(file)
for f in files:
    shutil.copy(f, f'{top_dir}/Lipsia/post_med')

files = []
for i, sub in enumerate(subs_control):
    file = (f'{top_dir}/Lipsia/{sub}/ses-1/func/{sub}_ses-1_task_nback_run-1_lisa-precoloring_4-3_result.v')
    files.append(file)
for f in files:
    shutil.copy(f, f'{top_dir}/Lipsia/pre_control')
    
files = []
for i, sub in enumerate(subs_control):
    file = (f'{top_dir}/Lipsia/{sub}/ses-3/func/{sub}_ses-3_task_nback_run-1_lisa-precoloring_4-3_result.v')
    files.append(file)
for f in files:
    shutil.copy(f, f'{top_dir}/Lipsia/post_control')
    
#%%
zmap = nib.load('/home/alado/datasets/RBH/Lipsia/connect_results/ECM/sub-02_ses-1_ecm_result.nii')
zmap = zmap.get_fdata()




























    