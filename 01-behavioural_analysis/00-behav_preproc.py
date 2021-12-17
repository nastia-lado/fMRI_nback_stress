#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert CSV with the information about all sessions to separate TSV files
some preprocessing, e.g. creating column names, is done manually
CORRECT FOR SCANNER ONSET
Original TSV might come from E-Data Aid

25-02-2021
"""
import pandas as pd
import numpy as np
import json
import os
import glob
import pydicom
import shutil

#%%
top_dir = '/home/alado/datasets/RBH'
#out_dir = '/home/alado/datasets/RBH/func_connect/04-correlation_matrices/'

groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = groups['sub']
sess = ['ses-1', 'ses-3']

trial_types = [1, 2, 3, 4]
#max_trials = [60, 38, 80, 36]
columns = ['sub', 'ses', 'Acc_0', 'Acc_1', 'Acc_2', 'Acc_4', 'RT_0', 'RT_1', 'RT_2', 'RT_4',
           'pRT_0', 'pRT_1', 'pRT_2', 'pRT_4',]
new_df = pd.DataFrame(columns = columns)

columns = ['sub', 'age', 'sex']
demogr_data = pd.DataFrame(columns = columns)

#%%Get demographics data from dicoms
#run only once
folder_names = []
for name in glob.glob('/home/alado/datasets/Raw_Data/*T2'):
    folder_names.append(name)

demogr_data['sub'] = folder_names

#get file names
folder_names2 = []
for i in range(len(folder_names)):
    for name in glob.glob(f'{folder_names[i]}/*'):
        folder_names2.append(name)

folder_names = []
for i in range(len(folder_names2)):
    for name in glob.glob(f'{folder_names2[i]}/*NBACKA*'):
        folder_names.append(name)

file_names = []
for i in range(len(folder_names)):
    fnames = os.listdir(f'{folder_names[i]}')
    name = fnames[0]
    name = f'{folder_names[i]}/{fnames[0]}'
    file_names.append(name)

age = []
sex = []

#get age and sex data from dicoms
for i in range(len(file_names)):
    ds = pydicom.read_file(file_names[i])
    sex.append(ds.PatientSex)
    age.append(ds.PatientAge)
    
#save lists to df and drop long pasrts of strings
demogr_data['age'] = age
demogr_data['sex'] = sex
demogr_data['sub'] = demogr_data['sub'].str[30:-18]
demogr_data['age'] = demogr_data['age'].str[1:-1]
demogr_data['age'] = pd.to_numeric(demogr_data['age'])
demogr_data = demogr_data.sort_values(by=['sub'])
demogr_data = demogr_data.reset_index(drop=True)

#Add demogr info to df groups
groups['sex'] = demogr_data['sex']
groups['age'] = demogr_data['age']

#save new group file
groups.to_csv(f'{top_dir}/behavioural/subj_info.csv', index=False)

#%%prepare files with info about stimuli presented
#run once
nbackA = pd.read_csv(f'{top_dir}/behavioural/nbackA.csv')
nbackA['type_stim'] = nbackA['type_stim'].replace(['c'],'target')
nbackA['type_stim'] = nbackA['type_stim'].replace(['b'],'distractor')
nbackA.to_csv(f'{top_dir}/behavioural/nbackA.csv', index=False)

nbackB = pd.read_csv(f'{top_dir}/behavioural/nbackB.csv')
nbackB['type_stim'] = nbackB['type_stim'].replace(['c'],'target')
nbackB['type_stim'] = nbackB['type_stim'].replace(['b'],'distractor')
nbackB.to_csv(f'{top_dir}/behavioural/nbackB.csv', index=False)

#%%create separate events files for each subject
all_data = pd.read_csv(f'{top_dir}/behavioural/nbackB_all_subj.tsv', delimiter='\t')
subjects = np.unique(all_data['subj'])
sessions = np.unique(all_data['session'])
for s in subjects:
    for ses in sessions:
        new_tsv = all_data.loc[(all_data['subj'] == s) & (all_data['session'] == ses)]
        #delete first column
        if s < 10:
            new_tsv.to_csv(f'{top_dir}/behavioural/events/sub-0{s}_ses-{ses}_task-nback_run-2_events.tsv', sep='\t', index=False)
        else:
            new_tsv.to_csv(f'{top_dir}/behavioural/events/sub-{s}_ses-{ses}_task-nback_run-2_events.tsv', sep='\t', index=False)

#%%copy files to correct folders
for j in range(len(sess)):
    for i in range(len(subs)):
        original = f'{top_dir}/behavioural/events/{subs[i]}_{sess[j]}_task-nback_run-2_events.tsv'
        target = f'{top_dir}/Nifti/{subs[i]}/{sess[j]}/func/{subs[i]}_{sess[j]}_task-nback_run-2_events.tsv'

        shutil.copyfile(original, target)

#%%
for j in range(len(sess)):
    for i in range(len(subs)):
        run2_path = f'{top_dir}/Nifti/{subs[i]}/{sess[j]}/func/{subs[i]}_{sess[j]}_task-nback_run-2_events.tsv'
        run2 = pd.read_csv(run2_path, delimiter='\t')
        if 'subj' in run2.columns:
            run2 = run2.drop(columns=['subj']) #otherwise trial_type columns will be twice in dataframe
        if 'session' in run2.columns:
            run2 = run2.drop(columns=['session']) #otherwise trial_type columns will be twice in dataframe
        #save
        run2.to_csv(f'{top_dir}/Nifti/{subs[i]}/{sess[j]}/func/{subs[i]}_{sess[j]}_task-nback_run-2_events.tsv', sep='\t',  index=False)
#%%

for j in range(len(sess)):
    for i in range(len(subs)):
        run2_path = f'{top_dir}/Nifti/{subs[i]}/{sess[j]}/func/{subs[i]}_{sess[j]}_task-nback_run-2_events.tsv'
        run2 = pd.read_csv(run2_path, delimiter='\t')























