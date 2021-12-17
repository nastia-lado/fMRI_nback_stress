#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nback conditions: target or distractor
nback analysis:
1. hit - correct response to a target
2. miss = no response to a target
3. correct rejection - no response to a distractor
4. false alarm - response when no target is present
Stimuli\Response	Targe	Distractor
       Target	hit	          miss
     Distractor	false alarm	  correct rejection

Accuracy=(hits+correct_rejections)/all_trials
#Measures
#Accuracy
#Accuracy was calculated as the actual number of hits divided by the maximum number of hits.
#accuracy=#hitsNcongruent
#This imperfect definition is prone to respond to all stimuli behavior. In this case accuracy will be one.

#Penalized reaction time
#Penalized reaction time (PRT) tries to combine accuracy with the reaction time.
#Calculating PRT consists of three steps.
#First, all labeled reaction times are selected. Second, all reaction times for false alarms are
#set to 2000ms (maximal possible RT). Third, fake reaction times for misses are added to penalize for
#not responding. Then mean is calculated for task runs and individual blocks. 
#Note that mean PRT over blocks is usually different than the one calculated for the 
#enire task mean (some blocks consist of more responses than others). Formally, PRT is defined as
#prt=1#false alarms+#hits+#misses(∑r=1Nresponsesprtr+#misses×2000ms)
#where prtr is 2000ms for false alarms and RT for hits.

#D-prime
#D-prime based on signal detection theory takes into account both response sensitivity and specificity.
#First, hit rate hr and false alarm rate fr is calculated as:
#hr=#hits#hits+#misses
#fr=#false alarms#false alarms+#correct rejections
#Second, to avoid problems with infinite values, all hr or fr values equal to 1 are set to 0.99 and
#all hr or fr values equal to 0 are set to 0.01. Finally, d-prime is calculated as difference between
#pdf transormed values of hr and fr:
#d′=f(hr)−f(fr)
#where f(x) is the inverse of the cumulative distribution function for Gaussian distribution.
#Note that mean d-prime over task blocks is not equal to mean calculated for the entire task.

25-02-2021
"""
import pandas as pd
import numpy as np
import json
import os
import glob
import pydicom
from scipy.stats import norm
from scipy.stats import ttest_rel, ttest_ind, chisquare, chi2_contingency
from statsmodels.stats import multitest

#%%
def calculate_dprime(no_hit, no_crr, no_mis, no_fal):
    '''Calculates d-prime signal detection index

    Args: 
        no_hit (int): number of hits
        no_crr (int): number of correct rejections
        no_mis (int): number of misses
        no_fal (int): number of false alarms

    Returns:
        (float) D-prime index.    
    '''

    hit_rate = no_hit / (no_hit + no_mis)
    fal_rate = no_fal / (no_fal + no_crr)

    # Corner cases (infinity problem)
    if fal_rate == 0: fal_rate = 0.01
    if fal_rate == 1: fal_rate = 0.99

    if hit_rate == 0: hit_rate = 0.01
    if hit_rate == 1: hit_rate = 0.99

    return norm.ppf(hit_rate) - norm.ppf(fal_rate)
#%%
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/func_connect/04-correlation_matrices/'

groups = pd.read_csv(f'{top_dir}/behavioural/subj_info.csv')
subs = groups['sub']
nbackA = pd.read_csv(f'{top_dir}/behavioural/nbackA.csv')
nbackB = pd.read_csv(f'{top_dir}/behavioural/nbackB.csv')
nbackA['task_type'] = 'nbackA'
nbackB['task_type'] = 'nbackB'
nback = pd.concat([nbackA, nbackB], axis=0)

#nback.value_counts(['trial_type'])
#nback.value_counts(['trial_type','type_stim'])
trial_types = [1, 2, 3, 4]
max_trials = [100, 76, 160, 72]
max_targets = [50, 36, 72, 28] 
max_distr = [50, 40, 88, 44]
sess = ['ses-1', 'ses-3']

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

#%%demogr stat

#age
#sport/control
print('results for sport/control groups')
arr0 = groups.loc[idx_subs_sport,'age']
arr1 = groups.loc[idx_subs_control,'age']   
t_val, p_val = ttest_ind(arr0, arr1, 0)
np.mean(arr0)
print(t_val, p_val)
print(f'mean age sport {np.mean(arr0)}')
print(f'std age sport {np.std(arr0)}')

#sport/med
print('results for sport/med groups')
arr0 = groups.loc[idx_subs_sport,'age']
arr1 = groups.loc[idx_subs_med,'age']   
t_val, p_val = ttest_ind(arr0, arr1, 0)
print(t_val, p_val)
print(f'mean age med {np.mean(arr1)}')
print(f'std age med {np.std(arr1)}')

#med/control
print('results for med/control groups')
arr0 = groups.loc[idx_subs_med,'age']
arr1 = groups.loc[idx_subs_control,'age']   
t_val, p_val = ttest_ind(arr0, arr1, 0)
print(t_val, p_val)
print(f'mean age control {np.mean(arr1)}')
print(f'std age control {np.std(arr1)}')

#sample age
print('results for the entire sample')
arr1 = groups.loc[:,'age']
print(f'mean age sample {np.mean(arr1)}')
print(f'std age sample {np.std(arr1)}')

#gender
print('gender - results for all groups')
crosstab = pd.crosstab(groups['group'], groups['sex'])   
result = chi2_contingency(crosstab)
chi2 = result[0]
p = result[1]
print(chi2, p)

#%%
no_hit = np.zeros((groups.shape[0], int(len(sess)), int(len(trial_types)))) #hit
no_crr = np.zeros((groups.shape[0], int(len(sess)), int(len(trial_types)))) #miss
no_mis = np.zeros((groups.shape[0], int(len(sess)), int(len(trial_types)))) #correct rejection
no_fal = np.zeros((groups.shape[0], int(len(sess)), int(len(trial_types)))) #false alarm
hit_rate = np.zeros((groups.shape[0], int(len(sess)), int(len(trial_types))))
fal_rate = np.zeros((groups.shape[0], int(len(sess)), int(len(trial_types))))

for j in range(len(sess)):
    for i in range(len(subs)):
        run1_path = f'{top_dir}/Nifti/{subs[i]}/{sess[j]}/func/{subs[i]}_{sess[j]}_task-nback_run-1_events.tsv'
        run1 = pd.read_csv(run1_path, delimiter='\t')
        run1 = run1.drop(columns=['trial_type']) #otherwise trial_type columns will be twice in dataframe
        
        run2_path = f'{top_dir}/Nifti/{subs[i]}/{sess[j]}/func/{subs[i]}_{sess[j]}_task-nback_run-2_events.tsv'
        run2 = pd.read_csv(run2_path, delimiter='\t')
        run2 = run2.drop(columns=['trial_type']) #otherwise trial_type columns will be twice in dataframe
        
        df = pd.concat([run1, run2], axis=0)
        df = pd.concat([df, nback], axis=1)
        
        #do it for each trial type separately
        for ttype in trial_types:
            #no_crr = df_signal.loc[filt_rows, f'crr_{modality}'].sum()
            no_hit[i,j,ttype-1] = sum((df['trial_type'] == ttype) & (df['accuracy'] == 1) & (df['type_stim'] == 'target'))
            no_mis[i,j,ttype-1] = sum((df['trial_type'] == ttype) & (df['accuracy'] == 0) & (df['type_stim'] == 'target'))
            no_crr[i,j,ttype-1] = sum((df['trial_type'] == ttype) & (df['accuracy'] == 1) & (df['type_stim'] == 'distractor'))
            no_fal[i,j,ttype-1] = sum((df['trial_type'] == ttype) & (df['accuracy'] == 0) & (df['type_stim'] == 'distractor'))

            hit_rate[i,j,ttype-1] = no_hit[i,j,ttype-1] / (no_hit[i,j,ttype-1] + no_mis[i,j,ttype-1])
            fal_rate[i,j,ttype-1] = no_fal[i,j,ttype-1] / (no_fal[i,j,ttype-1] + no_crr[i,j,ttype-1])

#save all results to a df subj_info
new_columns = ['hits_1_1', 'hits_2_1', 'hits_3_1', 'hits_4_1'] 
df = pd.DataFrame(data=no_hit[:,0,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)
             
new_columns = ['hits_1_2', 'hits_2_2', 'hits_3_2', 'hits_4_2'] 
df = pd.DataFrame(data=no_hit[:,1,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['misses_1_1', 'misses_2_1', 'misses_3_1', 'misses_4_1'] 
df = pd.DataFrame(data=no_mis[:,0,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['misses_1_2', 'misses_2_2', 'misses_3_2', 'misses_4_2'] 
df = pd.DataFrame(data=no_mis[:,1,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)
 
new_columns = ['corr_rej_1_1', 'corr_rej_2_1', 'corr_rej_3_1', 'corr_rej_4_1'] 
df = pd.DataFrame(data=no_crr[:,0,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['corr_rej_1_2', 'corr_rej_2_2', 'corr_rej_3_2', 'corr_rej_4_2'] 
df = pd.DataFrame(data=no_crr[:,1,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['false_al_1_1', 'false_al_2_1', 'false_al_3_1', 'false_al_4_1'] 
df = pd.DataFrame(data=no_fal[:,0,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['false_al_1_2', 'false_al_2_2', 'false_al_3_2', 'false_al_4_2'] 
df = pd.DataFrame(data=no_fal[:,1,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['hit_rate_1_1', 'hit_rate_2_1', 'hit_rate_3_1', 'hit_rate_4_1'] 
df = pd.DataFrame(data=hit_rate[:,0,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['hit_rate_1_2', 'hit_rate_2_2', 'hit_rate_3_2', 'hit_rate_4_2'] 
df = pd.DataFrame(data=hit_rate[:,1,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['fal_rate_1_1', 'fal_rate_2_1', 'fal_rate_3_1', 'fal_rate_4_1'] 
df = pd.DataFrame(data=fal_rate[:,0,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['fal_rate_1_2', 'fal_rate_2_2', 'fal_rate_3_2', 'fal_rate_4_2'] 
df = pd.DataFrame(data=fal_rate[:,1,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

#%%
acc_target = np.zeros((groups.shape[0], int(len(sess)), int(len(trial_types))))
acc_distr = np.zeros((groups.shape[0], int(len(sess)), int(len(trial_types))))
acc_all = np.zeros((groups.shape[0], int(len(sess)), int(len(trial_types)))) 
RT_target = np.zeros((groups.shape[0], int(len(sess)), int(len(trial_types))))
RT_distr = np.zeros((groups.shape[0], int(len(sess)), int(len(trial_types))))
RT_all = np.zeros((groups.shape[0], int(len(sess)), int(len(trial_types)))) 

for j in range(len(sess)):
    for i in range(len(subs)):
        run1_path = f'/home/alado/datasets/RBH/Nifti/{subs[i]}/{sess[j]}/func/{subs[i]}_{sess[j]}_task-nback_run-1_events.tsv'
        run1 = pd.read_csv(run1_path, delimiter='\t')
        run1 = run1.drop(columns=['trial_type']) #otherwise trial_type columns will be twice in dataframe
        
        run2_path = f'{top_dir}/Nifti/{subs[i]}/{sess[j]}/func/{subs[i]}_{sess[j]}_task-nback_run-2_events.tsv'
        run2 = pd.read_csv(run2_path, delimiter='\t')
        run2 = run2.drop(columns=['trial_type']) #otherwise trial_type columns will be twice in dataframe
        
        df = pd.concat([run1, run2], axis=0)
        df = pd.concat([df, nback], axis=1)
        
        for ttype in trial_types:
            #RT to hits
            rt = df.loc[(df['trial_type'] == ttype) & (df['accuracy'] == 1) & (df['type_stim'] == 'target')]
            mean = rt['RT'].mean()
            RT_target[i,j,ttype-1] = mean
            
            #RT to corect rejections (correct responses to distractors)
            rt = df.loc[(df['trial_type'] == ttype) & (df['accuracy'] == 1) & (df['type_stim'] == 'distractor')]
            mean = rt['RT'].mean()
            RT_distr[i,j,ttype-1] = mean
            
            #RT to hits and corect rejections (correct responses to distractors)
            rt = df.loc[(df['trial_type'] == ttype) & (df['accuracy'] == 1)]
            mean = rt['RT'].mean()
            RT_all[i,j,ttype-1] = mean
            
            #acc to targets
            ac = df.loc[(df['trial_type'] == ttype) & (df['type_stim'] == 'target')]
            acc_target[i,j,ttype-1] = ac[ac.accuracy == 1].shape[0] / max_targets[ttype-1] * 100
            
            #acc to distractors
            ac = df.loc[(df['trial_type'] == ttype) & (df['type_stim'] == 'distractor')]
            acc_distr[i,j,ttype-1] = ac[ac.accuracy == 1].shape[0] / max_distr[ttype-1] * 100
            
            #acc_all
            ac = df.loc[(df['trial_type'] == ttype)]
            acc_all[i,j,ttype-1] = ac[ac.accuracy == 1].shape[0] / max_trials[ttype-1] * 100

#save all results to a df subj_info
new_columns = ['acc_target_1_1', 'acc_target_2_1', 'acc_target_3_1', 'acc_target_4_1'] 
df = pd.DataFrame(data=acc_target[:,0,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['acc_target_1_2', 'acc_target_2_2', 'acc_target_3_2', 'acc_target_4_2'] 
df = pd.DataFrame(data=acc_target[:,1,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['acc_distr_1_1', 'acc_distr_2_1', 'acc_distr_3_1', 'acc_distr_4_1'] 
df = pd.DataFrame(data=acc_distr[:,0,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['acc_distr_1_2', 'acc_distr_2_2', 'acc_distr_3_2', 'acc_distr_4_2'] 
df = pd.DataFrame(data=acc_distr[:,1,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['acc_all_1_1', 'acc_all_2_1', 'acc_all_3_1', 'acc_all_4_1'] 
df = pd.DataFrame(data=acc_all[:,0,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['acc_all_1_2', 'acc_all_2_2', 'acc_all_3_2', 'acc_all_4_2'] 
df = pd.DataFrame(data=acc_all[:,1,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['RT_target_1_1', 'RT_target_2_1', 'RT_target_3_1', 'RT_target_4_1'] 
df = pd.DataFrame(data=RT_target[:,0,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['RT_target_1_2', 'RT_target_2_2', 'RT_target_3_2', 'RT_target_4_2'] 
df = pd.DataFrame(data=RT_target[:,1,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['RT_distr_1_1', 'RT_distr_2_1', 'RT_distr_3_1', 'RT_distr_4_1'] 
df = pd.DataFrame(data=RT_distr[:,0,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['RT_distr_1_2', 'RT_distr_2_2', 'RT_distr_3_2', 'RT_distr_4_2'] 
df = pd.DataFrame(data=RT_distr[:,1,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['RT_all_1_1', 'RT_all_2_1', 'RT_all_3_1', 'RT_all_4_1'] 
df = pd.DataFrame(data=RT_all[:,0,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['RT_all_1_2', 'RT_all_2_2', 'RT_all_3_2', 'RT_all_4_2'] 
df = pd.DataFrame(data=RT_all[:,1,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)
         
#%%compute PRT and D-prime
ttr = 2033
pRT = np.zeros((groups.shape[0], int(len(sess)), int(len(trial_types))))
dPrime = np.zeros((groups.shape[0], int(len(sess)), int(len(trial_types))))
for j in range(len(sess)):
    for i in range(len(subs)):
        run1_path = f'{top_dir}/Nifti/{subs[i]}/{sess[j]}/func/{subs[i]}_{sess[j]}_task-nback_run-1_events.tsv'
        run1 = pd.read_csv(run1_path, delimiter='\t')
        run1 = run1.drop(columns=['trial_type']) #otherwise trial_type columns will be twice in dataframe
        
    
        run2_path = f'{top_dir}/Nifti/{subs[i]}/{sess[j]}/func/{subs[i]}_{sess[j]}_task-nback_run-2_events.tsv'
        run2 = pd.read_csv(run2_path, delimiter='\t')
        run2 = run2.drop(columns=['trial_type']) #otherwise trial_type columns will be twice in dataframe
        
        df = pd.concat([run1, run2], axis=0)
        df = pd.concat([df, nback], axis=1)
        
        for ttype in trial_types:
            hit = no_hit[i,j,ttype-1]
            mis = no_mis[i,j,ttype-1]
            crr = no_crr[i,j,ttype-1]
            fal = no_fal[i,j,ttype-1]
            #penalized reaction time
            rt = df.loc[(df['trial_type'] == ttype) & (df['accuracy'] == 1)]
            sum_rt = rt['RT'].sum()
            pRT[i,j,ttype-1] = (sum_rt + ttr * (fal + mis)) / max_trials[ttype-1]
              
              
            dPrime[i,j,ttype-1] = calculate_dprime(hit, crr, mis, fal)

new_columns = ['pRT_1_1', 'pRT_2_1', 'pRT_3_1', 'pRT_4_1'] 
df = pd.DataFrame(data=pRT[:,0,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['pRT_1_2', 'pRT_2_2', 'pRT_3_2', 'pRT_4_2'] 
df = pd.DataFrame(data=pRT[:,1,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['dPrime_1_1', 'dPrime_2_1', 'dPrime_3_1', 'dPrime_4_1'] 
df = pd.DataFrame(data=dPrime[:,0,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)

new_columns = ['dPrime_1_2', 'dPrime_2_2', 'dPrime_3_2', 'dPrime_4_2'] 
df = pd.DataFrame(data=dPrime[:,1,:], columns=new_columns) 
groups = pd.concat([groups, df], axis=1)


#%%save to df
#groups = groups.fillna(0)
groups.to_csv(f'{top_dir}/behavioural/task_performance.csv')

#%%very weird block, probably useless
columns = ['acc_all_1_1', 'acc_all_2_1', 'acc_all_3_1', 'acc_all_4_1',
            'acc_all_1_2', 'acc_all_2_2', 'acc_all_3_2', 'acc_all_4_2',
           'RT_all_1_1', 'RT_all_2_1', 'RT_all_3_1', 'RT_all_4_1', 
           'RT_all_1_2', 'RT_all_2_2', 'RT_all_3_2', 'RT_all_4_2',
            'pRT_1_1', 'pRT_2_1', 'pRT_3_1', 'pRT_4_1',
            'pRT_1_2', 'pRT_2_2', 'pRT_3_2', 'pRT_4_2',
            'dPrime_1_1', 'dPrime_2_1', 'dPrime_3_1', 'dPrime_4_1',
            'dPrime_1_2', 'dPrime_2_2', 'dPrime_3_2', 'dPrime_4_2'] 
for group in range(len(subs_labels_grouped)):
    idx_subs = idx_subs_grouped[group]
    n_subs = n_subs_grouped[group]
    #string = f'results for {subs_labels_grouped[group]} group'
    #list_strings.append(string)
    print(f'results for {subs_labels_grouped[group]} group')
    #arr = groups[idx_subs,0,:]
    for i in columns:
        print(f'{i}')
        mean = groups[i].mean()
        print(f'mean: {mean}')
        std = groups[i].std()
        print(f'std: {std}')
#%%statistics accuracy
print('statistics accuracy')
multitest_methods_names = ['bonferroni','holm','holm-sidak','simes-hochberg','fdr_bh',
                           'fdr_by','fdr_tsbh','fdr_tsbky','fdr_gbs']
np.set_printoptions(formatter={'float_kind':'{:f}'.format})
#pre/post t-test for each group
for group in range(len(subs_labels_grouped)):
    idx_subs = idx_subs_grouped[group]
    n_subs = n_subs_grouped[group]
    #string = f'results for {subs_labels_grouped[group]} group'
    #list_strings.append(string)
    print(f'results for {subs_labels_grouped[group]} group')

    arr0 = acc_all[idx_subs,0,:]
    arr1 = acc_all[idx_subs,1,:]
    
    print(f'mean pre {np.mean(arr0, axis=0)}')
    print(f'mean post {np.mean(arr1, axis=0)}')
    print(f'pre {np.std(arr0, axis=0)}')
    print(f'post {np.std(arr1, axis=0)}')  
    
    t_val, p_val = ttest_rel(arr0, arr1, 0)
    print(t_val, p_val)
    #for method in multitest_methods_names:
    #print(method)
    output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
    reject = output[0]
    pval_corrected = output[1]
    print(reject, pval_corrected)

#post sport/control
print('results for sport/control groups post')
arr0 = acc_all[idx_subs_sport,0,:]
arr1 = acc_all[idx_subs_control,1,:]   
print(f'mean sport {np.mean(arr0, axis=0)}')
print(f'mean control {np.mean(arr1, axis=0)}')  
print(f'sport {np.std(arr0, axis=0)}')
print(f'control {np.std(arr1, axis=0)}') 
t_val, p_val = ttest_ind(arr0, arr1, 0)
print(t_val, p_val)
#for method in multitest_methods_names:
#print(method)
reject = output[0]
pval_corrected = output[1]
print(reject, pval_corrected)
    
#post sport/med
print('results for sport/med groups post')
arr0 = acc_all[idx_subs_sport,0,:]
arr1 = acc_all[idx_subs_med,1,:]   
t_val, p_val = ttest_ind(arr0, arr1, 0)
print(t_val, p_val)
print(f'mean sport {np.mean(arr0, axis=0)}')
print(f'mean med {np.mean(arr1, axis=0)}')
print(f'sport {np.std(arr0, axis=0)}')
print(f'med {np.std(arr1, axis=0)}') 
#for method in multitest_methods_names:
#print(method)
output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
reject = output[0]
pval_corrected = output[1]
print(reject, pval_corrected)

#post med/control
print('results for med/control groups post')
arr0 = acc_all[idx_subs_med,0,:]
arr1 = acc_all[idx_subs_control,1,:]   
t_val, p_val = ttest_ind(arr0, arr1, 0)
print(t_val, p_val)
print(f'mean med {np.mean(arr0, axis=0)}')
print(f'mean control {np.mean(arr1, axis=0)}')
print(f'med {np.std(arr0, axis=0)}')
print(f'control {np.std(arr1, axis=0)}') 
#for method in multitest_methods_names:
#print(method)
output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
reject = output[0]
pval_corrected = output[1]
print(reject, pval_corrected)

#%%statistics RT
#pre/post t-test for each group
print('statistics RT')
for group in range(len(subs_labels_grouped)):
    idx_subs = idx_subs_grouped[group]
    n_subs = n_subs_grouped[group]
    #string = f'results for {subs_labels_grouped[group]} group'
    #list_strings.append(string)
    print(f'results for {subs_labels_grouped[group]} group')

    arr0 = RT_all[idx_subs,0,:]
    arr1 = RT_all[idx_subs,1,:]
    
    print(f'mean pre {np.mean(arr0, axis=0)}')
    print(f'mean post {np.mean(arr1, axis=0)}')
    print(f'pre {np.std(arr0, axis=0)}')
    print(f'post {np.std(arr1, axis=0)}') 
        
    t_val, p_val = ttest_rel(arr0, arr1, 0)
    print(t_val, p_val)
    #for method in multitest_methods_names:
    #print(method)
    output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
    reject = output[0]
    pval_corrected = output[1]
    print(reject, pval_corrected)

#post sport/control
print(f'results for sport/control groups post')
arr0 = RT_all[idx_subs_sport,0,:]
arr1 = RT_all[idx_subs_control,1,:]   
t_val, p_val = ttest_ind(arr0, arr1, 0)
print(t_val, p_val)
print(f'mean sport {np.mean(arr0, axis=0)}')
print(f'mean control {np.mean(arr1, axis=0)}')  
print(f'sport {np.std(arr0, axis=0)}')
print(f'control {np.std(arr1, axis=0)}') 
#for method in multitest_methods_names:
#print(method)
output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
reject = output[0]
pval_corrected = output[1]
print(reject, pval_corrected)
    
#post sport/med
print(f'results for sport/med groups post')
arr0 = RT_all[idx_subs_sport,0,:]
arr1 = RT_all[idx_subs_med,1,:]   
t_val, p_val = ttest_ind(arr0, arr1, 0)
print(t_val, p_val)
print(f'mean sport {np.mean(arr0, axis=0)}')
print(f'mean med {np.mean(arr1, axis=0)}')  
print(f'sport {np.std(arr0, axis=0)}')
print(f'med {np.std(arr1, axis=0)}') 
#for method in multitest_methods_names:
#print(method)
output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
reject = output[0]
pval_corrected = output[1]
print(reject, pval_corrected)

#post med/control
print(f'results for med/control groups post')
arr0 = RT_all[idx_subs_med,0,:]
arr1 = RT_all[idx_subs_control,1,:]   
t_val, p_val = ttest_ind(arr0, arr1, 0)
print(t_val, p_val)
print(f'mean med {np.mean(arr0, axis=0)}')
print(f'mean control {np.mean(arr1, axis=0)}')  
print(f'med {np.std(arr0, axis=0)}')
print(f'control {np.std(arr1, axis=0)}') 
#for method in multitest_methods_names:
#print(method)
output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
reject = output[0]
pval_corrected = output[1]
print(reject, pval_corrected)

#%%statistics D-prime
print('statistics D-prime')
#pre/post t-test for each group
for group in range(len(subs_labels_grouped)):
    idx_subs = idx_subs_grouped[group]
    n_subs = n_subs_grouped[group]
    #string = f'results for {subs_labels_grouped[group]} group'
    #list_strings.append(string)
    print(f'results for {subs_labels_grouped[group]} group')

    arr0 = dPrime[idx_subs,0,:]
    arr1 = dPrime[idx_subs,1,:]
    
    print(f'mean pre {np.mean(arr0, axis=0)}')
    print(f'mean post {np.mean(arr1, axis=0)}')
    print(f'pre {np.std(arr0, axis=0)}')
    print(f'post {np.std(arr1, axis=0)}') 
    
    t_val, p_val = ttest_rel(arr0, arr1, 0)
    print(t_val, p_val)
    #for method in multitest_methods_names:
    #print(method)
    output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
    reject = output[0]
    pval_corrected = output[1]
    print(reject, pval_corrected)

#post sport/control
print(f'results for sport/control groups post')
arr0 = dPrime[idx_subs_sport,0,:]
arr1 = dPrime[idx_subs_control,1,:]   
t_val, p_val = ttest_ind(arr0, arr1, 0)
print(t_val, p_val)
print(f'mean sport {np.mean(arr0, axis=0)}')
print(f'mean control {np.mean(arr1, axis=0)}')  
print(f'sport {np.std(arr0, axis=0)}')
print(f'control {np.std(arr1, axis=0)}') 
#for method in multitest_methods_names:
#print(method)
output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
reject = output[0]
pval_corrected = output[1]
print(reject, pval_corrected)
    
#post sport/med
print(f'results for sport/med groups post')
arr0 = dPrime[idx_subs_sport,0,:]
arr1 = dPrime[idx_subs_med,1,:]   
t_val, p_val = ttest_ind(arr0, arr1, 0)
print(t_val, p_val)
print(f'mean sport {np.mean(arr0, axis=0)}')
print(f'mean med {np.mean(arr1, axis=0)}')  
print(f'sport {np.std(arr0, axis=0)}')
print(f'med {np.std(arr1, axis=0)}') 
#for method in multitest_methods_names:
#print(method)
output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
reject = output[0]
pval_corrected = output[1]
print(reject, pval_corrected)

#post med/control
print(f'results for med/control groups post')
arr0 = dPrime[idx_subs_med,0,:]
arr1 = dPrime[idx_subs_control,1,:]   
t_val, p_val = ttest_ind(arr0, arr1, 0)
print(t_val, p_val)
print(f'mean med {np.mean(arr0, axis=0)}')
print(f'mean control {np.mean(arr1, axis=0)}')  
print(f'med {np.std(arr0, axis=0)}')
print(f'control {np.std(arr1, axis=0)}') 
#for method in multitest_methods_names:
#print(method)
output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
reject = output[0]
pval_corrected = output[1]
print(reject, pval_corrected)

#%%statistics pRT
print('statistics pRT')
#pre/post t-test for each group
for group in range(len(subs_labels_grouped)):
    idx_subs = idx_subs_grouped[group]
    n_subs = n_subs_grouped[group]
    #string = f'results for {subs_labels_grouped[group]} group'
    #list_strings.append(string)
    print(f'results for {subs_labels_grouped[group]} group')

    arr0 = pRT[idx_subs,0,:]
    arr1 = pRT[idx_subs,1,:]
        
    print(f'mean pre {np.mean(arr0, axis=0)}')
    print(f'mean post {np.mean(arr1, axis=0)}')
    print(f'pre {np.std(arr0, axis=0)}')
    print(f'post {np.std(arr1, axis=0)}') 
    
    t_val, p_val = ttest_rel(arr0, arr1, 0)
    print(t_val, p_val)
    #for method in multitest_methods_names:
    #print(method)
    output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
    reject = output[0]
    pval_corrected = output[1]
    print(reject, pval_corrected)

#post sport/control
print(f'results for sport/control groups post')
arr0 = pRT[idx_subs_sport,0,:]
arr1 = pRT[idx_subs_control,1,:]   
t_val, p_val = ttest_ind(arr0, arr1, 0)
print(t_val, p_val)
print(f'mean sport {np.mean(arr0, axis=0)}')
print(f'mean control {np.mean(arr1, axis=0)}')  
print(f'sport {np.std(arr0, axis=0)}')
print(f'control {np.std(arr1, axis=0)}') 
#for method in multitest_methods_names:
#print(method)
output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
reject = output[0]
pval_corrected = output[1]
print(reject, pval_corrected)
    
#post sport/med
print(f'results for sport/med groups post')
arr0 = pRT[idx_subs_sport,0,:]
arr1 = pRT[idx_subs_med,1,:]   
t_val, p_val = ttest_ind(arr0, arr1, 0)
print(t_val, p_val)
print(f'mean sport {np.mean(arr0, axis=0)}')
print(f'mean med {np.mean(arr1, axis=0)}')  
print(f'sport {np.std(arr0, axis=0)}')
print(f'med {np.std(arr1, axis=0)}') 
#for method in multitest_methods_names:
#print(method)
output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
reject = output[0]
pval_corrected = output[1]
print(reject, pval_corrected)

#post med/control
print(f'results for med/control groups post')
arr0 = pRT[idx_subs_med,0,:]
arr1 = pRT[idx_subs_control,1,:]   
t_val, p_val = ttest_ind(arr0, arr1, 0)
print(t_val, p_val)
print(f'mean med {np.mean(arr0, axis=0)}')
print(f'mean control {np.mean(arr1, axis=0)}')  
print(f'med {np.std(arr0, axis=0)}')
print(f'control {np.std(arr1, axis=0)}') 
#for method in multitest_methods_names:
#print(method)
output = multitest.multipletests(p_val, alpha=alpha, method='bonferroni')
reject = output[0]
pval_corrected = output[1]
print(reject, pval_corrected)


#%% subjects with worst performance

#accuracy
#high accuracy - good, low accuracy - bad
check_columns = ['acc_all_1_1', 'acc_all_2_1', 'acc_all_3_1', 'acc_all_4_1']
for i in check_columns: 
    print(i)
    a = groups.nsmallest(10, f'{i}', keep='all')
    a = a['sub']
    print(np.array(a))

check_columns = ['RT_all_1_1', 'RT_all_2_1', 'RT_all_3_1', 'RT_all_4_1'] 
for i in check_columns: 
    print(i)
    a = groups.nsmallest(10, f'{i}', keep='all')
    a = a['sub']
    print(np.array(a))

check_columns = ['pRT_1_1', 'RT_distr_2_1', 'pRT_3_1', 'pRT_4_1'] 
for i in check_columns: 
    print(i)
    a = groups.nsmallest(10, f'{i}', keep='all')
    a = a['sub']
    print(np.array(a))

check_columns = ['dPrime_1_1', 'dPrime_2_1', 'dPrime_3_1', 'dPrime_4_1'] 
for i in check_columns: 
    print(i)
    a = groups.nsmallest(10, f'{i}', keep='all')
    a = a['sub']
    print(np.array(a))












