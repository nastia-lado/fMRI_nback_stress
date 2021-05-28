#Processing behavioral logs
#Last edited: 20-10-2020


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from scipy.stats import norm

#%%
def convert_logfile(fname: str, task_meta: dict) -> pd.DataFrame:
    '''Converts logfile into concise and easy to work DataFrame
    
    Args:
        fname: path to log file
        task_meta: task metadata
        
    Returns:
        Table containing information about each trial. Each row contain block
        number, task condition, correct and actual response for both visual and
        auditory stimuli.    
    '''
    # Setup local variables
    n_stims = task_meta['n_stims']
    n_blocks = task_meta['n_blocks']
    n_conditions = task_meta['n_conditions']
    n_trials = n_stims * n_blocks * n_conditions
    ttr = task_meta['ttr']
    
    # Create block and condition indices
    ind_block = np.repeat(np.arange(n_blocks), n_stims * n_conditions)[:, np.newaxis]
    ind_cond = np.tile(
        np.concatenate((np.ones(n_stims), 2*np.ones(n_stims))), n_blocks
    )[:, np.newaxis]

    df_ind = pd.DataFrame(
        np.hstack((ind_block, ind_cond)), 
        columns=['block', 'condition']).astype('int')

    # Load stimulus data
    df = pd.read_csv(fname, delimiter='\t', skiprows=3)

    stim_filter = df['Code'].str.contains('yes') | df['Code'].str.contains('no')
    df_stim = df.loc[stim_filter, :]

    # Split visual and audio stimuli
    df_stim_vis = df_stim.loc[df_stim['Event Type'] == 'Picture',  
                              ['xs(str)', 'Time']].reset_index(drop=True)
    df_stim_aud = df_stim.loc[df_stim['Event Type'] == 'Sound',  
                              'xs(str)'].reset_index(drop=True)
    df_stim_vis.columns = ['ans_vis', 'stim_onset']
    df_stim_aud.name = 'ans_aud'

    # Merge visual and audio stimuli again & add response columns
    df_stim = pd.concat([df_ind, df_stim_vis, df_stim_aud], axis=1, sort=False)
    df_stim['resp_aud'], df_stim['resp_aud_time'], \
    df_stim['resp_vis'], df_stim['resp_vis_time'] = np.zeros((4, n_trials), dtype='int')
    
    # Load response data
    resp_filer = df['Code'].isin(['1', '2'])
    df_resp = df.loc[resp_filer, :]

    # Analyze trialwise responses
    for i, row in df_stim.iterrows():

        r = df_resp.loc[(df_resp['Time'] > row['stim_onset']) & \
                        (df_resp['Time'] < row['stim_onset'] + ttr), 
                        ['Code', 'Time']]

        if '2' in r['Code'].unique():
            df_stim.loc[i, 'resp_aud'] = 1
            df_stim.loc[i, 'resp_aud_time'] = r.loc[r['Code']=='2', 'Time'].values[0] 
        if '1' in r['Code'].unique():
            df_stim.loc[i, 'resp_vis'] = 1
            df_stim.loc[i, 'resp_vis_time'] = r.loc[r['Code']=='1', 'Time'].values[0] 

    for modality in ['vis', 'aud']:
        df_stim[f'resp_{modality}_time'] -= df_stim['stim_onset'] 
        df_stim.loc[df_stim[f'resp_{modality}_time'] < 0, 
                    f'resp_{modality}_time'] = np.nan
        df_stim[f'resp_{modality}_time'] /= 10
        df_stim[f'ans_{modality}'] = df_stim[f'ans_{modality}'].map({' yes': 1, ' no': 0})

    # Drop and reorder columns
    df_stim = df_stim[['block', 'condition', 
                       'ans_vis', 'resp_vis', 'resp_vis_time',
                       'ans_aud', 'resp_aud', 'resp_aud_time']]
    
    return df_stim


#%%Behavioral measures
#Introduction
#We assume that each measure is calculated separately for single subject, session,
#task condition (1-back or 2-back) and stimuli modality (visual or auditory).
#Each behavioral response can be divided into one of four categories: hit, miss,
#correct rejection or false alarm:

#here a table
#Stimuli\Response	Congruent	Incongruent
#       Congruent	hit	          miss
#       Incongruent	false alarm	  correct rejection

#Subjects were instructed to respond with their thumb to congruent stimuli and omit response to
#incongruent stimuli. Note that in this task setting "real misses", i.e. when subject failed to respond
#within required time window cannot be distinguished from correct rejections and misses. Number of
#responses of each type was calculated on the level of entire run and for individual task blocks.
#Reaction time (RT) was calculated for hits and false alarms.

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


def calculate_behavioral_measures(df_stim: pd.DataFrame, task_meta: dict) -> tuple:
    '''Calculates main three behavioral measures: accuracy,  pRT and D-prime.
    
    Args: 
        df_stim: trialwise task information
        task_meta: task_meta
        
    Returns:
        Aggregated behavioral measures for entire task. 
            (n_measures x n_conditions x n_modalities) 
        Aggregated behavioral measures for specific blocks.
            (n_measures x n_conditions x n_modalities x n_blocks)
    '''

    def _get_resp_patterns(df_signal, filt_rows, modality):
        '''Calculates number of hits, correct rejections, misses and false alarms.

        Args: 
            modality (str): either 'vis' or 'aud'

        Returns:
            Number of hits, correct rejections, misses and false alarms.

        '''
        no_hit = df_signal.loc[filt_rows, f'hit_{modality}'].sum()
        no_crr = df_signal.loc[filt_rows, f'crr_{modality}'].sum()
        no_mis = df_signal.loc[filt_rows, f'mis_{modality}'].sum()
        no_fal = df_signal.loc[filt_rows, f'fal_{modality}'].sum()

        return (no_hit, no_crr, no_mis, no_fal)

    def _calculate_dprime(no_hit, no_crr, no_mis, no_fal):
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

    df_signal = df_stim[['block', 'condition']].copy()
    n_modalities = task_meta['n_modalities']
    n_conditions = task_meta['n_conditions']
    n_blocks = task_meta['n_blocks']
    ttr = task_meta['ttr'] / 10

    # Signal theory measures
    for s in ['vis', 'aud']:
        df_signal[f'hit_{s}'] = (df_stim[f'ans_{s}'] == 1) & (df_stim[f'resp_{s}'] == 1)
        df_signal[f'crr_{s}'] = (df_stim[f'ans_{s}'] == 0) & (df_stim[f'resp_{s}'] == 0)
        df_signal[f'mis_{s}'] = (df_stim[f'ans_{s}'] == 1) & (df_stim[f'resp_{s}'] == 0)
        df_signal[f'fal_{s}'] = (df_stim[f'ans_{s}'] == 0) & (df_stim[f'resp_{s}'] == 1)
        df_signal[f'hit_time_{s}'] = df_signal[f'hit_{s}'].replace(False, np.nan) \
                                   * df_stim[f'resp_{s}_time']
        df_signal[f'fal_time_{s}'] = df_signal[f'fal_{s}'].replace(False, np.nan) \
                                   * df_stim[f'resp_{s}_time']

    acc_block, prt_block, dpr_block = np.zeros((3, n_conditions, n_modalities, n_blocks))
    acc, prt, dpr = np.zeros((3, n_conditions, n_modalities))
    hit_rate, fal_rate = np.zeros((2, n_conditions, n_modalities))

    for idx_c, condition in enumerate([1, 2]):
        for idx_m, modality in enumerate(['vis', 'aud']):

            filt_rows = (df_signal['condition'] == condition)

            # Count hits, correct rejections, misses and false alarms
            no_hit, no_crr, no_mis, no_fal = _get_resp_patterns(df_signal, filt_rows, modality)

            # Calculate behavioral measures (whole task)
            prt[idx_c, idx_m] = \
                (df_signal.loc[filt_rows, f'hit_time_{modality}'].sum() + ttr * (no_fal + no_mis)) \
              / (no_hit + no_fal + no_mis)
            dpr[idx_c, idx_m] = _calculate_dprime(no_hit, no_crr, no_mis, no_fal)
            
            # Grab also simpler measures for exclusion criteria
            hit_rate[idx_c, idx_m] = no_hit / (no_hit + no_mis)
            fal_rate[idx_c, idx_m] = no_fal / (no_fal + no_crr)

            for idx_b in range(n_blocks):

                filt_rows = (df_signal['condition'] == condition) \
                          & (df_signal['block'] == idx_b)

                # Count hits, correct rejections, misses and false alarms
                no_hit, no_crr, no_mis, no_fal = _get_resp_patterns(
                    df_signal, filt_rows, modality
                )

                # Calculate behavioral measures (block level)
                acc_block[idx_c, idx_m, idx_b] = no_hit / (no_hit + no_mis)
                prt_block[idx_c, idx_m, idx_b] = \
                    (df_signal.loc[filt_rows, f'hit_time_{modality}'].sum() + ttr * (no_fal + no_mis)) \
                  / (no_hit + no_fal + no_mis)
                dpr_block[idx_c, idx_m, idx_b] = _calculate_dprime(no_hit, no_crr, no_mis, no_fal)

    acc = np.nanmean(acc_block, axis=2)
    
    return (np.stack((acc, prt, dpr, hit_rate, fal_rate)), np.stack((acc_block, prt_block, dpr_block)))


#%%Load group assignment table and exclude subjects with incomplete data.

df_group = pd.read_csv('data/group_assignment.csv')
df_group = df_group.loc[~df_group['group'].isna(), :]
df_group.head()

#Process all logs and save aggregated behavioral data to file. Data description:

#beh: np.array containing behavioral measures calculated for entire dual n-back run
#beh_block: contains the same measures calculated for individual task blocks
#meta: dictionary describing fields in beh and beh_block

#%%
root = '/home/kmb/Desktop/Neuroscience/Projects/FINC_learning_brain/data/sourcedata/behavioral'

task_meta = {
    'n_stims': 12,
    'n_blocks': 10,
    'n_modalities': 2,
    'n_conditions': 2,
    'n_sessions': 4,
    'n_measures': 3,
    'ttr': 20000,
    'n_subjects': df_group.shape[0]
}

beh = np.full((task_meta['n_subjects'], task_meta['n_sessions'], 
               task_meta['n_measures'] + 2, task_meta['n_conditions'], 
               task_meta['n_modalities']), 
              np.nan)
beh_block = np.full((task_meta['n_subjects'], task_meta['n_sessions'], 
                     task_meta['n_measures'], task_meta['n_conditions'], 
                     task_meta['n_modalities'], task_meta['n_blocks']), 
                    np.nan)

for ix_ses, ses in enumerate(['1', '2', '3', '4']):
    for ix_sub, sub in enumerate(df_group['sub']):
        
        logpath = os.path.join(root, f'0{sub[-2:]}d_{ses}-dual_n-back_modified.log')
        print(logpath)
        
        if os.path.exists(logpath):

            beh[ix_sub, ix_ses], beh_block[ix_sub, ix_ses] = \
                    calculate_behavioral_measures(convert_logfile(logpath, task_meta), task_meta)
                
# Create metadata describing beh and beh_block fields
meta = {
    'dim1': df_group['sub'].tolist(),
    'dim2': [f'ses-{ses}' for ses in range(1, task_meta['n_sessions'] + 1)],
    'dim3': ['acc', 'prt', 'dpr', 'hit_rate', 'fal_rate'],
    'dim4': ['1-back', '2-back'],
    'dim5': ['vis', 'aud'],
    'dim6': [f'block-{block:02}' for block in range(1, task_meta['n_blocks'] + 1)],
    'exp': list(df_group['group'] == 'Experimental'),
    'con': list(df_group['group'] == 'Control'),
}

# Save aggregated behavioral measures
np.save('data/aggregated_behavioral_data.npy', beh)
np.save('data/aggregated_behavioral_data_block.npy', beh_block)
with open('data/aggregated_behavioral_data.json', 'w') as f:
    json.dump(meta, f)






























