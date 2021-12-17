#Working memory training: Preparation of normalized modularity
#Turning MATLAB array into tidy .csv table for further statistical analyses.
# NB! rewrite the previous one and merge with this one

#Last edited: 16-10-2020

#Step 0: Loading libraries

import numpy as np
import pandas as pd
import scipy

#%%Step 1: Preparing data

# Selecting subjects which finished the study
groups = pd.read_csv('../data/behavioral/group_assignment.csv')

trained = (groups.group == 'Experimental') | (groups.group == 'Control')
trained_subs = groups[trained].reset_index()
subs = trained_subs['sub'].values

# Creating vectors to filter by group
experimental = (trained_subs == 'Experimental')
control = (trained_subs == 'Control')
exp_vector = experimental['group'].values
con_vector = control['group'].values

# Loading normalized modularity values
mat = scipy.io.loadmat('mean_power_normalized_modularity.mat')
Q = mat['q_norm']

#%%Step 2: Creating dataframe
sess = ['Naive', 'Early', 'Middle', 'Late']
cond = ['Rest', '1-back', '2-back']
g = trained_subs['group'].values

Q_results = pd.DataFrame()#pd.DataFrame(columns=['sub', 'group', 'ses', 'cond', 'Q'])
for i, sub in enumerate(subs):
    for j, ses in enumerate(sess):
        for k, con in enumerate(cond):
            val = Q[i, j, k]
            result = pd.DataFrame({'Subject': sub, 
                                   'Group': trained_subs['group'].values[i], 
                                   'Session': ses, 
                                   'Condition': con, 
                                   'Q_norm': val},
                                  index=[0])
            Q_results = pd.concat([Q_results, result], axis = 0)

Q_results.to_csv('../data/neuroimaging/03-modularity/static/Q_normalized_power_tidy.csv', index=False)
Q_results

