#Motion and outliers control
#Last edited: 11-10-2020

#Step 0: Loading libraries
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("..")

import os
import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from fctools import denoise, figures, stats
from nistats.design_matrix import make_first_level_design_matrix

# Matplotlib settings
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'Helvetica'

small = 25
medium = 25
bigger = 25

plt.rc('font', size=small)          # controls default text sizes
plt.rc('axes', titlesize=small)     # fontsize of the axes title
plt.rc('axes', linewidth=2.2)
plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
plt.rc('legend', fontsize=small)    # legend fontsize
plt.rc('figure', titlesize=bigger)  # fontsize of the figure title
plt.rc('lines', linewidth=2.2, color='gray')

#%%Step 1: Data preparation
# Setting main input directory
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/func_connect/05-motion_and_outlier_control/'

# Selecting subjects who finished the study
groups = pd.read_csv('/home/alado/datasets/RBH/behavioural/subj_info.csv')
subs = pd.Series.tolist(groups['sub'][groups['group'].isin(['sport', 'med', 'control'])])
subs.remove('sub-16')
subs.remove('sub-17')
subs.remove('sub-20')
subs.remove('sub-21')
subs.remove('sub-22')
subs.remove('sub-23')
subs.remove('sub-24')
subs.remove('sub-50')

#WHAT IS IT?
trained = (groups.group == 'sport') | (groups.group == 'med')
trained_subs = groups[trained]
#subs = trained_subs['sub'].values
#print(f'Sample size: {len(subs)}')

# Setting sessions and task names
sess = ['ses-1', 'ses-3']
#tasks = ['rest']
tasks = ['nback']

# Loading events
#condition = ['1', '2', '3', '4', '5']
#condition = denoise.get_condition_column(events)
#condition['no'] = np.arange(len(condition))
#condition.head()

suffix = '_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

#%%Step 2: Looping over subjects and merging their confound files

#tasks = ['rest', 'nback']
tasks = ['nback']
confounds = pd.DataFrame()

for sub in subs:
    for ses in sess:
        for task in tasks:
            
            # Getting directory/file names
            sub_dir = f'{top_dir}{sub}/{ses}/func/'
            sub_name = f'{sub}_{ses}_task-{task}' 
            sub_dir = f'{top_dir}/preprocessed/fmriprep/{sub}/{ses}/func/'
            sub_name = f'{sub}_{ses}_task-nback_run-1' 
            epi_preproc_path = f'{sub_dir}{sub_name}{suffix}'
            events_dir = f'{top_dir}/Nifti/{sub}/{ses}/func/'
            events_path = f'{events_dir}{sub_name}_events.tsv'
            
            # Loading events
            events = pd.read_csv(events_path, delimiter='\t')
            #condition = ['1', '2', '3', '4', '5']
            condition = denoise.get_condition_column(events)
            condition['no'] = np.arange(len(condition))
            condition.head()
           

            # Loading confound data
            confounds_path1 = f'{top_dir}/func_connect/01-conf_clean_acompcor/{sub}/{sub_name}_bold_confounds_clean_acompcor.csv'
            confounds_path2 = f'{sub_dir}{sub_name}_desc-confounds_regressors.tsv'
            
            if not os.path.exists(confounds_path1):
                print(f'{sub}{ses}{task} does not exist')
            else:
            
                conf1 = pd.read_csv(confounds_path1)
                conf1 = pd.DataFrame(conf1, columns =['scrubbing'])
                conf1['sub'] = sub
                conf1['ses'] = ses
                conf1['task'] = task
                conf1['no'] = np.arange(len(conf1))

                conf2 = pd.read_csv(confounds_path2, delimiter = '\t')
                conf2 = pd.DataFrame(conf2, columns =['FramewiseDisplacement'])
                conf2.FramewiseDisplacement[0] = 0
                conf2['no'] = np.arange(len(conf2))

                conf_all = pd.merge(conf1, conf2, on = 'no')
                
                if task == 'rest':
                    conf_all['condition'] = 'rest'
                
                else:
                    conf_all = pd.merge(conf_all, condition, on = 'no')

            confounds = pd.concat((confounds, conf_all))

confounds = pd.merge(confounds, trained_subs, on = 'sub')
confounds = confounds.rename(index=str, columns={"group": "Group", "ses": "Session", "condition": "Condition" })

confounds.to_csv(f'{top_dir}/func_connect/05-motion_and_outlier_control/coundfounds_summary.csv', 
                 sep = ',', index = False)
confounds.head()

# Read confounds from .csv
#confounds = pd.read_csv('/home/finc/Dropbox/Projects/LearningBrain/data/neuroimaging/01-extracted_timeseries/coundfounds_summary.csv')
#confounds.head()

#%%Step 3: Summarizing pandas dataframe

f = {'scrubbing':['sum'], 'FramewiseDisplacement':['mean']}

# Total
outlier_all = confounds.groupby(['sub','Session','Group','task']).agg(f).reset_index()

outlier_all['OutlierPerc'] = [((row.scrubbing['sum']/340)*100) if row.task[0] == 'nback' else ((row.scrubbing['sum']/305)*100) for i, row in outlier_all.iterrows()]

#outlier_all['OutlierPerc'] = (outlier_all.scrubbing['sum']/340)*100
outlier_all['FD'] = outlier_all.FramewiseDisplacement['mean']

# Grouped by condition
outlier_cond = confounds.groupby(['sub','Session','Group', 'Condition']).agg(f).reset_index()
outlier_cond['OutlierPerc'] = (outlier_cond.scrubbing['sum']/150)*100
outlier_cond['FD'] = outlier_cond.FramewiseDisplacement['mean']
        
outlier_cond = outlier_cond[outlier_cond.Condition != 'intro']
outlier_all

#%%Step 4: Plotting
# Setting colors for groups
col_groups = ['#379dbc','#ee8c00']
sns.set_palette(col_groups)
sns.palplot(sns.color_palette(col_groups))

# Plotting mean total framewise displacement (FD)
ax = figures.swarm_box_plot(x="Session", y="FD", hue = 'Group', data = outlier_all[outlier_all.task == 'nback'])
ax.set(title=' ')
ax.set(ylabel='Mean FD (mm)')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.hlines(0.2, -1, 4, colors='darkgray', linestyles ='dashed')
plt.setp(ax.spines.values(), linewidth=2.2)
plt.ylim(0.02, 0.35)  

plt.savefig(f'{out_dir}fig_S1a.pdf', bbox_inches="tight", dpi=300)

# Plotting total percent of outlier scans
ax = figures.swarm_box_plot(x="Session", y="OutlierPerc", hue = 'Group', data =  outlier_all[outlier_all.task == 'nback'])
ax.set(title=' ')
ax.set(ylabel='Outlier volumes (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.hlines(10, -1, 4, colors='darkgray', linestyles ='dashed')
plt.setp(ax.spines.values(), linewidth=2.2)
plt.ylim(-1, 23) 
plt.savefig(f'{out_dir}fig_S1b.pdf', bbox_inches="tight", dpi=300)

#REST!
# Plotting mean total framewise displacement (FD)
#ax = figures.swarm_box_plot(x="Session", y="FD", hue = 'Group', data = outlier_all[outlier_all.task == 'rest'])
#ax.set(title=' ')
#ax.set(ylabel='Mean FD (mm)')

#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.hlines(0.2, -1, 4, colors='darkgray', linestyles ='dashed')
#plt.setp(ax.spines.values(), linewidth=2.2)
#plt.ylim(0.02, 0.35)  

#plt.savefig(f'{out_dir}fig_S1a.pdf', bbox_inches="tight", dpi=300)


# Plotting total percent of outlier scans
#ax = figures.swarm_box_plot(x="Session", y="OutlierPerc", hue = 'Group', data =  outlier_all[(outlier_all.task == 'rest')])
#ax.set(title=' ')
#ax.set(ylabel='Outlier volumes (%)')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.hlines(10, -1, 4, colors='darkgray', linestyles ='dashed')
#plt.setp(ax.spines.values(), linewidth=2.2)
#plt.ylim(-1, 23) 
#plt.savefig(f'{out_dir}fig_S1b.pdf', bbox_inches="tight", dpi=300)

#--- setting colors for conditions

col_cond = ['#88d958','#f98766']
sns.set_palette(col_cond)
sns.palplot(sns.color_palette(col_cond))

outlier_cond.head()

#CONTROL!!!!!
# Plotting mean framewise displacement (FD) for each condition: Control
#ax = figures.swarm_box_plot(x="Session", y="FD", hue = 'Condition', data = outlier_cond[outlier_cond.Group == 'control'])
#ax.set(title='Control')
#ax.set(ylabel='Mean FD (mm)')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.hlines(0.2, -1, 4, colors='darkgray', linestyles ='dashed')
#plt.ylim(0.02, 0.35)  
#plt.setp(ax.spines.values(), linewidth=2.2)
#plt.savefig(f'{out_dir}fig_S1c.pdf', bbox_inches="tight", dpi=300)

# Plotting mean framewise displacement (FD) for each condition: Control
ax = figures.swarm_box_plot(x="Session", y="FD", hue = 'Condition', data = outlier_cond[outlier_cond.Group == 'med'])
ax.set(title='med')
ax.set(ylabel='Mean FD (mm)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.hlines(0.2, -1, 4, colors='darkgray', linestyles ='dashed')
plt.setp(ax.spines.values(), linewidth=2.2)
plt.ylim(0.02, 0.35)  
plt.savefig(f'{out_dir}fig_S1d.pdf', bbox_inches="tight", dpi=300)

# Plotting mean framewise displacement (FD) for each condition: Control
ax = figures.swarm_box_plot(x="Session", y="FD", hue = 'Condition', data = outlier_cond[outlier_cond.Group == 'sport'])
ax.set(title='sport')
ax.set(ylabel='Mean FD (mm)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.hlines(0.2, -1, 4, colors='darkgray', linestyles ='dashed')
plt.setp(ax.spines.values(), linewidth=2.2)
plt.ylim(0.02, 0.35)  
plt.savefig(f'{out_dir}fig_S1d.pdf', bbox_inches="tight", dpi=300)

# Plotting % outlier scans for each condition: Control
#ax = figures.swarm_box_plot(x="Session", y="OutlierPerc", hue = 'Condition', data = outlier_cond[outlier_cond.Group == 'control'])
#ax.set(title='Control')
#ax.set(ylabel='Outlier volumes (%)')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#plt.hlines(10, -1, 4, colors='darkgray', linestyles ='dashed')
#plt.setp(ax.spines.values(), linewidth=2.2)
#plt.ylim(-1, 23) 
#plt.savefig(f'{out_dir}fig_S1e.pdf', bbox_inches="tight", dpi=300)

# Plotting % outlier scans for each condition: Control
ax = figures.swarm_box_plot(x="Session", y="OutlierPerc", hue = 'Condition', data = outlier_cond[outlier_cond.Group == 'med'])
ax.set(title='med')
ax.set(ylabel='Outlier volumes (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.hlines(10, -1, 4, colors='darkgray', linestyles ='dashed')
plt.setp(ax.spines.values(), linewidth=2.2)
plt.ylim(-1, 23) 
plt.savefig(f'{out_dir}fig_S1f.pdf', bbox_inches="tight", dpi=300)

# Plotting % outlier scans for each condition: Control
ax = figures.swarm_box_plot(x="Session", y="OutlierPerc", hue = 'Condition', data = outlier_cond[outlier_cond.Group == 'sport'])
ax.set(title='sport')
ax.set(ylabel='Outlier volumes (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.hlines(10, -1, 4, colors='darkgray', linestyles ='dashed')
plt.setp(ax.spines.values(), linewidth=2.2)
plt.ylim(-1, 23) 
plt.savefig(f'{out_dir}fig_S1f.pdf', bbox_inches="tight", dpi=300)

#%%Step 5: Deciding which subjects to exclude
outlier_all.head()

criteria = (outlier_all.FD > 0.2) | (outlier_all.OutlierPerc > 10)
excluded = outlier_all[criteria]
ex = np.unique(excluded['sub'].values)

print(f'Subjects to exclude due to FD > 0.2 and % of outlier scans > 10%: {ex}')

excluded

criteria_cond = ((outlier_cond.FD > 0.2) | (outlier_cond.OutlierPerc > 10)) & (outlier_cond.Condition != 'rest')
excluded_cond = outlier_cond[criteria_cond]

ex_cond = np.unique(excluded_cond['sub'].values)

print(f'Subjects to exclude due to FD > 0.2 and % of outlier scans > 10%: {ex_cond}')

excluded_cond

#to_exclude = ['sub-13', 'sub-21', 'sub-23', 'sub-50']

#%%Step 6: Calculate mean of FD and % of outlier scans
clean_cond = outlier_cond[~outlier_cond['sub'].isin(to_exclude)]
clean_all = outlier_all[~outlier_all['sub'].isin(to_exclude)]

clean_cond.groupby(['Group', 'Session','Condition']).mean()[['Group', 'FD', 'OutlierPerc']]

clean_all.groupby(['Group', 'Session']).mean()[['Group', 'FD', 'OutlierPerc']]

#%%Step 7: Calculate test statistic to compare groups/sessions/conditions
sess = ['ses-1', 'ses-2', 'ses-3', 'ses-4']
conds = ['1-back', '2-back']
groups = ['Control', 'Experimental']

#Comparing conditions

# Differences in FD between conditions for experimental group
stats.ttest_rel_cond('Experimental','FD', data = clean_cond)


# Differences in FD between conditions for control group
stats.ttest_rel_cond('Control','FD', data = clean_cond)

# Differences in % of oultier scans between conditions for experimental group
stats.ttest_rel_cond('Experimental','OutlierPerc', data = clean_cond)

# Differences in % of oultier scans between conditions for control group
stats.ttest_rel_cond('Control','OutlierPerc', data = clean_cond)

#%%Step 8: Comparing sessions 

stats.ttest_rel_sess('Experimental','FD', data = clean_all)[:,:,0]

stats.ttest_rel_sess('Experimental','OutlierPerc', data = clean_all)[:,:,0]

stats.ttest_rel_sess('Control','FD', data = clean_all)[:,:,0]

stats.ttest_rel_sess('Control','OutlierPerc', data = clean_all)[:,:,0]

#%% Step 9: Comparing groups 

stats.ttest_ind_groups('FD', clean_all)

stats.ttest_ind_groups('OutlierPerc', clean_all)










































