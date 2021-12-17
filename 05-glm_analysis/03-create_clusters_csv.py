"""
Merge all data frames into one
08.02.2021
@author: alado
"""

import glob
import pandas as pd
import numpy as np

#%%
out_dir = '/home/alado/datasets/RBH/GLM/SLA/'
top_dir = '/home/alado/datasets/RBH/GLM/SLA/voxel/grey_matter_mask'

methods = ['bonferroni', 'fdr', 'fpr', 'permuted', 'uncorrected']
#contrasts = ['4back-2back', '4back-1back', '4back-0back', '4back-fix', '2back-1back', 
#             '2back-0back', '2back-fix', '0back','1back','2back','4back','fix']

contrasts = ['4back-2back', '4back-1back', '4back-0back', '4back-fix', 
                     '2back-1back', '2back-0back', '2back-fix', 'nback-fix', 
                     'task-0back', 'linear_param(slope)_effect', 'quadratic_param(slope)_effect']

#%%
def create_clusters_csv(top_dir, file_name, comp_type):
    
    methods = ['bonferroni', 'fdr', 'fpr', 'permuted', 'uncorrected']
    contrasts = ['4back-2back', '4back-1back', '4back-0back', '4back-fix', 
                     '2back-1back', '2back-0back', '2back-fix', 'nback-fix', 
                     'task-0back', 'linear_param(slope)_effect', 'quadratic_param(slope)_effect']
    
    path = f'{top_dir}/{comp_type}/tables/*'    
    folder_names = []    
    for name in glob.glob(path):
        folder_names.append(name)
    
    if not folder_names:
        print('Folder name list is empty - no such folders exist')
        return
    
    #get file names
    file_names = []
    for i in range(len(folder_names)):
        for name in glob.glob(f'{folder_names[i]}/{file_name}'):
            file_names.append(name)  
    
    columns = ['Unnamed: 0', 'Cluster ID', 'X', 'Y', 'Z', "Peak Stat", "Cluster Size (mm3)", 'method']
    new_clusters_table = pd.DataFrame(columns=columns)     
    for i in range(len(file_names)):
        df =  pd.read_csv(file_names[i])
        if df.empty:
            continue
        else:
            df['contrast'] = file_names[i][65:]
            df['method'] = file_names[i][69:]
            new_clusters_table = pd.concat([new_clusters_table, df], axis=0)
    
    for i in methods:
        new_clusters_table.loc[new_clusters_table['method'].str.contains(i), 'method'] = i
        
    for i in contrasts:
        new_clusters_table.loc[new_clusters_table['contrast'].str.contains(f'_{i}_'), 'contrast'] = i

    new_clusters_table.to_csv(f'{top_dir}/{comp_type}_{file_name[:-2]}_all_clusters_table.csv')
    print(f'clusters csv is created for {file_name[:-2]} {comp_type}')
    return


#%%pre/post sport
#get folder names

file_name = 'sport_*'
create_clusters_csv(top_dir, file_name, comp_type='pre_post')

#pre/post med
file_names = 'med_*'
create_clusters_csv(top_dir, file_names, comp_type='pre_post')

#post control
file_names = 'control_*'
create_clusters_csv(top_dir, file_names, comp_type='post')

#post med
file_names = 'med_*'
create_clusters_csv(top_dir, file_names, comp_type='post')

#post sport
file_names = 'sport_*'
create_clusters_csv(top_dir, file_names, comp_type='post')

#post_1vs1
file_names = 'post_*'
create_clusters_csv(top_dir, file_names, comp_type='post_1vs1')

file_name = 'allgroups_*'
create_clusters_csv(top_dir, file_name, comp_type='pre')


#%%pre all
#get folder names
folder_names = []
for name in glob.glob(f'{top_dir}/pre/tables/*'):
    folder_names.append(name)

#get file names
file_names = []
for i in range(len(folder_names)):
    for name in glob.glob(f'{folder_names[i]}/*'):
        file_names.append(name)
        
columns = ['Unnamed: 0', 'Cluster ID', 'X', 'Y', 'Z', "Peak Stat", "Cluster Size (mm3)", 'method']
new_clusters_table = pd.DataFrame(columns=columns)     
for i in range(len(file_names)):
    df =  pd.read_csv(file_names[i])
    if df.empty:
        continue
    else:
        df['contrast'] = file_names[i][65:]
        df['method'] = file_names[i][69:]
        new_clusters_table = pd.concat([new_clusters_table, df], axis=0)

for i in methods:
    new_clusters_table.loc[new_clusters_table['method'].str.contains(i), 'method'] = i
    
for i in contrasts:
    new_clusters_table.loc[new_clusters_table['contrast'].str.contains(f'_{i}_'), 'contrast'] = i

new_clusters_table.to_csv(f'{out_dir}/voxel/pre_all_clusters_table.csv')

#%%
file_names = 'sport_*'
create_clusters_csv(top_dir, file_names, comp_type='pre')



