"""
Merge all data frames into one
08.02.2021
@author: alado
"""

import glob
import pandas as pd
import numpy as np

#%%
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/GLM/SLA/'
top_dir = '/home/alado/datasets/RBH/GLM/SLA/voxel/grey_matter_mask'

methods = ['bonferroni', 'fdr', 'fpr', 'permuted', 'uncorrected']
contrasts = ['4back-2back', '4back-1back', '4back-0back', '4back-fix', '2back-1back', 
             '2back-0back', '2back-fix', '0back','1back','2back','4back','fix']


folder_dir = f'{top_dir}/GLM/SLA/voxel/grey_matter_mask/pre_post/ind_test/tables/*'
file_names = 'sport_*'
#%%
def create_clusters_csv(top_dir, file_names, comp_type, test_type):
    
    methods = ['bonferroni', 'fdr', 'fpr', 'permuted', 'uncorrected']
    contrasts = ['4back-2back', '4back-1back', '4back-0back', '4back-fix', '2back-1back', 
                 '2back-0back', '2back-fix', '0back','1back','2back','4back','fix']
    
    folder_names = []
    if test_type == None:
        path = f'{top_dir}/{comp_type}/tables/*'
    else:
        path = f'{top_dir}/{comp_type}/{test_type}/tables/*'
        
    for name in glob.glob(path):
        folder_names.append(name)
    
    if not folder_names:
        print('Folder name list is empty - no such folders exist')
        return
    
    #get file names
    file_names = []
    for i in range(len(folder_names)):
        for name in glob.glob(f'{folder_names[i]}/{file_names}'):
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
    
    if test_type == None:
        new_clusters_table.to_csv(f'{top_dir}/{comp_type}_{file_names[:-2]}_all_clusters_table.csv')
    else:
        new_clusters_table.to_csv(f'{top_dir}/{comp_type}_{test_type}_{file_names[:-2]}_all_clusters_table.csv')
    print(f'clusters csv if created for {file_names[:-2]}')
    return

#%%
folder_dir = f'{top_dir}/GLM/SLA/voxel/grey_matter_mask/pre_post/ind_test/tables/*'
file_names = 'sport_*'
create_clusters_csv(top_dir, file_names, comp_type='pre_post', test_type='ind_test')
#%%pre all
#get folder names
folder_names = []
for name in glob.glob(f'{top_dir}/GLM/SLA/voxel/grey_matter_mask/pre/tables/*'):
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

#%%pre/post sport
#get folder names
folder_names = []
for name in glob.glob(f'{top_dir}/GLM/SLA/voxel/grey_matter_mask/pre_post/ind_test/tables/*'):
    folder_names.append(name)

#get file names
file_names = []
for i in range(len(folder_names)):
    for name in glob.glob(f'{folder_names[i]}/sport_*'):
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

new_clusters_table.to_csv(f'{out_dir}/voxel/grey_matter_mask/pre_post_ind_test_sport_all_clusters_table.csv')

#%%pre/post sport
#get folder names
folder_names = []
for name in glob.glob(f'{top_dir}/GLM/SLA/voxel/grey_matter_mask/pre_post/rel_test/tables/*'):
    folder_names.append(name)

#get file names
file_names = []
for i in range(len(folder_names)):
    for name in glob.glob(f'{folder_names[i]}/sport_*'):
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

new_clusters_table.to_csv(f'{out_dir}/voxel/grey_matter_mask/pre_post_rel_test_sport_all_clusters_table.csv')

#%%pre/post med ind
#get folder names
folder_names = []
for name in glob.glob(f'{top_dir}/GLM/SLA/voxel/grey_matter_mask/pre_post/ind_test/tables/*'):
    folder_names.append(name)

#get file names
file_names = []
for i in range(len(folder_names)):
    for name in glob.glob(f'{folder_names[i]}/med_*'):
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

new_clusters_table.to_csv(f'{out_dir}/voxel/grey_matter_mask/pre_post_ind_test_med_all_clusters_table.csv')

#%%pre/post med rel
#get folder names
folder_names = []
for name in glob.glob(f'{top_dir}/GLM/SLA/voxel/grey_matter_mask/pre_post/rel_test/tables/*'):
    folder_names.append(name)

#get file names
file_names = []
for i in range(len(folder_names)):
    for name in glob.glob(f'{folder_names[i]}/med_*'):
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

new_clusters_table.to_csv(f'{out_dir}/voxel/grey_matter_mask/pre_post_rel_test_med_all_clusters_table.csv')

#%%post control
#get folder names
folder_names = []
for name in glob.glob(f'{top_dir}/GLM/SLA/voxel/grey_matter_mask/post/tables/*'):
    folder_names.append(name)

#get file names
file_names = []
for i in range(len(folder_names)):
    for name in glob.glob(f'{folder_names[i]}/control_*'):
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

new_clusters_table.to_csv(f'{out_dir}/voxel/grey_matter_mask/post_control_all_clusters_table.csv')

#%%post med
#get folder names
folder_names = []
for name in glob.glob(f'{top_dir}/GLM/SLA/voxel/grey_matter_mask/post/tables/*'):
    folder_names.append(name)

#get file names
file_names = []
for i in range(len(folder_names)):
    for name in glob.glob(f'{folder_names[i]}/med_*'):
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

new_clusters_table.to_csv(f'{out_dir}/voxel/grey_matter_mask/post_med_all_clusters_table.csv')

#%%post sport
#get folder names
folder_names = []
for name in glob.glob(f'{top_dir}/GLM/SLA/voxel/grey_matter_mask/post/tables/*'):
    folder_names.append(name)

#get file names
file_names = []
for i in range(len(folder_names)):
    for name in glob.glob(f'{folder_names[i]}/sport_*'):
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

new_clusters_table.to_csv(f'{out_dir}/voxel/grey_matter_mask/post_sport_all_clusters_table.csv')

#%%post_1vs1
#get folder names
folder_names = []
for name in glob.glob(f'{top_dir}/GLM/SLA/voxel/grey_matter_mask/post_1vs1/tables/*'):
    folder_names.append(name)

#get file names
file_names = []
for i in range(len(folder_names)):
    for name in glob.glob(f'{folder_names[i]}/med_*'):
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

new_clusters_table.to_csv(f'{out_dir}/voxel/grey_matter_mask/post_sport_vs_med_all_clusters_table.csv')






