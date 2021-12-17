"""
Conver MNI coordinate into AAL/BA system
10.02.2021
"""

import glob
import pandas as pd
from rpy2.robjects.packages import importr
from atlasreader import create_output
  
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
utils.install_packages('devtools')
devtools = importr('devtools')
utils.install_packages('remotes')
remotes = importr('remotes')
remotes.install_github("yunshiuan/label4MRI")
utils.install_packages('label4MRI')
label4MRI = importr('label4MRI')

#%%
top_dir = '/home/alado/datasets/RBH'
out_dir = '/home/alado/datasets/RBH/GLM/SLA'

file_names = []
for name in glob.glob(f'{out_dir}/voxel/grey_matter_mask/*_table.csv'):
    file_names.append(name)
    
for i in range(len(file_names)):
    df =  pd.read_csv(file_names[i])
    df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]
    
    aal_distance = []
    aal_label = []
    ba_distance = []
    ba_label = []
    for j in range(len(df)):
        a = label4MRI.mni_to_region_name(x = int(round(df.iloc[j,1])), y = int(round(df.iloc[j,2])), z = int(round(df.iloc[j,3])))
        bb = tuple(a)
        aal_distance.append(bb[0][0])
        aal_label.append(bb[1][0])
        ba_distance.append(bb[2][0])
        ba_label.append(bb[3][0])
    
    df['aal_distance'] = aal_distance
    df['aal_label'] = aal_label
    df['ba_distance'] = ba_distance
    df['ba_label'] = ba_label
    
    df.to_csv(f'{file_names[i][:-4]}_labels.csv')
    



























