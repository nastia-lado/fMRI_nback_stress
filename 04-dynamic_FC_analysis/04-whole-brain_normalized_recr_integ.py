#Whole-brain normalized recruitment/integration
#Last edited: 20-10-2020

#Step 0: Loading libraries

import sys
sys.path.append("..")
import os

#%matplotlib inline

import scipy.io as sio
import numpy as np
from nilearn import plotting 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from fctools import networks, figures

#---- matplotlib settings
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
plt.rcParams['font.family'] = 'Helvetica'

#%%Step 1: Selecting subjects to analysis

top_dir = '/home/finc/Dropbox/Projects/LearningBrain/'

# Selecting subjects which finished the study
groups = pd.read_csv('../data/behavioral/group_assignment.csv')

trained = (groups.group == 'Experimental') | (groups.group == 'Control')
trained_subs = groups[trained]
subs = trained_subs['sub'].values

# Creating vectors to filter by group
experimental = (trained_subs == 'Experimental')
control = (trained_subs == 'Control')

exp_vector = experimental['group'].values
con_vector = control['group'].values

# Dualnback - selecting subjects included into dinal analysis and creating group assignment vectors
dualnback_exclude = ['sub-13', 'sub-21', 'sub-23', 'sub-50'] # higly motion subjects in one of four sessions
dual_vector = [True if sub not in dualnback_exclude else False for sub in subs]
exp_vector = exp_vector[dual_vector]
con_vector = con_vector[dual_vector]
subs = trained_subs['sub'][dual_vector].values

n = sum(dual_vector)
print(f'Number of subject after excluding subjects with higly motion: {n}')
Number of subject after excluding subjects with higly motion: 42


#%%Step 2: Getting modules names and color pallete

# Loading Power et al. (2011) labels
path_power_labels = '../support/modules.txt'

with open(path_power_labels, 'r') as f:
    power_labels = f.read()
power_labels = power_labels.split('\n')
power_labels.sort()
power_labels.remove('')

power_unique_networks = sorted(list(set(power_labels))) 

power_labels_idx = []
for label in power_labels:
    power_labels_idx.append(power_unique_networks.index(label))
    
power_labels = np.array(power_labels_idx)


# Loading Schaefer et al (2018) labels
path_schaefer_labels = '../support/schaefer_networks_300.csv'
schaefer_labels = pd.read_csv(path_schaefer_labels, header=None)[0].values
schaefer_labels.sort()
schaefer_unique_networks = sorted(list(set(schaefer_labels))) 

schaefer_labels_idx = []
for label in schaefer_labels:
    schaefer_labels_idx.append(schaefer_unique_networks.index(label))

schaefer_labels = np.array(schaefer_labels_idx)


#%%Step 3: Calculating mean normalized network rectuitment/integration
#Normalization function

def normalize_networks_mean(matrix, labels, n_iter):
    '''Normalizing recruitment and integration values using permutation approach
    Null module allegiance matrices were created by randomly permuting the correspondence 
    between regions of interest (ROIs) and large-scale systems.
    We then calculated the functional cartography measures for all permuted matrices.  
    To obtain normalized values of recruitment and integration, 
    we divided them by the mean of the corresponding null distribution.
    This procedure yielded null distributions of recruitment 
    and integration coefficients resulting solely from the size of each system.
    
    Args:
        matrix: (N x N)
        labels: (N, )
        n_iter: int
    '''

    n_networks = len(np.unique(labels))

    def calculate_networks_mean(matrix, labels, n_networks):
        '''... '''
        nam = np.zeros((n_networks, n_networks))

        for i in range(n_networks):
            for j in range(n_networks):
                nam[i, j] = np.mean(matrix[np.nonzero(labels == i)][:, np.nonzero(labels == j)])

        return nam

    nam = calculate_networks_mean(matrix, labels, n_networks)
    nam_null = np.zeros((n_networks, n_networks))
    labels_null = labels.copy()

    for _ in range(n_iter):
        np.random.shuffle(labels_null)
        nam_null += calculate_networks_mean(matrix, labels_null, n_networks)

    nam_null /= n_iter    

    return np.divide(nam, nam_null)
Loading allegiance matrices
In [10]:
parcellation = 'power'

P = np.load(f'{top_dir}data/neuroimaging/03-modularity/dynamic/03-allegiance_matrices/allegiance_matrix_{parcellation}_dualnback_opt_mean.npy')
P.shape
Out[10]:
(46, 4, 264, 264)
Calculating mean normalized recruitment/integration
In [18]:
# Calculate mean normalized allegiance
n_net = len(eval(f'{parcellation}_unique_networks'))

norm_mean_allegiance = np.zeros((P.shape[0], P.shape[1], n_net, n_net))

for i in range(P.shape[0]):
    print(trained_subs['sub'].values[i])
    for j in range(P.shape[1]):
        norm_mean_allegiance[i, j] = normalize_networks_mean(P[i, j], eval(f'{parcellation}_labels'), 1000)

np.save(f'{top_dir}data/neuroimaging/03-modularity/dynamic/04-recruitment_integration/whole-brain_{parcellation}_normalized_mean_allegiance.npy', norm_mean_allegiance)

#%%Save results as tidy dataframe
# Create tidy data frame from mean allegiance values
filename = f'whole-brain_{parcellation}_normalized'
norm_mean_allegiance = np.load(f'{top_dir}data/neuroimaging/03-modularity/dynamic/04-recruitment_integration/{filename}_mean_allegiance.npy')

sessions = ['Naive', 'Early', 'Middle', 'Late']
normalized_measures = pd.DataFrame()
recruitment = pd.DataFrame()

for i, sub in enumerate(trained_subs['sub'].values):
    for j, ses in enumerate(sessions):
        for k, net1 in enumerate(eval(f'{parcellation}_unique_networks')): 
            
            networks_allegiance = pd.DataFrame()
            
            for l, net2 in enumerate(eval(f'{parcellation}_unique_networks')):
                
                norm_net_allegiance = norm_mean_allegiance[i,j,k,l]
                pre = pd.DataFrame([[sub, ses, trained_subs['group'].values[i], net1, norm_net_allegiance]], 
                                   columns=['Subject', 'Session', 'Group', 'Network', f'{net2}']) 
                
                if net1 == net2:
                    recruitment = pd.concat((recruitment, pd.DataFrame([[sub, 
                                                                         ses, 
                                                                         trained_subs['group'].values[i], 
                                                                         net1, 
                                                                         norm_net_allegiance]], 
                                   columns=['Subject', 'Session', 'Group', 'Network', 'Recruitment']))) 
                if l == 0:
                    networks_allegiance = pre
                if l > 0:
                    networks_allegiance = pd.merge(networks_allegiance, pre, on = ['Subject', 'Session', 'Group', 'Network'])
            
            normalized_measures = pd.concat((normalized_measures, networks_allegiance), axis=0)

normalized_measures.to_csv(f'{top_dir}data/neuroimaging/03-modularity/dynamic/04-recruitment_integration/{filename}_mean_allegiance_tidy.csv', 
                           index=False)
recruitment.to_csv(f'{top_dir}data/neuroimaging/03-modularity/dynamic/04-recruitment_integration/{filename}_recruitment_tidy.csv', 
                   index=False)

#%%Exploration plots¶

fig, ax = plt.subplots(2, 4)
fig.set_size_inches(22, 10)

group_vectors = [exp_vector, con_vector]
sessions = ['Naive', 'Early', 'Middle', 'Late']
group_lab = ['Experimental', 'Control']
norm_mean_allegiance_clean = norm_mean_allegiance[dual_vector]

for g, group in enumerate(group_vectors):
    for s, ses in enumerate(sessions):
        m_na = norm_mean_allegiance_clean[group, s, :, :].mean(axis=0)
        #ax[ses].imshow(m_na)
        sns.heatmap(m_na, yticklabels = power_unique_networks, 
                          xticklabels = power_unique_networks, 
                          square = True, 
                          cmap = "RdBu_r", 
                          ax=ax[g][s],
                          cbar=None, 







