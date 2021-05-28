#from https://github.com/poldrack/fmri-analysis-vm/blob/master/analysis/postFMRIPREPmodelling/
#First%20and%20Second%20Level%20Modeling%20(FSL).ipynb
#27.11.2020
#requires pip3 install pybids==0.6.5


import os,json,glob,sys
import numpy
import nibabel
import nilearn.plotting
import pandas as pd
import matplotlib.pyplot as plt
from nipype.caching import Memory
from  nipype.interfaces import fsl, ants      
from nipype.interfaces.base import Bunch
from bids.grabbids import BIDSLayout

datadir='/home/alado/datasets/RBH/preprocessed/derivatives/fmriprep'
results_dir = '/home/alado/datasets/RBH/GLM/1st_2nd'

mem = Memory(base_dir='.')

print('Using data from',datadir)

#%%
layout = BIDSLayout(datadir)
layout.get(type="bold", task="nback", extensions="nii.gz")[0].filename

#layout.get_subjects()
#layout.get_modalities()
#layout.get_types(modality='func')
#layout.get_tasks()
#%%
events = pd.read_csv(os.path.join(datadir, "task-nback_events.tsv"), sep="\t")
events

#%%
#at the end [5]
source_epi = layout.get(type="bold", task="nback", extensions="nii.gz")[5]

confounds = pd.read_csv(os.path.join(datadir, "derivatives", "fmriprep", 
                        "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                        "sub-%s_ses-%s_task-nback_bold_confounds.tsv"%(source_epi.subject,
                        source_epi.session)),
                        sep="\t", na_values="n/a")

info = [Bunch(conditions=['1',
                          '2',
                          '3',
                          '4',
                          '5'],
              onsets=[list(events[events.trial_type == '1'].onset),
                      list(events[events.trial_type == '2'].onset),
                      list(events[events.trial_type == '3'].onset),
                      list(events[events.trial_type == '4'].onset),
                      list(events[events.trial_type == '5'].onset)],
              durations=[list(events[events.trial_type == '1'].duration),
                          list(events[events.trial_type == '2'].duration),
                          list(events[events.trial_type == '3'].duration),
                          list(events[events.trial_type == '4'].duration),
                          list(events[events.trial_type == '5'].duration)],
             regressors=[list(confounds.FramewiseDisplacement.fillna(0)[4:]),
                         list(confounds.aCompCor0[4:]),
                         list(confounds.aCompCor1[4:]),
                         list(confounds.aCompCor2[4:]),
                         list(confounds.aCompCor3[4:]),
                         list(confounds.aCompCor4[4:]),
                         list(confounds.aCompCor5[4:]),
                        ],
             regressor_names=['FramewiseDisplacement',
                              'aCompCor0',
                              'aCompCor1',
                              'aCompCor2',
                              'aCompCor3',
                              'aCompCor4',
                              'aCompCor5',])
       ]

skip = mem.cache(fsl.ExtractROI)
skip_results = skip(in_file=os.path.join(datadir, "derivatives", "fmriprep", 
                                        "sub-%s"%source_epi.subject, "ses-%s"%source_epi.session, "func", 
                                        "sub-%s_ses-%s_task-fingerfootlips_bold_space-MNI152NLin2009cAsym_preproc.nii.gz"%(source_epi.subject,
                                                                                                                           source_epi.session)),
                     t_min=4, t_size=-1)

s = model.SpecifyModel()
s.inputs.input_units = 'secs'
s.inputs.functional_runs = skip_results.outputs.roi_file
s.inputs.time_repetition = layout.get_metadata(source_epi.filename)["RepetitionTime"]
s.inputs.high_pass_filter_cutoff = 128.
s.inputs.subject_info = info
specify_model_results = s.run()
s.inputs

