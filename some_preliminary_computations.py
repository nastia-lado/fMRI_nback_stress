#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 17:40:53 2021

@author: alado
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

#%%
def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")
#%%
epi_img = nib.load('/home/alado/datasets/RBH/Nifti/sub-02/ses-1/func/sub-02_ses-1_task-nback_run-1_bold.nii')
epi_img_data = epi_img.get_fdata()

mean = np.zeros(10)
for i in range(10):
    mean[i] = np.mean(epi_img_data[:,:,25,i])
    plt.imshow(epi_img_data[:,:,25,i], cmap="gray", origin="lower")
    plt.show()

#%%
plt.plot(mean)
plt.ylim((300,400))
plt.title('Mean intensity of the first 10 scans')
plt.xlabel('scans')
plt.ylabel('intensity (a.u.)')
plt.show()