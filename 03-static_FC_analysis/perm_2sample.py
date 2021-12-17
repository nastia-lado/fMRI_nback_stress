#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:55:57 2021

@author: alado
"""


def perm_2sample(X, Y, perms=1000, test='rel', alpha=.05, nan='propagate'):
    """Takes two equally shaped sets of data and performs a 2 sample permutation 
    T-test. This test can handel nan values, so if your sampels are not equal 
    in size you can inset nan's were needed. Data can be 1, 2, 3, or 4 
    dimensions. Data sets can either be arrays, up to 4 dimension, or the 
    outermost dimesion can be a list.
    Args:
        X: first data set where the first dimension is the observations.
        Y: second data set where the first dimension is the observations.
        perms:  the number of times to permute the data, typically 1000 or greater.
        alpha: significance value for the probability of rejecting the null hypothesis
            when it is true. Typically set at 0.05.
    Returns: 
        Global: array of T values where the observed T values were larger than 1-alpha% of 
            all the null permuted data. Best for data that is 1D or 2D unless 
            you want separate comparisons for each index. 
        Part: array of T values where the observed T value was larger than the permuted
            T distributions in the same index point of the array. Very conservative, observed 
            values must be greater than All permuted values in the same index point.
        P_array: array showing p values where the observed t values exceeded 1-alpha% 
            of the permuted values at each index point of a multidimensional array. 
            Typical permutation test output, best for multidimensional data where index matters. 
        null_Tmax: The single T value from the permuted null distribution that is above 1-alpha 
        t_score: array of T values from the observed data.
        nullT: list of arrays of T values from the permuted data.
        max_null: identify outliers in null permuted data after ttest
        max_obs: identify outliers in observed data after ttest"""
    
    from scipy.stats.mstats import mquantiles
    import numpy as np
        
    if np.array(X).shape != np.array(Y).shape:
        raise ValueError('datasets have diffrent dimensions')
    
    dims = len(np.array(X).shape) # find dimensions of data

##################### 4 dimensions to data ####################################
    if dims == 4:
        _, dim1, dim2, dim3 = np.array(X).shape
        t_score = np.zeros((dim1,dim2,dim3))      

    ##### test real data ######
        for i in range(dim1):
            which_subs = ~np.isnan(np.array(X)[:, i, 0, 0])
            samp1 = np.array(X)[which_subs, i, :, :]
            samp2 = np.array(Y)[which_subs, i, :, :]
            
            diff = samp1 - samp2
            n_sub = diff.shape[0]
            t_score[i, :, :] = diff.mean(0)/(diff.std(0)/np.sqrt(n_sub))
   
    ## set up data sets to be permuted over perms times##    
        nullT=[]
        null_count=np.zeros((dim1,dim2,dim3))

        for iteration in range(perms):
            permA=np.copy(X)
            permB=np.copy(Y)
            for a in range(len(permA)):    
    ### create dummy permutation of 1s & 0s in a dim1 x dim2 x dim3 array ###         
                perm_frame = np.random.randint(0,2,size=(dim1,dim2,dim3))
        ###### seperate permuted data into two artifical groups A and B ######
                permA[a]=perm_frame
                permB[a]=perm_frame
        
    #### replace dummy code with the condition data ####
                permA[a]=np.where(permA[a]==0, X[a], Y[a])
                permB[a]=np.where(permB[a]==0, Y[a], X[a])
                
    ##### t-test permuted data conditions A & B ####
            t_perms = np.zeros((dim1,dim2,dim3))
            
            for i in range(dim1):
                which_subs = ~np.isnan(np.array(X)[:, i, 0, 0])
                perm1 = permA[which_subs, i, :, :]
                perm2 = permB[which_subs, i, :, :]
                
                diff = perm1 - perm2
                n_sub = diff.shape[0]
                t_perms[i, :, :] = diff.mean(0)/(diff.std(0)/np.sqrt(n_sub)) 
                  
    # store values from each of the 1000 iterations to create Tmax distribution #
            nullT.append(t_perms)
            
    ##### get p values across all dimentions. 
        null_count=np.zeros((perms,dim1,dim2,dim3))
        for i in range(perms):
            null_count[i]=nullT[i]>=t_score

        null_count=np.sum(null_count, axis=0)
    
        P_val=null_count/perms  
        P_array = np.where(P_val<alpha, P_val,np.nan)
        
    ##### identify Tmax in the permuted data #####
        null_Tmax=mquantiles(np.array(nullT).max((1,2,3)), 1-alpha)
        
        ###identify outliers 
        max_null=np.max(nullT)
        max_obs=np.max(t_score)       

    #####compare observed diffrences vs Tmax permuted diffrences ######
        # index by index difference #
        Part=np.zeros((dim1,dim2,dim3))
        TorF = t_score[np.newaxis, :, :, :] > nullT
        mask = TorF.sum(0) == perms
        Part[mask] = t_score[mask]
        Part[mask == False] = np.nan
    
        # Global difference #    
        Global=np.zeros((dim1,dim2,dim3))
        mask = t_score > null_Tmax
        Global[mask] = t_score[mask]
        Global[mask == False] = np.nan

############ 3 dimension to data ##############################################
    elif dims == 3:
        _, dim1, dim2 = np.array(X).shape
        t_score = np.zeros((dim1,dim2))      

    ##### test real data ######
        for i in range(dim1):
            which_subs = ~np.isnan(np.array(X)[:, i, 0])
            samp1 = np.array(X)[which_subs, i, :]
            samp2 = np.array(Y)[which_subs, i, :]
            
            diff = samp1 - samp2
            n_sub = diff.shape[0]
            t_score[i, :] = diff.mean(0)/(diff.std(0)/np.sqrt(n_sub))
   
    ## set up data sets to be permuted over perms times##    
        nullT=[]
        null_count=np.zeros((dim1,dim2))
        
        for iteration in range(perms):
            permA=np.copy(X)
            permB=np.copy(Y)
            for a in range(len(permA)):    
    
        ### create dummy permutation of 1s & 0s in a dim1 x dim2 x dim3 array ###        
                perm_frame = np.random.randint(0,2,size=(dim1,dim2))
                
        ###### seperate permuted data into two artifical groups A and B ######
                permA[a]=perm_frame
                permB[a]=perm_frame
        
    #### replace dummy code with the condition data ####
                permA[a]=np.where(permA[a]==0, X[a], Y[a])
                permB[a]=np.where(permB[a]==0, Y[a], X[a])
          
    ##### t-test permuted data conditions A & B ####
            t_perms = np.zeros((dim1,dim2))
            
            for i in range(dim1):
                which_subs = ~np.isnan(np.array(X)[:, i, 0])
                perm1 = permA[which_subs, i, :]
                perm2 = permB[which_subs, i, :]
                
                diff = perm1 - perm2
                n_sub = diff.shape[0]
                t_perms[i, :] = diff.mean(0)/(diff.std(0)/np.sqrt(n_sub))    
                
    # store values from each of the 1000 iterations to create Tmax distribution #
            nullT.append(t_perms)

    ##### get p values across all dimentions. 
        null_count=np.zeros((perms,dim1,dim2))
        for i in range(perms):
            null_count[i]=nullT[i]>=t_score

        null_count=np.sum(null_count, axis=0)
    
        P_val=null_count/perms  
        P_array = np.where(P_val<alpha, P_val,np.nan)
        
    ##### identify Tmax in the permuted data #####
        null_Tmax=mquantiles(np.array(nullT).max((1,2)), 1-alpha)
        
        ###identify outliers 
        max_null=np.max(nullT)
        max_obs=np.max(t_score)       

    #####compare observed diffrences vs Tmax permuted diffrences ######
        # index by index difference #
        Part=np.zeros((dim1,dim2))
        TorF = t_score[np.newaxis, :, :] > nullT
        mask = TorF.sum(0) == perms
        Part[mask] = t_score[mask]
        Part[mask == False] = np.nan
    
        # Global difference #    
        Global=np.zeros((dim1,dim2))
        mask = t_score > null_Tmax
        Global[mask] = t_score[mask]
        Global[mask == False] = np.nan
        
########### 2 dimensions to data ############################################## 
    elif dims == 2:
        _, dim1 = np.array(X).shape
        t_score = np.zeros((dim1))      

    ##### test real data ######
        for i in range(dim1):
            which_subs = ~np.isnan(np.array(X)[:, i])
            samp1 = np.array(X)[which_subs, i]
            samp2 = np.array(Y)[which_subs, i]
            
            diff = samp1 - samp2
            n_sub = diff.shape[0]
            t_score[i] = diff.mean(0)/(diff.std(0)/np.sqrt(n_sub))
   
    ## set up data sets to be permuted over perms times##    
        nullT=[]
        null_count=np.zeros((dim1))
        
        for iteration in range(perms):
            permA=np.copy(X)
            permB=np.copy(Y)
            for a in range(len(permA)):    
                
    ### create dummy permutation of 1s & 0s in a dim1 x dim2 x dim3 array ###
                perm_frame = np.random.randint(0,2,size=(dim1))
        ###### seperate permuted data into two artifical groups A and B ######
                permA[a]=perm_frame
                permB[a]=perm_frame
                
    #### replace dummy code with the condition data ####
                permA[a]=np.where(permA[a]==0, X[a], Y[a])
                permB[a]=np.where(permB[a]==0, Y[a], X[a])
   
    ##### t-test permuted data conditions A & B ####
            t_perms = np.zeros((dim1,dim2))
            
            for i in range(dim1):
                which_subs = ~np.isnan(np.array(X)[:, i, 0])
                perm1 = permA[which_subs, i]
                perm2 = permB[which_subs, i]
                
                diff = perm1 - perm2
                n_sub = diff.shape[0]
                t_perms[i] = diff.mean(0)/(diff.std(0)/np.sqrt(n_sub))                   
    
    # store values from each of the 1000 iterations to create Tmax distribution #
            nullT.append(t_perms)
            
    ##### get p values across all dimentions. 
        null_count=np.zeros((perms,dim1))
        for i in range(perms):
            null_count[i]=nullT[i]>=t_score

        null_count=np.sum(null_count, axis=0)
    
        P_val=null_count/perms  
        P_array = np.where(P_val<alpha, P_val,np.nan)
        
    ##### identify Tmax in the permuted data #####
        null_Tmax=mquantiles(np.array(nullT).max((1)), 1-alpha)
        
        ###identify outliers 
        max_null=np.max(nullT)
        max_obs=np.max(t_score)       

    #####compare observed diffrences vs Tmax permuted diffrences ######
        # index by index difference #
        Part=np.zeros((dim1))
        TorF = t_score[np.newaxis, :] > nullT
        mask = TorF.sum(0) == perms
        Part[mask] = t_score[mask]
        Part[mask == False] = np.nan
    
        # Global difference #    
        Global=np.zeros((dim1))
        mask = t_score > null_Tmax
        Global[mask] = t_score[mask]
        Global[mask == False] = np.nan
    
############################################################################
    
    else: raise ValueError('X has to be 2 to 4 dimensional')
    return Global, Part, P_array, null_Tmax, t_score, nullT, max_obs, max_null 