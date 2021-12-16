#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Nichole Etienne
#CS 534 


# In[ ]:


#required libraries 
from __future__ import print_function
import tensorflow as tf

import random
import numpy as np
import pandas as pd

import scipy.io
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import cv2


# In[ ]:


import os
fl = os.path.join('/Volumes/LaCie/seizure-prediction/Patient_1/Patient_1','Patient_1_test_segment_0001.mat')
data = scipy.io.loadmat(fl)
print(data)


# In[ ]:


from scipy.io import loadmat
annots = loadmat('/Volumes/LaCie/seizure-prediction/Patient_1/Patient_1','Patient_1_test_segment_0001.mat')


# In[ ]:


#channel 0
annots['interictal_segment_1'][0][0][0][0].shape


# In[ ]:


#channel 1
annots['interictal_segment_1'][0][0][0][1].shape


# In[ ]:


#plot
interictal_tst = '/Volumes/LaCie/seizure-prediction/Patient_1/Patient_1/Patient_1_interictal_segment_0001.mat'
preictal_tst = '/Volumes/LaCie/seizure-predictionPatient_1/Patient_1/Patient_1_preictal_segment_0001.mat'
interictal_data = scipy.io.loadmat(interictal_tst)
preictal_data = scipy.io.loadmat(preictal_tst)


# In[ ]:


interictal_array = interictal_data['interictal_segment_1'][0][0][0]
preictal_array = preictal_data['preictal_segment_1'][0][0][0]


# In[ ]:


#1 second of data of channel 0 for preictal patient1 

l = list(range(10000))
for i in l[::5000]:
    print('Interictal')
    i_secs = interictal_array[14][i:i+5000]
    print(interictal_array[14])
    i_f, i_t, i_Sxx = spectrogram(i_secs, return_onesided=False)
    i_SS = np.log1p(i_Sxx)
    plt.imshow(i_SS[:] / np.max(i_SS), cmap='gray')
    plt.show()
    print('Preictal')
    p_secs = preictal_array[1][i:i+5000]
    p_f, p_t, p_Sxx = spectrogram(p_secs, fs=5000, return_onesided=False)
    p_SS = np.log1p(p_Sxx)
    plt.imshow(p_SS[:] / np.max(p_SS), cmap='gray')
    plt.show()
    


# In[ ]:


# training and testing data
all_X = []
all_Y = []

types = ['Patient_1_interictal_segment', 'Patient_1_preictal_segment']

for i,typ in enumerate(types):
    # Looking at 18 files for each event for a balanced dataset
    for j in range(18):
        fl = '/Volumes/LaCie/seizure-prediction/Patient_1/Patient_1/{}_{}.mat'.format(typ, str(j + 1).zfill(4))
        data = scipy.io.loadmat(fl)
        k = typ.replace('Patient_1_', '') + '_'
        d_array = data[k + str(j + 1)][0][0][0]
        # 10 minutes
        lst = list(range(3000000))  
        for m in lst[::5000]:
            arr=[]
            # spectrogram every second-1
            p_secs = d_array[0][m:m+5000]
            p_f, p_t, p_Sxx = spectrogram(p_secs, fs=5000, return_onesided=False)
            p_SS = np.log1p(p_Sxx)
            arr1 = p_SS[:] / np.max(p_SS)
            arr.append(arr1)
            p_secs = d_array[1][m:m+5000]
            p_f, p_t, p_Sxx = spectrogram(p_secs, fs=5000, return_onesided=False)
            p_SS = np.log1p(p_Sxx)         
            arr2 = p_SS[:] / np.max(p_SS)   
            arr.append(arr2)
            p_secs = d_array[2][m:m+5000]
            p_f, p_t, p_Sxx = spectrogram(p_secs, fs=5000, return_onesided=False)
            p_SS = np.log1p(p_Sxx)
            arr3 = p_SS[:] / np.max(p_SS)
            arr.append(arr3)
            arr=np.reshape(arr,(256,22,3))
            resized = cv2.resize(src=arr, dsize=(128,128), interpolation = cv2.INTER_AREA)
            all_X.append(resized)
            all_Y.append(i)


# In[ ]:


#library 
import matplotlib.pyplot as plt

w=10
h=10
fig=plt.figure(figsize=(128, 128))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    img = np.random.randint(100)
    fig.add_subplot(rows, columns, i)
    plt.imshow(all_X[i][:,:,0])
plt.show()


# In[ ]:


import matplotlib.pyplot as plt

w=10
h=10
fig=plt.figure(figsize=(128, 128))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    img = np.random.randint(100)
    fig.add_subplot(rows, columns, i)
    plt.imshow(all_X[i])
plt.show()


# In[ ]:


for i in range(len(all_X)):
  name=f'/Volumes/outputimage/image{i}_label{all_Y[i]}.jpg'
  plt.imsave(name,all_X[i])


# In[ ]:


os.listdir('/Volumes/output')


# In[ ]:




