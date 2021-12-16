#!/usr/bin/env python
# coding: utf-8

# # Binary Classification- KNN

# NIchole Etienne 

# In[1]:


import os
import re
import h5py
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from collections import Counter
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler 
import matplotlib.pyplot as plt


# In[2]:


data_dir = '/Volumes/LaCie/seizure-prediction/'
patient_names = [name for name in next(os.walk(data_dir))[1]
                 if name != '.ipynb_checkpoints' 
                 if name != 'Patient_1' 
                 if name != 'Patient_2']
patient_names


# In[ ]:


#load 


# In[3]:


for i in range(len(patient_names) - 4):
    patient_name = patient_names[i]
    files_dir = data_dir + patient_name + '/' + 'Power_Values' + '/'
    patient_files = os.listdir(files_dir)
    interictal_features = []
    preictal_features = []
    for j in tqdm(range(len(patient_files)), desc=patient_name):
        patient_file_name = patient_files[j]
        h5_path = files_dir + patient_file_name
        if not re.findall('_test_segment_', patient_file_name):
            with h5py.File(h5_path, "r") as f:
                group_key = list(f.keys())[0]
                data = np.array(f.get(group_key))
            if re.findall('_interictal_segment_', patient_file_name):
                interictal_features.append(data)
            if re.findall('_preictal_segment_', patient_file_name):
                preictal_features.append(data)    
    interictal_ys = np.zeros((len(interictal_features),))
    preictal_ys = np.ones(len(preictal_features))
    
    interictal_features = np.stack(interictal_features, axis=0)
    preictal_features = np.stack(preictal_features, axis=0)
    
    X = np.vstack((interictal_features, preictal_features))
    y = np.concatenate((interictal_ys, preictal_ys), axis=0)


# In[4]:


#print('X:', X.shape)
#print('y:', y.shape)


# In[ ]:


#plots


# In[5]:


delta = X[:,:,0:1]
theta = X[:,:,1:2]
alpha = X[:,:,2:3]
beta = X[:,:,3:4]
low_gamma = X[:,:,4:5]
high_gamma = X[:,:,5:6]


# In[ ]:


#dimensiona of different bands


# In[6]:


# delta across all channels
fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8)
axes[1,4].boxplot(delta[i])
for i in range(0,8):
    axes[0,i].boxplot(delta[i])
    axes[0,i].set_ylabel('Power')
for i in range(0, 8):
    axes[1,i].boxplot(delta[i+8])
    axes[1,i].set_ylabel('Power')
fig.suptitle(r'$\delta$ Band Power Distribution in Channels')
fig.tight_layout()


# In[7]:


# theta across all channels
fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8)
axes[1,4].boxplot(theta[i])
for i in range(0,8):
    axes[0,i].boxplot(theta[i])
    axes[0,i].set_ylabel('Power')
for i in range(0, 8):
    axes[1,i].boxplot(theta[i+8])
    axes[1,i].set_ylabel('Power')
fig.suptitle(r'$\theta$ Band Power Distribution in Channels')
fig.tight_layout()


# In[8]:


# alpha across all channels
fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8)
axes[1,4].boxplot(alpha[i])
for i in range(0,8):
    axes[0,i].boxplot(alpha[i])
    axes[0,i].set_ylabel('Power')
for i in range(0, 8):
    axes[1,i].boxplot(alpha[i+8])
    axes[1,i].set_ylabel('Power')
fig.suptitle(r'$\alpha$ Band Power Distribution in Channels')
fig.tight_layout()


# In[9]:


# beta across all channels
fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8)
axes[1,4].boxplot(beta[i])
for i in range(0,8):
    axes[0,i].boxplot(beta[i])
    axes[0,i].set_ylabel('Power')
for i in range(0, 8):
    axes[1,i].boxplot(beta[i+8])
    axes[1,i].set_ylabel('Power')
fig.suptitle(r'$\beta$ Band Power Distribution in all Channels')
fig.tight_layout()


# In[10]:


# low_gamma across all channels
fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8)
axes[1,4].boxplot(low_gamma[i])
for i in range(0,8):
    axes[0,i].boxplot(low_gamma[i])
    axes[0,i].set_ylabel('Power')
for i in range(0, 8):
    axes[1,i].boxplot(low_gamma[i+8])
    axes[1,i].set_ylabel('Power')
fig.suptitle(r'Low $\gamma$ Band Power Distribution in all Channels')
fig.tight_layout()


# In[11]:


# high_gamma across all channels
fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8)
axes[1,4].boxplot(high_gamma[i])
for i in range(0,8):
    axes[0,i].boxplot(high_gamma[i])
    axes[0,i].set_ylabel('Power')
for i in range(0, 8):
    axes[1,i].boxplot(high_gamma[i+8])
    axes[1,i].set_ylabel('Power')
fig.suptitle(r'High $\gamma$ Band Power Distribution in all Channels')
fig.tight_layout()


# In[ ]:


#class imbalance


# In[5]:


# visualise class scatter
counter = Counter(y)
# scatter plot of examples by class label
for label, _ in counter.items():
    row_ix = np.where(y == label)[0]
    plt.scatter(X[row_ix, 0], X[row_ix, 1], label=str(label))
plt.legend()
plt.show()
print(counter)


# In[ ]:


#Undersampling


# In[6]:


# NearMiss-1 undersampling method
undersample = NearMiss(version=1, n_neighbors=5)
X_under, y_under = undersample.fit_resample(X, y)
# summarize the class distribution
counter = Counter(y_under)

# scatter plot
for label, _ in counter.items():
    row_ix = np.where(y_under == label)[0]
    plt.scatter(X_under[row_ix, 0], X_under[row_ix, 1], label=str(label))
plt.legend()
plt.show()
print(counter)


# In[ ]:


#Oversampling


# In[7]:


# random sampling 
oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)
counter = Counter(y_over)
# scatter 
for label, _ in counter.items():
    row_ix = np.where(y_over == label)[0]
    plt.scatter(X_over[row_ix, 0], X_over[row_ix, 1], label=str(label))
plt.legend()
plt.show()
print(Counter(y_over))


# In[ ]:


#validation set 


# In[8]:


# imbalanced class
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)


# In[9]:


print(X_train.shape)
print(X_val.shape)


# In[10]:


# undersampled class
X_train_under, X_val_under, y_train_under, y_val_under = train_test_split(X_under, y_under, test_size=0.33, 
                                                                         random_state=42)


# In[11]:


print(X_train_under.shape)
print(X_val_under.shape)


# In[12]:


# oversampled class
X_train_over, X_val_over, y_train_over, y_val_over = train_test_split(X_over, y_over, test_size=0.33,
                                                                     random_state=42)


# In[13]:


print(X_train_over.shape)
print(X_val_over.shape)


# In[ ]:


#performance of each model 


# In[14]:


# define hyperparameter
k = 3
target_names = ['Interictal', 'Preictal']


# In[15]:


# imbalanced class
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
y_preds = knn.predict(X_val)

acc = accuracy_score(y_val, y_preds)
print('Imbalanced class')
print(classification_report(y_val, y_preds, target_names=target_names))


# In[16]:


# ROC for  imbalanced class
y_val_score = knn.predict_proba(X_val)[:, 1]
fpr, tpr, _ = roc_curve(y_val, y_val_score)
roc_auc = auc(fpr, tpr)

lw = 2 # 
plt.figure()
plt.plot(fpr, tpr, color='tab:blue',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='tab:red', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Imbalanced')
plt.legend(loc="lower right")
plt.show()


# In[17]:


# undersampled class
knn_under = KNeighborsClassifier(n_neighbors=k)
knn_under.fit(X_under, y_under)
y_preds_under = knn.predict(X_val_under)

acc_under = accuracy_score(y_val_under, y_preds_under)

print('Undersampled class')
print(classification_report(y_val_under, y_preds_under, target_names=target_names))


# In[18]:


# ROC - undersampled class
y_val_under_score = knn.predict_proba(X_val_under)[:, 1]
fpr, tpr, _ = roc_curve(y_val_under, y_val_under_score)
roc_auc = auc(fpr, tpr)

lw = 2 # line width
plt.figure()
plt.plot(fpr, tpr, color='tab:blue',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='tab:red', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Undersampled')
plt.legend(loc="lower right")
plt.show()


# In[19]:


# oversampled class
knn_over = KNeighborsClassifier(n_neighbors=k)
knn_over.fit(X_over, y_over)
y_preds_over = knn.predict(X_val_over)

acc_over = accuracy_score(y_val_over, y_preds_over)
print('Oversampled class')
print(classification_report(y_val_over, y_preds_over, target_names=target_names))


# In[20]:


# ROC - oversampled class
y_val_over_score = knn.predict_proba(X_val_over)[:, 1]
fpr, tpr, _ = roc_curve(y_val_over, y_val_over_score)
roc_auc = auc(fpr, tpr)

lw = 2 # line width
plt.figure()
plt.plot(fpr, tpr, color='tab:blue',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='tab:red', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic - Oversampled')
plt.legend(loc="lower right")
plt.show()


# In[ ]:


#cross validation 


# In[21]:


# 10-fold cross-validation with k=3 for k-NN
knn = KNeighborsClassifier(n_neighbors=5)

k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')

grid.fit(X, y)

grid_mean_scores = grid.cv_results_['mean_test_score']
plt.plot(k_range, grid_mean_scores)
plt.xlabel(r'$k$')
plt.ylabel('Cross-Validated Accuracy')

print('Max accuracy: {max_acc:.2f}'.format(max_acc = np.max(grid_mean_scores)))


# In[ ]:


#end 

