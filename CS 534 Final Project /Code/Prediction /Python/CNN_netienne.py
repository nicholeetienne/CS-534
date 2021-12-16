#!/usr/bin/env python
# coding: utf-8

#Nichole Etienne 
# In[ ]:


import os
directory=os.listdir('/Volumes/train')
directoryt=os.listdir('/Volumes/test')
print(len(directory))
print(len(directoryt))


# In[ ]:


import cv2
from tqdm.notebook import tqdm
import numpy as np


# In[ ]:


#  validation and training data
def load_train():
  train_img=[]
  train_label=[]
  val_img=[]
  val_label=[]
  for i in tqdm(range(len(directory))):
    if i>9000 and i<12000:
      final_path=os.path.join('/Volumes/training_data',directory[i])
      img=cv2.imread(final_path)
      val_img.append(img)
      val_label.append(int(directory[i][-5]))
    elif i>18000 and i<21000:
      final_path=os.path.join('/Volumes/training_data',directory[i])
      img=cv2.imread(final_path)
      val_img.append(img)
      val_label.append(int(directory[i][-5]))  
    else:
      final_path=os.path.join('/Volumes/training_data',directory[i])
      img=cv2.imread(final_path)
      train_img.append(img)
      train_label.append(int(directory[i][-5]))
# Labels 1 - Interictal data and 0 - preictal data
  return train_img,train_label,val_img,val_label
  


# In[ ]:


train_img,train_label,val_img,val_label=load_train()


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
    plt.imshow(train_img[i])
plt.show()


# In[ ]:


#arrays 
x_train=np.asarray(train_img)
y_train=np.asarray(train_label)
x_val=np.asarray(val_img)
y_val=np.asarray(val_label)


# In[ ]:


from tqdm.notebook import tqdm
import cv2

def load_test():
  test_img=[]
  test_label=[]
  for i in tqdm(range(len(directoryt))):
    final_path=os.path.join('/Volumes/test',directoryt[i])
    img=cv2.imread(final_path)
    test_img.append(img)
    test_label.append(int(directoryt[i][-5]))
  return test_img,test_label


# In[ ]:


test_img,test_label=load_test()


# In[ ]:


import numpy as np
testset=[]
for ele in test_img:
  arr=np.asarray(ele)
  arr=np.expand_dims(arr, axis=0)
  testset.append(arr)


# # EFFICIENTNET

# In[ ]:


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras import layers as L
import efficientnet.tfkeras as efn
from tensorflow.keras.utils import to_categorical


# In[ ]:


#EfficientNet
def normalize(image):
  image -= tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])  # RGB
  image /= tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])  # RGB
  #officially given by pytorch.
  return image

def get_model(input_size, backbone='efficientnet-b3', weights='imagenet', tta=False):
  print(f'Using backbone {backbone} and weights {weights}')
  x = L.Input(shape=input_size, name='imgs', dtype='float32')
  y = normalize(x)
  if backbone.startswith('efficientnet'):
    model_fn = getattr(efn, f'EfficientNetB{backbone[-1]}')

  y = model_fn(input_shape=input_size, weights=weights, include_top=False)(y)
  y = L.GlobalAveragePooling2D()(y)
  y = L.Dropout(0.2)(y)

  y = L.Dense(2, activation='softmax')(y)
  model = tf.keras.Model(x, y)
  return model


model = get_model(input_size= (128,128,3))


# 

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator()


# In[ ]:


model.compile(optimizer=Adam(lr=0.001),
                loss=categorical_crossentropy,
                metrics=[categorical_accuracy])


# In[ ]:


from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',
                                         factor=0.1,
                                         patience=4,
                                         cooldown=2,
                                         min_lr=0.00001,
                                         verbose=1)


# In[ ]:


#
history=model.fit(datagen.flow(x_train, to_categorical(y_train,2), batch_size=64),
                    steps_per_epoch=len(x_train) / 64,validation_data=datagen.flow(x_val, to_categorical(y_val,2), batch_size=64), epochs=50, callbacks=[reduce_learning_rate])


# In[ ]:


model.save_weights('/Volumes/models/final_model_b3epoch100.h5')


# In[ ]:


result_efficientnet=[]
for ele in testset:
  res=model.predict(ele)
  res=np.argmax(res,axis=1)
  result_efficientnet.append(res)


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(test_label, result_efficientnet)


# In[ ]:


import sklearn
sklearn.metrics.precision_recall_fscore_support(test_label, result_efficientnet)


# In[ ]:


from sklearn.metrics import accuracy_score
accuracy_score(test_label,result_efficientnet)


# In[ ]:




