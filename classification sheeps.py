#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import glob as gb
import cv2
import os
import numpy as np
import pandas as pd
import tqdm.notebook as tqdm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
import splitfolders
from tensorflow.keras.optimizers import SGD


# In[2]:


path = r'I:\Python\dataSet\tensorflow/SheepFaceImages2/'


# In[9]:


splitfolders.ratio(path,output='SheepFaceImages2',seed=42,ratio=(.7,.2,.1),group_prefix=None)


# In[22]:


splitfolders.fixed(path,output='SheepFaceImages2',seed=42,fixed=(150,140),oversample=False,group_prefix=None)


# In[3]:


for folder in os.listdir(path + 'train'):
    files_train = gb.glob(str(path + 'train//' + folder + '/*.jpg'))
    print(f'number of files {len(files_train)} in folder {folder}')


# In[4]:


for folder in os.listdir(path + 'test'):
    files_test = gb.glob(str(path + 'test//' + folder + '/*.jpg'))
    print(f'number of files {len(files_test)} in folder {folder}')


# In[5]:


for folder in os.listdir(path + 'val'):
    files_val = gb.glob(str(path + 'val//' + folder + '/*.jpg'))
    print(f'number of files {len(files_val)} in folder {folder}')


# In[6]:


code = {'Marino':0,'Poll Dorset':1,'Suffolk':2,'White Suffolk':3}
def getCode(n):
    for x, y in code.items():
        if n == y :
            return x 


# In[7]:


def sizeFun(name_file):
    list_size = []
    for files in name_file:
        imgs = cv2.imread(files)
        list_size.append(imgs.shape)
    ls = pd.Series(list_size).value_counts()
    print(ls)


# In[8]:


sizeFun(files_train)
sizeFun(files_test)
sizeFun(files_val)


# In[9]:


Image_size = 100   
def resizeFunTrain(path,Image_size,code):
    x_train = []
    y_train = []
    ls = []
    for folder in os.listdir(path + 'train'):
        files_train = gb.glob(str(path + 'train//' + folder + '/*.jpg'))
        for files in files_train:
            imgs = cv2.imread(files)
            img_resize = cv2.resize(imgs,(Image_size,Image_size))
            x_train.append(img_resize)
            y_train.append(code[folder])
            ls.append(img_resize.shape)
    valu_resize = pd.Series(ls).value_counts()
    return x_train, y_train,valu_resize


# In[10]:


def resizeFunTest(path,Image_size,code):
    x_test = []
    y_test = []
    ls = []
    for folder in os.listdir(path + 'test'):
        files_test = gb.glob(str(path + 'test//' + folder + '/*.jpg'))
        for files in files_test:
            imgs = cv2.imread(files)
            img_resize = cv2.resize(imgs,(Image_size,Image_size))
            x_test.append(img_resize)
            y_test.append(code[folder])
            ls.append(img_resize.shape)
    valu_resize = pd.Series(ls).value_counts()
    return x_test, y_test,valu_resize 


# In[11]:


def resizeFunVal(path,Image_size,code):
    x_val = []
    y_val = []
    ls = []
    for folder in os.listdir(path + 'val'):
        files_val = gb.glob(str(path + 'val//' + folder + '/*.jpg'))
        for files in files_val:
            imgs = cv2.imread(files)
            img_resize = cv2.resize(imgs,(Image_size,Image_size))
            x_val.append(img_resize)
            y_val.append(code[folder])
            ls.append(img_resize.shape)
    valu_resize = pd.Series(ls).value_counts()
    return x_val, y_val,valu_resize


# In[44]:


x_train, y_train, size_img_train  = resizeFunTrain(path,Image_size,code)
x_test, y_test, size_img_test  = resizeFunTest(path,Image_size,code)
x_val, y_val, size_img_val = resizeFunVal(path,Image_size,code)


# In[13]:


#x = x_train + x_test + x_val
#y = y_train + y_test + y_val


# In[14]:


np.random.shuffle(x_train)
np.random.shuffle(y_train)

np.random.shuffle(x_test)
np.random.shuffle(y_test)

np.random.shuffle(x_val)
np.random.shuffle(y_val)


# In[15]:


plt.figure(figsize=(20,20))
for n, i in enumerate(np.random.randint(0,len(x_train),10)):
    plt.subplot(5,5,n+1)
    plt.imshow(x_train[i])
    plt.title(getCode(y_train[i]))


# In[16]:


#X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[17]:


#x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size = 0.2)


# In[45]:


x_train = np.array(x_train)
y_train = np.array(y_train)
x_val = np.array(x_val)
y_val = np.array(y_val)


# In[46]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu', input_shape=(Image_size,Image_size,3)),
    tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(4,4),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(4, activation='softmax')
])


# In[47]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy' , metrics=['accuracy'])


# In[63]:


model.fit(x_train, y_train, epochs=40)


# In[ ]:





# In[5]:


#model = tf.keras.models.load_model('model_sheeps.model')


# In[48]:


early_stopping = tf.keras.callbacks.EarlyStopping(patience=8)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.0000001)
call_backs = [early_stopping, reduce_lr] 


# In[49]:


data_train_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                       shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
data_val_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


# In[50]:


train_generator = data_train_generator.flow(x_train,y_train,batch_size=30)
valid_generator = data_val_generator.flow(x_val,y_val)


# In[25]:


plt.figure(figsize=(20,20))
for i in range(0,15):
    plt.subplot(5,3,i+1)
    for x, y in train_generator:
        img = x[0]
        plt.imshow(img)
        break
plt.tight_layout()
plt.show()


# In[135]:


#history = model.fit(train_generator, epochs=40, validation_data=valid_generator, verbose=1)


# In[51]:


history = model.fit(train_generator, batch_size=45, validation_data=valid_generator, epochs=30,
                    verbose=1, callbacks = [early_stopping, reduce_lr])


# In[59]:


plt.plot(history.history['val_loss'], color='red', label='val_loss')
plt.plot(history.history['loss'], color='blue', label='train_loss')
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.legend()


# In[62]:


plt.plot(history.history['val_accuracy'], color='red', label='val_accuracy')
plt.plot(history.history['accuracy'], color='blue', label='train_accuracy')
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.legend()


# In[66]:


model.evaluate(np.array(x_test), np.array(y_test))


# In[68]:


pred = model.predict(np.array(x_test))


# In[87]:


codes = ['Marino','Poll Dorset','Suffolk','White Suffolk']
cn = np.argmax(pred[150])
plt.imshow(x_test[150])
plt.xlabel(codes[cn])


# In[ ]:




