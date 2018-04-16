# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 16:42:50 2018

@author: bsrav
"""

import os
from keras.utils import np_utils
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import pandas as pd
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.model_selection import train_test_split
from keras.layers import MaxPooling2D
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 16

#%%
directory = "E:\\Machine_learning_nd\\machine-learning\\projects\\leafPrediction"
train_dir = os.path.join(directory, 'train')
test_dir = os.path.join(directory, 'test')


def read_img(filepath, size):
    img = image.load_img(os.path.join(directory, filepath), target_size=size)
    img = image.img_to_array(img)
    return img


labels = dict((name, 0) for name in os.listdir(train_dir))
target = []
i = 0
num_cat = len(labels.keys())
cat = list(labels.keys())
for key in labels.keys():
  labels[key] = i
  i+=1

#train dataset
train = []
for category, category_id in labels.items():
    for file in os.listdir(os.path.join(train_dir, category)):
        train.append(['train\\{}\\{}'.format(category, file), category_id, category])
train = pd.DataFrame(train, columns=['file', 'category_id', 'category'])



train = pd.concat([train[train['category'] == c][:200] for c in cat])
train = train.sample(frac=1)
train.index = np.arange(len(train))


#%% plotting the training images 
fig = plt.figure(1, figsize=(num_cat, num_cat))
grid = ImageGrid(fig, 111, nrows_ncols=(num_cat, num_cat), axes_pad=0.05)
i = 0
for category, category_id in labels.items():
    for filepath in train[train['category'] == category]['file'].values[:num_cat]:
        ax = grid[i]
        img = read_img(filepath, (224, 224))
        ax.imshow(img / 255.)
        ax.axis('off')
        if i % num_cat == num_cat - 1:
            ax.text(250, 112, filepath.split('\\')[0], verticalalignment='center')
        i += 1
plt.show();
#%% train and validation
tdata = []
tlabel = []
for index, row in train.iterrows(): 
  tdata.append(read_img(row['file'], (224, 224)))
  tlabel.append(row['category_id'])

tdata = np.asarray(tdata)
tlabel = np.asarray(tlabel)
tlabel = np_utils.to_categorical(tlabel)

X_train, X_val, y_train, y_val = train_test_split(tdata, tlabel, test_size=0.20, random_state=42)

#%%

#test dataset
#test_dir = "E:\\Machine_learning_nd\\machine-learning\\projects\\leafPrediction\\test"
#test = []
#
#for files in os.listdir(test_dir):
#     test.append(read_img(files, (224, 224)))

#%%
#model = applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3))
model = applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(224,224,3),pooling = 'avg')

for layer in model.layers[:]:
    layer.trainable = False
    
x = model.output
#x = Flatten()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.4)(x)
predictions = Dense(12, activation="softmax")(x)

model_final = Model(inputs = model.input, outputs = predictions)

model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.adam(), metrics=["categorical_accuracy"])
#%%
batch_size = 32
epochs = 100

E_Stop = EarlyStopping(monitor='val_categorical_accuracy', min_delta=0, patience=5, verbose=0, mode='auto') 

model_final.fit(X_train, y_train, batch_size = batch_size, epochs = epochs,
                validation_data=(X_val, y_val),callbacks=[E_Stop],shuffle = True)
#%%


#%%


test = []
for file in os.listdir(test_dir):
    test.append(['test/{}'.format(file), file])
test = pd.DataFrame(test, columns=['filepath', 'file'])
#%%
X_test = []
X_test_file = []
for index, row in test.iterrows():
    #print(row['filepath'])
    X_test.append(read_img(row['filepath'], (224, 224)))
    X_test_file.append(row['file'])
X_test = np.asarray(X_test)

#%%

inver_label = {}
for key in labels:
    inver_label[labels[key]] = key

#%%
predictions = model_final.predict(X_test).argmax(-1)
pred_final = []
for i in range(len(predictions)):
    pred_final.append([X_test_file[i],inver_label[predictions[i]]])

result = pd.DataFrame(pred_final,columns = ['file','species'])

result.to_csv('results.csv',index=False)
