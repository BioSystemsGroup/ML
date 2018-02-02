#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:11:39 2017

@author: rim
"""

import os
import numpy as np
from keras import applications
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


import math
import numpy as np
import pandas as pd
import h5py

import os
path = "./"
#path = "/home/rim/Test12042017/H5224RGB"
os.chdir(path)
#pathdata = "/home/rim/Test12042017/H5256Lg/"
#path = "H5256Lg/"
path224RGB = "224RGB/"

os.getcwd()

with h5py.File('data224C.h5', 'r') as hf:
    print([key for key in hf.keys()])
    Xdataset= hf[path224RGB+'resized_images'][:]
 
Xdataset = Xdataset.reshape(Xdataset.shape[0],Xdataset.shape[1],Xdataset.shape[2],3)
print("Xdataset.shape: ", str(Xdataset.shape))
TRAIN_RATIO = 0.80#0.9999 # change this number as needed
msk = np.random.rand(len(Xdataset)) < TRAIN_RATIO
X_train_orig = Xdataset[msk]
X_test_orig = Xdataset[~msk]

dataset = pd.read_csv("summaryCLint.csv", sep=',')
maxMod = max(dataset['BodyXferMod'])
maxRate = max(dataset['BodyXferRate'])

#columnnames = list(Ydataset.columns.values)
columnnames = list(dataset.columns.values)
#['Experiments', 'BodyXferMod', 'BodyXferRate', 'CulturePbind', 
#'CulturePmet1', 'CulturePmet2', 'CLint']
CLintdataset = dataset[columnnames[6]]
CLintdataset = CLintdataset.values.reshape(CLintdataset.shape[0],1)
df1 = dataset[msk]
df2 = dataset[~msk]
Ycolumnnames = columnnames[1:3]

CLintStore_train = df1[columnnames[6]].values
CLintStore_test = df2[columnnames[6]].values
CLintStore_train = CLintStore_train.reshape(CLintStore_train.shape[0],1)  
CLintStore_test = CLintStore_test.reshape(CLintStore_test.shape[0],1)                     
Y_train_orig = df1[Ycolumnnames]#.as_matrix()
Y_test_orig = df2[Ycolumnnames]#.as_matrix()
Y_train_orig_norm = Y_train_orig.copy(deep=True)
Y_test_orig_norm = Y_test_orig.copy(deep=True)
for column in Ycolumnnames:
    Y_train_orig_norm[column] = Y_train_orig[column]/max(Y_train_orig[column])
    Y_test_orig_norm[column] = Y_test_orig[column]/max(Y_test_orig[column])

X_train = X_train_orig/255
X_test = X_test_orig/255
Y_train = Y_train_orig_norm.values#.values.reshape((2, X_train.shape[1]))
Y_test = Y_test_orig_norm.values#.values.reshape((2, X_test.shape[1]))
                          
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
print ("CLint_train shape: " + str(CLintStore_train.shape))
print ("CLint_test shape: " + str(CLintStore_test.shape))



vgg_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])
layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
X=layer_dict['block5_pool'].output
            
X = Flatten(name='flatten')(X)
X = Dense(4096, activation='relu', kernel_initializer = 'he_normal', name='fc')(X)
X = Dense(4096, activation='relu', kernel_initializer = 'he_normal', name='fc2')(X)
X = Dense(2048, activation='relu', kernel_initializer = 'he_normal', name='fc3')(X)
X = Dense(1024, activation='relu', kernel_initializer = 'he_normal', name='fc4')(X)
X = Dense(512, activation='relu', kernel_initializer = 'he_normal', name='fc5')(X)
X = Dense(256, activation='relu', kernel_initializer = 'he_normal', name='fc6')(X)
X = Dense(2, activation='linear', name='regression')(X)

Input2 = Input(shape=[1], name='intrinsicClearance') 
custom_model =Model(input=[vgg_model.input, Input2], output=X)  
for layer in custom_model.layers:
    layer.trainable = True
for layer in custom_model.layers[:-7]:
    layer.trainable = False



sgd = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

custom_model.compile(loss='mse',
              optimizer=sgd,  
              metrics=["mse","mae"])
    
custom_model.fit(x= [X_train, CLintStore_train], y=Y_train, epochs = 100, batch_size = 32)

ydata = dataset[columnnames[0:3]] #  Experiment   BodyXferMod   BodyXferRate

f_output_mod = open('VGGCLint_mod.csv', 'w')
f_output_rate = open('VGGCLint_rate.csv', 'w')
ydata = dataset[columnnames]
for idx in range(dataset.shape[0]):

    preds = custom_model.predict([Xdataset[idx:idx+1]/255, CLintdataset[idx:idx+1]])
    mod_real = ydata.iat[idx, 1]
    rate_real = ydata.iat[idx, 2]
    mod_pred = preds[0][0]*maxMod
    rate_pred = preds[0][1]*maxRate
    f_output_mod.write(str(idx) + ", predicted: " + str(mod_pred) + "," + ", target: " + str(mod_real)+ "\n")
    f_output_rate.write(str(idx) + ", predicted: " + str(rate_pred) + "," + ", target: " + str(rate_real)+ "\n")
  
custom_model.save_weights('vGGCLint_weights')