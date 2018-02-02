#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 16:11:39 2017

@author: rim
"""

import os
import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.utils import layer_utils, np_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from sklearn.preprocessing import LabelEncoder

import math
import numpy as np
import pandas as pd
import h5py

import os
path = "./"
#path = "/home/rim/Test01042018/"
os.chdir(path)
os.getcwd()


dataset = pd.read_csv("summarynorm3bio_with_preds.csv", sep=',')
testdataset = pd.read_csv("summarynorm3testbio_with_preds.csv", sep=',')

TRAIN_RATIO = 0.90#0.9999 # change this number as needed
# splitting from https://stackoverflow.com/a/24147363/4725731
# np.random.seed(123) # uncomment this line to get the same split between runs
msk = np.random.rand(len(dataset)) < TRAIN_RATIO 
df1 = dataset[msk]  
df2 = dataset[~msk]
train_row_indices = np.add(df1.index.tolist(), 2) # subtract one to fix indexing to match summary.csv
validation_row_indices = np.add(df2.index.tolist(), 2)
test_row_indices = np.add(testdataset.index.tolist(), 2)

df1=df1.reset_index(drop=True)
df2=df2.reset_index(drop=True)
testdataset=testdataset.reset_index(drop=True)   

columnnames = list(df1.columns.values)
testcolumnnames = list(testdataset.columns.values)
columnnames == testcolumnnames
#True
#['Experiments', 'BodyXferMod', 'BodyXferRate', 'CulturePbind', 
#'CulturePmet1', 'CulturePmet2', 'CLint', 'ExpectedER', 
#'forwardBias', 'LateralBias', 'MousePbind', 'MousePmet1', 
#'MousePmet2', 'ER', 'Label', 'PredictedMod', 'PredictedRate', 
#'PredictedPbind', 'PredictedPmet1', 'PredictedPmet2']

Y_train_orig = df1['ER']
Y_validation_orig = df2['ER']
Y_test_orig = testdataset['ER']

Xcolumnnames = ['BodyXferMod', 'BodyXferRate','CulturePbind', 'CulturePmet1','CulturePmet2','CLint']
#Xcolumnnames = ['PredictedMod', 'PredictedRate', 'PredictedPbind', 'PredictedPmet1', 'PredictedPmet2']
X_train = df1[Xcolumnnames].values
X_validation = df2[Xcolumnnames].values
X_test = testdataset[Xcolumnnames].values

X_validation = (X_validation-np.mean(X_train))/np.std(X_train)  
X_test = (X_test-np.mean(X_train))/np.std(X_train)
X_train = (X_train-np.mean(X_train))/np.std(X_train)

Y_train = Y_train_orig.values.reshape((X_train.shape[0],1))
Y_validation = Y_validation_orig.values.reshape((X_validation.shape[0],1))
Y_test = Y_test_orig.values.reshape((X_test.shape[0],1)) 

print("number of training examples = " + str(X_train.shape[0]))
print("number of validation examples = " + str(X_validation.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_validation shape: " + str(X_validation.shape))
print("Y_validation shape: " + str(Y_validation.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


def RegressionModel(input_shape):
    
    X_input = Input(input_shape)
    X = Dense(200, activation='relu', kernel_initializer = 'he_normal', name='fc')(X_input)
    X = Dropout(0.65)(X)
    X = Dense(200, activation='relu', kernel_initializer = 'he_normal', name='fc2')(X)
    X = Dropout(0.65)(X)
    X = Dense(200, activation='relu', kernel_initializer = 'he_normal', name='fc3')(X)
    X = Dropout(0.60)(X)
    X = Dense(1, activation='linear', kernel_initializer = 'he_normal', name='fc4')(X)
 
    model = Model(inputs=X_input, outputs = X , name='RegressionModel')
    
    return model


regressionModel =RegressionModel(X_train.shape[1:])

sgd = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

regressionModel.compile(loss='mse', optimizer=sgd, metrics=["mse"]) 

#multiClassificationModel.load_weights('K5C_str_ste_4_std_weights01182018')
regressionModel.fit(x= X_train, y=Y_train, epochs = 100, batch_size = 64)
regressionModel.summary()

f_output_ERtrain = open('K4L_str_ste_7_stdATrain.csv', 'w')               
f_output_ERvalidation = open('K4L_str_ste_7_stdAValidation.csv', 'w')
f_output_ERtest = open('K4L_str_ste_7_stdATest.csv', 'w')

def within15percent(pred, tgt):
    return (abs(pred-tgt)/tgt) <0.15

counter = 0
preds = regressionModel.predict(X_train)
for idx in range(X_train.shape[0]):
       
    ER_real = Y_train[idx][0]
    ER_pred = preds[idx][0]
    f_output_ERtrain.write(str(idx) + ", predicted: " + str(ER_pred) + "," + ", target: " + str(ER_real)+ "\n")
    pred =ER_pred
    tgt = ER_real
    if within15percent(pred, tgt):
        counter += 1
correct_percent_rate = float(counter)/len(X_train)
print("Train_accuracy:" + str(correct_percent_rate))

counter = 0
preds = regressionModel.predict(X_validation)
for idx in range(X_validation.shape[0]):
 
    ER_real = Y_validation[idx][0]
    ER_pred = preds[idx][0]
    f_output_ERvalidation.write(str(idx) + ", predicted: " + str(ER_pred) + "," + ", target: " + str(ER_real)+ "\n")
    pred =ER_pred
    tgt = ER_real
    if within15percent(pred, tgt):
        counter += 1
correct_percent_rate = float(counter)/len(X_validation)
print("Validation_accuracy:" + str(correct_percent_rate))

counter = 0
preds = regressionModel.predict(X_test)
for idx in range(X_test.shape[0]):
        
    ER_real = Y_test[idx][0]
    ER_pred = preds[idx][0]
    f_output_ERtest.write(str(idx) + ", predicted: " + str(ER_pred) + "," + ", target: " + str(ER_real)+ "\n")
    pred =ER_pred
    tgt = ER_real
    if within15percent(pred, tgt):
        counter += 1
correct_percent_rate = float(counter)/len(X_test)
print("Test_accuracy:" + str(correct_percent_rate))


regressionModel.save_weights('K4L_str_ste_7_stdAweights')





