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
os.chdir(path)
os.getcwd()


dataset = pd.read_csv("summarynorm3bio.csv", sep=',')
testdataset = pd.read_csv("summarynorm3testbio.csv", sep=',')

dataset['Level'] = pd.cut(dataset['ER'], bins=[0, 0.3, 0.7, 10], labels = ['Low','Intermediate','High'], include_lowest=True)
dataset['Level'].value_counts()
#Intermediate    30582
#High             6268
#Low               247

testdataset['Level'] = pd.cut(testdataset['ER'], bins=[0, 0.3, 0.7, 10], labels = ['Low','Intermediate','High'], include_lowest=True)
testdataset['Level'].value_counts()
#Intermediate    35848
#High             2756
#Low               265

encoder = LabelEncoder()
encoder.fit(dataset['Level'])
encoded_Y = encoder.transform(dataset['Level'])
dummy_y = np_utils.to_categorical(encoded_Y)


testencoder = LabelEncoder()
testencoder.fit(testdataset['Level'])
testencoded_Y = testencoder.transform(testdataset['Level'])
testdummy_y = np_utils.to_categorical(testencoded_Y)


TRAIN_RATIO = 0.90#0.9999 # change this number as needed
# splitting from https://stackoverflow.com/a/24147363/4725731
# np.random.seed(123) # uncomment this line to get the same split between runs
msk = np.random.rand(len(dataset)) < TRAIN_RATIO 
df1 = dataset[msk]  
df2 = dataset[~msk]
dummy_y1 = dummy_y[msk]
dummy_y2 = dummy_y[~msk]    

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
#['Experiments', 'BodyXferMod', 'BodyXferRate', 'CulturePbind', 'CulturePmet1',
# 'CulturePmet2', 'CLint', 'ExpectedER', 'forwardBias', 'LateralBias', 
# 'MousePbind', 'MousePmet1', 'MousePmet2', 'ER', 'Label', 'Level']

Y_train = dummy_y1
Y_validation = dummy_y2
Y_test = testdummy_y

#Xcolumnnames = ['BodyXferMod', 'BodyXferRate', 'CLint']
Xcolumnnames = columnnames[1:7]
#['BodyXferMod', 'BodyXferRate', 'CulturePbind', 'CulturePmet1', 'CulturePmet2', 'CLint']

X_train = df1[Xcolumnnames].values
X_validation = df2[Xcolumnnames].values
X_test = testdataset[Xcolumnnames].values

X_validation = (X_validation-np.mean(X_train))/np.std(X_train)  
X_test = (X_test-np.mean(X_train))/np.std(X_train)
X_train = (X_train-np.mean(X_train))/np.std(X_train) 

print("number of training examples = " + str(X_train.shape[0]))
print("number of validation examples = " + str(X_validation.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_validation shape: " + str(X_validation.shape))
print("Y_validation shape: " + str(Y_validation.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


def MultiClassificationModel(input_shape):
    
    X_input = Input(input_shape)
    X = Dense(200, activation='relu', kernel_initializer = 'he_normal', name='fc')(X_input)
    X = Dropout(0.1)(X)
    X = Dense(200, activation='relu', kernel_initializer = 'he_normal', name='fc2')(X)
    X = Dropout(0.1)(X)
    X = Dense(200, activation='relu', kernel_initializer = 'he_normal', name='fc3')(X)
    X = Dropout(0.1)(X)
    X = Dense(3, activation='softmax', kernel_initializer = 'he_normal', name='fc4')(X)
    
    model = Model(inputs=X_input, outputs = X , name='MultiClassificationModel')
    
    return model


multiClassificationModel = MultiClassificationModel(X_train.shape[1:])

sgd = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

multiClassificationModel.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])

#multiClassificationModel.load_weights('K5C_str_ste_4_std_weights01182018')
multiClassificationModel.fit(x= X_train, y=Y_train, epochs = 100, batch_size = 64)
multiClassificationModel.summary()

predseval_train = multiClassificationModel.evaluate(X_train, Y_train)
print()
print ("Train Loss = " + str(predseval_train[0]))
print ("Train Accuracy = " + str(predseval_train[1]))

predseval_validation = multiClassificationModel.evaluate(X_validation, Y_validation)
print()
print ("Validation Loss = " + str(predseval_validation[0]))
print ("Validation Accuracy = " + str(predseval_validation[1]))

predseval_test = multiClassificationModel.evaluate(X_test, Y_test)
print()
print ("Test Loss = " + str(predseval_test[0]))
print ("Test Accuracy = " + str(predseval_test[1]))

f_output_Classificationtest = open('K4C_str_ste_7stdATEST.csv', 'w')

preds = multiClassificationModel.predict(X_test)
y_test_class = np.argmax(Y_test, axis=1)
y_pred_class = np.argmax(preds, axis=1)
for idx in range(X_test.shape[0]):
   
    ER_Real = testdataset['ER'][idx]
    Real_Class = y_test_class[idx]
    Pred_Class = y_pred_class[idx]
    f_output_Classificationtest.write(str(idx) + ", predicted: " + str(Pred_Class) + \
                                      "," + ", target: " + str(Real_Class) + \
                                      "," + ", Real ER: " + str(ER_Real) + "\n")

multiClassificationModel.save_weights('K4C_str_ste_7_stdA_weights')

