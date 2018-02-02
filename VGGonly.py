import os
import numpy as np
from keras import applications
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
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
os.chdir(path)
#pathdata = "/home/rim/Test12042017/H5256Lg/"
#path = "H5256Lg/"
path224RGB = "224RGB/"

os.getcwd()

with h5py.File('data224C.h5', 'r') as hf:
    print([key for key in hf.keys()])
    Xdataset= hf[path224RGB+'resized_images'][:]
#below for 128L only    
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
#Ycolumnnames = columnnames[1:2]
####
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


def VGGModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    

    X_input = Input(input_shape)
    
    # Creating a Neural Network (VGG-16)

    X = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(X_input)
    X = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(X)

    # Block 2
    X = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(X)
    X = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(X)

    # Block 3
    X = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(X)
    X = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(X)
    X = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(X)

    # Block 4
    X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(X)
    X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(X)
    X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(X)

    # Block 5
    X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(X)
    X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(X)
    X = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(X)
    
    X = Flatten()(X)
    X = Dense(4096, activation='relu', kernel_initializer = 'he_normal', name='fc')(X)
    X = Dense(4096, activation='relu', kernel_initializer = 'he_normal', name='fc2')(X)
    X = Dense(2048, activation='relu', kernel_initializer = 'he_normal', name='fc3')(X)
    X = Dense(1024, activation='relu', kernel_initializer = 'he_normal', name='fc4')(X)
    X = Dense(512, activation='relu', kernel_initializer = 'he_normal', name='fc5')(X)
    X = Dense(256, activation='relu', kernel_initializer = 'he_normal', name='fc6')(X)
    X = Dense(2, activation='linear', name='regression')(X)
    model = Model(inputs=X_input, outputs = X, name='HappyModel')
    print(model.summary())
    
    return model
 
vggModel = VGGModel(X_train.shape[1:])

weights_path = './vgg16_weights_tf_dim_ordering_tf_kernels.h5'
vggModel.load_weights(weights_path, by_name=True)


# We have to fine tune only last layer then we block all layers and train only the last one
for layer in vggModel.layers:
    layer.trainable = True
for layer in vggModel.layers[:-7]:
    layer.trainable = False


sgd = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
vggModel.compile(loss='mse',
              optimizer=sgd,  
              metrics=["mse","mae"])

vggModel.fit(x=X_train, y=Y_train, epochs = 100, batch_size = 32)


vggModel.summary()

preds = vggModel.predict(x=X_test)


ydata = dataset[columnnames[0:3]] #  Experiment   BodyXferMod   BodyXferRate

f_output_mod = open('VGGonly_MOD.csv', 'w')
f_output_rate = open('VGGonly_RATE.csv', 'w')
ydata = dataset[columnnames]
for idx in range(dataset.shape[0]):

    preds = vggModel.predict(Xdataset[idx:idx+1]/255)
    mod_real = ydata.iat[idx, 1]
    rate_real = ydata.iat[idx, 2]
    mod_pred = preds[0][0]*maxMod
    rate_pred = preds[0][1]*maxRate
    f_output_mod.write(str(idx) + ", predicted: " + str(mod_pred) + "," + ", target: " + str(mod_real)+ "\n")
    f_output_rate.write(str(idx) + ", predicted: " + str(rate_pred) + "," + ", target: " + str(rate_real)+ "\n")

vggModel.save_weights('VGGonly_weights')