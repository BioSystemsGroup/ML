#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 01:39:34 2017

@author: rim
"""
#tensorflow"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import math
import os

path="./"
os.chdir(path)
os.getcwd()
dataset = pd.read_csv("summaryCLint.csv", sep=',')

TRAIN_RATIO = 0.80#0.9999 # change this number as needed
# splitting from https://stackoverflow.com/a/24147363/4725731
# np.random.seed(123) # uncomment this line to get the same split between runs
msk = np.random.rand(len(dataset)) < TRAIN_RATIO 
df1 = dataset[msk]  
df2 = dataset[~msk]  

train_row_indices = np.add(df1.index.tolist(), 2) # subtract one to fix indexing to match summary.csv
test_row_indices = np.add(df2.index.tolist(), 2)
doublecheck_later = dataset.iloc[df2.index,:]
print("doublecheck_later Test data: ")
print(doublecheck_later)

df1=df1.reset_index(drop=True)
df2=df2.reset_index(drop=True)
columnnames = list(df1.columns.values)

#['Experiments', 'BodyXferMod', 'BodyXferRate', 'CulturePbind', 
#'CulturePmet1', 'CulturePmet2', 'CLint']

Y_train_orig = df1['CLint']
Y_test_orig = df2['CLint']

Xcolumnnames = columnnames[1:6]
#['BodyXferMod', 'BodyXferRate', 'CulturePbind', 'CulturePmet1', 'CulturePmet2']
X_train = df1[Xcolumnnames]
X_test = df2[Xcolumnnames]

X_train = X_train.T
X_test = X_test.T
Y_train = Y_train_orig.values.reshape((1, X_train.shape[1]))
Y_test = Y_test_orig.values.reshape((1, X_test.shape[1]))

print("number of training examples = " + str(X_train.shape[1]))
print("number of test examples = " + str(X_test.shape[1]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
        
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    #Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X.iloc[:, permutation] 
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    #Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X.iloc[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # end case: (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X.iloc[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches


def create_placeholders(n_x, n_y):
    X = tf.placeholder("float", [n_x, None])
    Y = tf.placeholder("float", [n_y, None])
    pkeep = tf.placeholder(tf.float32)
    return X, Y, pkeep

def initialize_parameters():    
    K = 28
    L = 22
    C = 17
    n_x = 5
    
    tf.set_random_seed(1)
    
    W1 = tf.get_variable("W1", [K, n_x], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [K, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [L, K], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [L, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [C, L], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [C, 1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable("W4", [1, C], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b4 = tf.get_variable("b4", [1, 1], initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4}
    
    return parameters

def forward_propagation(X, parameters, pkeep):    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    A1d = tf.nn.dropout(A1, pkeep)
    Z2 = tf.add(tf.matmul(W2, A1d),b2)
    A2 = tf.nn.relu(Z2)
    A2d = tf.nn.dropout(A2, pkeep)
    Z3 = tf.add(tf.matmul(W3, A2d),b3)
    A3 = tf.nn.relu(Z3)
    A3d = tf.nn.dropout(A3, pkeep)
    Z4 = tf.add(tf.matmul(W4, A3d),b4)
        
    return Z4

def compute_cost(Z4,Y):    
    cost = tf.reduce_mean(tf.squared_difference(Z4, Y))
    A4 = tf.nn.relu(Z4)
    cost2 = tf.reduce_mean(tf.squared_difference(A4, Y))
    return cost, cost2

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 300, minibatch_size = 64, print_cost = True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = [] 
    costs2 = []
    
    X, Y, pkeep= create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z4 = forward_propagation(X, parameters, pkeep)
    A4 = tf.nn.relu(Z4)
    cost, cost2 = compute_cost(Z4, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    optimizer2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost2)
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0. #defines a cost related to an epoch
            epoch_cost2 =0.
            num_minibatches = int(m/minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:
                #select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                _ , minibatch_cost = sess.run([optimizer,cost], feed_dict={X:minibatch_X, Y:minibatch_Y, pkeep:1.0})
                _ , minibatch_cost2 = sess.run([optimizer2,cost2], feed_dict={X:minibatch_X, Y:minibatch_Y, pkeep:1.0})
                
                epoch_cost += minibatch_cost / num_minibatches
                epoch_cost2 += minibatch_cost2 / num_minibatches
                
            #Print the cost every epoch
            if print_cost == True and epoch % 50 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 ==0:
                costs.append(epoch_cost)
                costs2.append(epoch_cost2)
                
        #plot the cost
        
        fig, axes = plt.subplots(nrows=1, ncols=2)
            
        plt.tight_layout() #to address overlapping problems

        axes[0].plot(np.squeeze(costs))
        axes[0].set_ylabel('cost Z4')
        axes[0].set_xlabel('iterations (per tens)')
        axes[0].set_title("Learning rate=" + str(learning_rate))
        
        axes[1].plot(np.squeeze(costs2))
        axes[1].set_ylabel('cost A4')
        axes[1].set_xlabel('iterations (per tens)')
        axes[1].set_title("Learning rate=" + str(learning_rate))
        
        plt.show()
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        train_results = open('train_results.csv', 'w')
        test_results = open('test_results.csv', 'w')
        print("Incorrect predictions saved in {} and {}".format('train_results.csv', 'test_results.csv'))
        
        x_train_eval = sess.run(Z4, feed_dict = {X:X_train, pkeep:1.0})
        #x_train_eval = Z4.eval(feed_dict={X: X_train})
        i = 0
        for idx, val in Y_train_orig.iteritems():
            if np.greater_equal(np.abs(val - x_train_eval[0][idx]), 0.15*val):
                train_results.write(str(train_row_indices[i]) + ", predicted: " + str(x_train_eval[0][idx]) + ", target: " + str(dataset.iloc[train_row_indices[i]-2,:][6]) + "\n")        
            i += 1

        x_test_eval = sess.run(Z4, feed_dict = {X:X_test, pkeep:1.0})
        print("x_test_eval:", x_test_eval)
              
        i = 0
        for idx, val in Y_test_orig.iteritems():                        
            if np.greater_equal(np.abs(val - x_test_eval[0][idx]), 0.15*val):
                test_results.write(str(test_row_indices[i]) + ", predicted: " + str(x_test_eval[0][idx]) +  ", target: " + str(dataset.iloc[test_row_indices[i]-2,:][6]) +  "\n")
            
            i += 1

        
        correct_prediction = tf.less(tf.abs(tf.subtract(Z4,Y)), 0.15*Y)
        
        #calculate accuaracy 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        print("Train Accuracy Z4: ", sess.run(accuracy, feed_dict = {X: X_train, Y:Y_train, pkeep: 1.0}))
        print("Test Accuracy Z4: ", sess.run(accuracy, feed_dict = {X: X_test, Y:Y_test, pkeep: 1.0}))
        
        #correct_prediction2 = tf.equal(tf.argmax(A4), tf.argmax(Y))
        correct_prediction2 = tf.less(tf.abs(tf.subtract(A4,Y)), 0.15*Y)
        accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))
        
        print("Train Accuracy A4: ", sess.run(accuracy2, feed_dict = {X: X_train, Y:Y_train, pkeep: 1.0}))
        print("Test Accuracy A4: ", sess.run(accuracy2, feed_dict = {X: X_test, Y:Y_test, pkeep: 1.0}))
        
        saver.save(sess, 'FourLayerCLintRun.ckpt')
        return parameters, costs, costs2 
parameters, costs, costs2 = model(X_train, Y_train, X_test, Y_test)


def run_test(X_train, Y_train, X_test, Y_test):
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    X, Y, pkeep= create_placeholders(n_x, n_y)
    Z4 = forward_propagation(X, parameters, pkeep)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('FourLayerCLintRun.ckpt.meta')
        saver.restore(sess, 'FourLayerCLintRun.ckpt')
        sess.run(Z4, feed_dict = {X:X_test, pkeep:1.0})
        correct_prediction = tf.less(tf.abs(tf.subtract(Z4,Y)), 0.15*Y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Saved Test Accuracy Z4: ", sess.run(accuracy, feed_dict = {X: X_test, Y:Y_test, pkeep: 1.0}))

run_test(X_train,Y_train, X_test,Y_test)
