#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:19:36 2018

@author: hananhindy
"""
# based on the implementation in https://github.com/sorenbouma/keras-oneshot

from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam

class SiameseNet:
    def __init__(self, input_shape, network_id = 'kdd_0'):
        self.left_input = Input(input_shape)
        self.right_input = Input(input_shape)
        self.convnet = Sequential()

        if network_id == 'kdd_0':
            self.convnet.add(Dense(units = 95, kernel_initializer = 'uniform', activation = 'relu', input_shape = input_shape))
            self.convnet.add(Dense(units = 70, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 47, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 23, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
        elif network_id == 'kdd_1':
            self.convnet.add(Dense(units = 98, kernel_initializer = 'uniform', activation = 'relu', input_shape = input_shape))
            self.convnet.add(Dense(units = 79, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 59, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 39, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 20, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
        elif network_id == 'kdd_2':
            self.convnet.add(Dense(units = 101, kernel_initializer = 'uniform', activation = 'relu', input_shape = input_shape))
            self.convnet.add(Dense(units = 84, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 67, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 51, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 34, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 17, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))
        elif network_id == 'kdd_3':
            self.convnet.add(Dense(units = 103, kernel_initializer = 'uniform', activation = 'relu', input_shape = input_shape))
            self.convnet.add(Dense(units = 89, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 74, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 59, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 44, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 30, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 15, kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dense(units = 5, kernel_initializer = 'uniform', activation = 'relu'))       
    
        #call the convnet Sequential model on each of the input tensors so params will be shared
        self.encoded_l = self.convnet(self.left_input)
        self.encoded_r = self.convnet(self.right_input)
        
        #layer to merge two encoded inputs with the l1 distance between them
        self.L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        
        #call this layer on list of two input tensors.
        self.L1_distance = self.L1_layer([self.encoded_l, self.encoded_r])
        self.prediction = Dense(1,activation='sigmoid')(self.L1_distance)
        self.siamese_net = Model(inputs=[self.left_input,self.right_input],outputs=self.prediction)

        self.optimizer = Adam(0.00006)
        self.siamese_net.compile(loss="binary_crossentropy",optimizer=self.optimizer)
        self.siamese_net.count_params()
