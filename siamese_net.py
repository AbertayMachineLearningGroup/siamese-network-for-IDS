#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:19:36 2018

@author: hananhindy
"""
# based on the implementation in https://github.com/sorenbouma/keras-oneshot

from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import load_model

class SiameseNet:
    def __init__(self, input_shape, network_id = 'new_archi_lr_0_0001', dataset_name = 'kdd', verbose = True):
        
        self.left_input = Input(input_shape)
        self.right_input = Input(input_shape)
        self.convnet = Sequential()
        dropout_1 = 0.1
        dropout_2 = 0.1
        dropout_3 = 0.1       
        lr = 0.001
        
        if network_id == 'new_archi_dropout_gradient':
            dropout_1 = 0.3
            dropout_2 = 0.2
        elif network_id == 'new_archi_dropout_less':
            dropout_1 = 0.05
            dropout_2 = 0.05
            dropout_3 = 0.05
        elif network_id == 'new_archi_lr_0_0001':
            lr = 0.0001
        elif network_id == 'new_archi_lr_0_0006':
            lr = 0.0006
        
        if dataset_name == 'kdd' or dataset_name == 'nsl-kdd':
            self.convnet.add(Dense(units = 98, kernel_regularizer=l2(1e-2), kernel_initializer = 'uniform', activation = 'relu', input_shape = input_shape))
            self.convnet.add(Dropout(dropout_1))
            self.convnet.add(Dense(units = 79, kernel_regularizer=l2(1e-2),kernel_initializer =  'uniform',  activation = 'relu'))
            self.convnet.add(Dropout(dropout_2))
            self.convnet.add(Dense(units = 59, kernel_regularizer=l2(1e-2),kernel_initializer =  'uniform',  activation = 'relu'))
            self.convnet.add(Dropout(dropout_3))
            self.convnet.add(Dense(units = 39, kernel_regularizer=l2(1e-2),kernel_initializer =  'uniform',  activation = 'relu'))
            self.convnet.add(Dropout(dropout_3))
            self.convnet.add(Dense(units = 20, kernel_regularizer=l2(1e-2),kernel_initializer =  'uniform',  activation = 'relu'))
        elif dataset_name == 'CICIDS' or dataset_name == 'CICIDS2':
            dropout_1 = 0.1
            dropout_2 = 0.05
            lr = 0.0001
            self.convnet.add(Dense(units = 25, kernel_regularizer=l2(1e-2), kernel_initializer = 'uniform', activation = 'relu', input_shape = input_shape))
            self.convnet.add(Dropout(dropout_1))
            self.convnet.add(Dense(units = 20, kernel_regularizer=l2(1e-2), kernel_initializer = 'uniform', activation = 'relu'))
            self.convnet.add(Dropout(dropout_2))
            self.convnet.add(Dense(units = 15, kernel_regularizer=l2(1e-2), kernel_initializer = 'uniform', activation = 'relu'))
        elif dataset_name == 'SCADA' or dataset_name == 'SCADA_Reduced':
            self.convnet.add(Dense(units = 8, kernel_regularizer=l2(1e-2), kernel_initializer = 'uniform', activation = 'relu', input_shape = input_shape))
               
        #call the convnet Sequential model on each of the input tensors so params will be shared
        self.encoded_l = self.convnet(self.left_input)
        self.encoded_r = self.convnet(self.right_input)
        
        #layer to merge two encoded inputs with the l1 distance between them
        self.L1_layer = Lambda(self.euclidean_distance, output_shape=self.eucl_dist_output_shape)
        
        #call this layer on list of two input tensors.
        self.L1_distance = self.L1_layer([self.encoded_l, self.encoded_r])
        self.siamese_net = Model(inputs=[self.left_input,self.right_input],outputs=self.L1_distance)
        
        self.optimizer = Adam(lr)
        self.siamese_net.compile(loss=self.contrastive_loss, optimizer=self.optimizer, metrics=[self.accuracy])
        
        if verbose:
            print('Siamese Network Created\n')
            
    
    def euclidean_distance(self, vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))
    
    def eucl_dist_output_shape(self, shapes):
        shape1, shape2 = shapes
        return (shape1[0], 1)

    def contrastive_loss(self, y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        margin = 1
        sqaure_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)
    
    def accuracy(self, y_true, y_pred):
        '''Compute classification accuracy with a fixed threshold on distances.
        '''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))
    
    def load_saved_model(self, file_name):
        self.siamese_net = load_model(file_name, custom_objects={'contrastive_loss': self.contrastive_loss, 'accuracy': self.accuracy, 'euclidean_distance': self.euclidean_distance, 'eucl_dist_output_shape': self.eucl_dist_output_shape})
        
        
        
