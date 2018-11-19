#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:19:58 2018

@author: hananhindy
"""
import pandas as pd
import numpy as np
import numpy.random as rng
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class DatasetHandler:
    def __init__(self, path, dataset_name, verbose = True):
        self.dataset_name = dataset_name
        if dataset_name == 'kdd':
            self.dataset = pd.read_csv(path, header=None)
            self.dataset[41] = self.dataset[41].str[:-1]
            self.dataset[42] = ''        #to add class (DoS, U2R, ....)
            self.dataset = self.dataset.values
            
            self.add_kdd_main_classes(verbose)
        elif dataset_name == 'STA':
           normal_path = path + '/11jun.10percentNormalno_syn.csv'
           dos_path = path + '/14jun.10percentAttackno_syn.csv'
           ddos_path = path + '/15jun.10percentAttackno_syn.csv'
            
           self.dataset_dictionary = {}  
           self.dataset_dictionary['normal'] = pd.read_csv(normal_path, header = None).values
           self.dataset_dictionary['dos'] = pd.read_csv(dos_path, header = None).values
           self.dataset_dictionary['ddos'] = pd.read_csv(ddos_path, header = None).values
        elif dataset_name == 'SCADA':
           self.dataset = pd.read_csv(path)
           self.dataset = self.dataset.dropna().values
    
    def add_kdd_main_classes(self, verbose):
        base_classes_map = {}
        base_classes_map['normal'] =  'normal'
        base_classes_map['back'] = 'dos'
        base_classes_map['buffer_overflow'] = 'u2r'
        base_classes_map['ftp_write'] =  'r2l'
        base_classes_map['guess_passwd'] =  'r2l'
        base_classes_map['imap'] =  'r2l'
        base_classes_map['ipsweep'] =  'probe'
        base_classes_map['land'] =  'dos'
        base_classes_map['loadmodule'] =  'u2r'
        base_classes_map['multihop'] =  'r2l'
        base_classes_map['nmap'] =  'probe'
        base_classes_map['neptune'] =  'dos'
        base_classes_map['perl'] =  'u2r'
        base_classes_map['phf'] =  'r2l'
        base_classes_map['pod'] =  'dos'
        base_classes_map['portsweep'] = 'probe'
        base_classes_map['rootkit'] =  'u2r'
        base_classes_map['satan'] =  'probe'
        base_classes_map['smurf'] =  'dos'
        base_classes_map['spy'] =  'r2l'
        base_classes_map['teardrop'] =  'dos'
        base_classes_map['warezclient'] =  'r2l'
        base_classes_map['warezmaster'] =  'r2l'
        
        for key in base_classes_map:
            if verbose:
                print('"{}" has {} instances'.
                      format(key, np.size(self.dataset[self.dataset[:,41] == key, :], axis=0)))
                
            self.dataset[self.dataset[:, 41] == key, 42] = base_classes_map[key]
    
    def get_classes(self):
        print(self.dataset_name)
        if self.dataset_name == 'kdd':
            temp = np.unique(self.dataset[:, 42])
            temp[0], temp[1] = temp[1], temp[0]
        elif self.dataset_name == 'STA':
            temp = [*self.dataset_dictionary.keys()]
        elif self.dataset_name == 'SCADA':
            temp = np.unique(self.dataset[:, 12])
            temp[0], temp[6] = temp[6], temp[0]

        return temp
    
    def encode_split(self, training_categories, testing_categories, max_instances_count = -1, verbose = True):
        self.training_categories = training_categories
        self.testing_categories = testing_categories
        
        if self.dataset_name == 'kdd':
            label_encoder_1 = LabelEncoder()
            label_encoder_2 = LabelEncoder()
            label_encoder_3 = LabelEncoder()
            one_hot_encoder = OneHotEncoder(categorical_features = [1,2,3])
    
            self.dataset[:, 1] = label_encoder_1.fit_transform(self.dataset[:, 1])
            self.dataset[:, 2] = label_encoder_2.fit_transform(self.dataset[:, 2])        
            self.dataset[:, 3] = label_encoder_3.fit_transform(self.dataset[:, 3])
            self.dataset_features = one_hot_encoder.fit_transform(self.dataset[:, :-2]).toarray()
        
        self.training_dataset = {}
        self.testing_dataset = {}
        self.training_instances_count = {} 
        self.testing_instances_count = {} 
        
        if verbose and max_instances_count != -1:
            print('! Restricting the numebr of instances from each class to {}'.format(max_instances_count))
        
        if training_categories == testing_categories:
            print('\nTraining:Testing 80%:20%\n')
            for category in training_categories:
                if self.dataset_name == 'STA':
                    temp = self.dataset_dictionary[category]
                elif self.dataset_name == 'kdd':
                    temp = self.dataset_features[self.dataset[:, 42] == category , :]
                elif self.dataset_name == 'SCADA':
                    temp = self.dataset[self.dataset[:, 12] == category, 0: 10]
                             
                temp_size = np.size(temp, axis = 0)
                if max_instances_count != -1:
                    temp_size = min(temp_size, max_instances_count)
                testing_start_index = int(0.8 * temp_size)
                
                self.training_dataset[category] = temp[0:testing_start_index, :]
                self.testing_dataset[category] = temp[testing_start_index:temp_size, :]
                self.training_instances_count[category] = testing_start_index
                self.testing_instances_count[category] = temp_size - testing_start_index
                
        else:
            if self.dataset_name == 'STA':
                print('ERROR! Cannot apply this to STA dataset')
                return
            
            for training in training_categories:
                if self.dataset_name == 'kdd':
                    temp = self.dataset_features[self.dataset[:, 42] == training , :]
                elif self.dataset_name == 'SCADA':
                    temp = self.dataset[self.dataset[:, 12] == training , 0: 10]

                temp_size = np.size(temp, axis = 0)
                if max_instances_count != -1:
                    temp_size = min(temp_size, max_instances_count)
                
                self.training_dataset[training] = temp[0:temp_size, :]
                self.training_instances_count[training] = temp_size
            
            for testing in testing_categories:
                if self.dataset_name == 'kdd':
                    temp = self.dataset_features[self.dataset[:, 42] == testing , :]
                elif self.dataset_name == 'SCADA':
                    temp = self.dataset[self.dataset[:, 12] == testing , 0: 10]
                    
                temp_size = np.size(temp, axis = 0)
                if max_instances_count != -1:
                    temp_size = min(temp_size, max_instances_count)      
                
                self.testing_dataset[testing] = temp[0:temp_size, :]
                self.testing_instances_count[testing] = temp_size
        
        if self.dataset_name == 'kdd':     
            self.number_of_features = np.size(self.dataset_features, axis = 1)
            del self.dataset
        elif self.dataset_name == 'SCADA':     
            self.number_of_features = 10
            del self.dataset
        else:
            self.number_of_features = np.size(self.dataset_dictionary[training_categories[0]], axis = 1)

        
    def get_batch(self, batch_size, verbose):
        #"""Create batch of n pairs, half same class, half different class"""
        #randomly sample several classes to use in the batch
        n_classes = np.size(self.training_categories)
        selected_classes = rng.choice(n_classes,size=(batch_size,),replace=True)
    
        #initialize 2 empty arrays for the input image batch
        pairs = [np.zeros((batch_size, self.number_of_features)) for i in range(2)]
        
        #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets = np.zeros((batch_size,))
        targets[batch_size//2:] = 1
    
        for i in range(batch_size):
            current_class = self.training_categories[selected_classes[i]]
            
            idx_1 = rng.randint(0, self.training_instances_count[current_class])
            pairs[0][i, :] = self.training_dataset[current_class][idx_1, :].reshape(self.number_of_features)
            
            if i >= batch_size // 2:
                class_2 = current_class  
            else: 
                #add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                class_2 = self.training_categories[(selected_classes[i] + rng.randint(1, n_classes)) % n_classes]
            
            idx_2 = rng.randint(0, self.training_instances_count[class_2])
            pairs[1][i, :] = self.training_dataset[class_2][idx_2, :].reshape(self.number_of_features)
    
        return pairs, targets

        
    def make_oneshot_task(self, testing_validation_windows):
        n_classes = np.size(self.testing_categories)
        chosen_classes = rng.choice(range(n_classes), size = (testing_validation_windows,),replace = False)       
         
        true_category = self.testing_categories[chosen_classes[0]]
        ex1, ex2 = rng.choice(self.testing_instances_count[true_category],replace=False,size=(2,))
        
        test_pair = np.asarray([self.testing_dataset[true_category][ex1, :]]*testing_validation_windows).reshape(testing_validation_windows, self.number_of_features)
        
        support_set = np.zeros((testing_validation_windows, self.number_of_features))
        
        for i in range(testing_validation_windows):
            current_category = self.testing_categories[chosen_classes[i]]
            index = rng.randint(0, self.testing_instances_count[current_category])
            support_set[i, :] = self.testing_dataset[current_category][index, :]
    
        support_set[0,:] = self.testing_dataset[true_category][ex2,:]
    
        support_set = support_set.reshape(testing_validation_windows, self.number_of_features)
        targets = np.zeros((testing_validation_windows,))
        targets[0] = 1
        targets, test_pair, support_set = shuffle(targets, test_pair, support_set)
        
        pairs = [test_pair,support_set]
    
        return pairs, targets 


    def test_oneshot(self, model, testing_batch_size, testing_validation_windows, verbose):
        n_correct = 0
        
        if verbose:
            print("\nEvaluating model on {} random {} way one-shot learning tasks ...".format(testing_batch_size, testing_validation_windows))
            
        for i in range(testing_batch_size):
            inputs, targets = self.make_oneshot_task(testing_validation_windows)
            probs = model.predict(inputs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
         
        accuracy = (100.0*n_correct / testing_batch_size)
        if verbose:
            print("Got an accuracy of {}% way one-shot learning accuracy".format(accuracy))
    
        return accuracy

    def test_oneshot_new_classes_vs_all(self, model, testing_batch_size, verbose):
        n_correct = 0
        n_correct_not_training = 0
       
        for i in range(testing_batch_size):
            n_classes = np.size(self.testing_categories)
            true_category_index = rng.choice(range(n_classes),size=(1,),replace=False)       
             
            true_category = self.testing_categories[true_category_index[0]]
            ex1, ex2 = rng.choice(self.testing_instances_count[true_category],replace=False,size=(2,))
            
            training_count = np.size(self.training_categories)            
            test_pair = np.asarray([self.testing_dataset[true_category][ex1, :]]*training_count).reshape(training_count, self.number_of_features)
            
            support_set = np.zeros((training_count, self.number_of_features))
            
            for i in range(training_count):
                current_category = self.training_categories[i]
                index = rng.randint(0, self.training_instances_count[current_category])
                support_set[i,:] = self.training_dataset[current_category][index, :]
                
            support_set = support_set.reshape(training_count, self.number_of_features)
            pairs = [test_pair,support_set]
            probs = model.predict(pairs)
            
            #Check if its in the training set 
            if np.argmax(probs) < 0.6:
                n_correct_not_training += 1
            
            
            test_pair = np.asarray([self.testing_dataset[true_category][ex1, :]]*(training_count+1)).reshape(training_count+1, self.number_of_features)
            
            support_set = np.zeros((training_count+1, self.number_of_features))
            
            for i in range(training_count):
                current_category = self.training_categories[i]
                index = rng.randint(0, self.training_instances_count[current_category])
                support_set[i,:] = self.training_dataset[current_category][index, :]                
        
            support_set[training_count,:] = self.testing_dataset[true_category][ex2,:]
        
            support_set = support_set.reshape(training_count+1, self.number_of_features)
            targets = np.zeros((training_count+1,))
            targets[training_count] = 1
            targets, test_pair, support_set = shuffle(targets, test_pair, support_set)
              
            pairs = [test_pair,support_set]
            
            probs = model.predict(pairs)
            if np.argmax(probs) == np.argmax(targets):
                n_correct+=1
         
        accuracy = (100.0*n_correct / testing_batch_size)
        accuracy_not_in_training = (100.0*n_correct_not_training / testing_batch_size)
        
        if verbose:
            print("Got an accuracy of {}% classifying new class vs all classes".format(accuracy))
            print("Got an accuracy of {}% new class classified as new attack".format(accuracy_not_in_training))
        
        return accuracy, accuracy_not_in_training
    
            
