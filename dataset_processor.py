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
from sklearn.cluster import KMeans
import uuid

class DatasetHandler:
    def __init__(self, path, dataset_name, verbose = True):
        self.number_of_reps = 1
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
    
    def encode_split(self, training_categories, testing_categories, max_instances_count = -1, k_fold = 0, verbose = True):
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
                     
                testing_start_index = int((1 - 0.2 * (k_fold + 1)) * temp_size)
                testing_end_index = int(testing_start_index + (0.2 * temp_size))
                
                self.training_dataset[category] = np.append(temp[0:testing_start_index, :], temp[testing_end_index: temp_size, :], axis = 0)
                self.testing_dataset[category] = temp[testing_start_index:testing_end_index, :]
                self.training_instances_count[category] = np.size(self.training_dataset[category], axis = 0)
                self.testing_instances_count[category] = np.size(self.testing_dataset[category], axis = 0)                
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
            
        
    def generate_training_representitives(self, number_of_reps, verbose = False):
        self.training_reps = {};
        self.number_of_reps = number_of_reps
        for category in self.training_categories:
            kmenas = KMeans(n_clusters=number_of_reps)
            kmenas.fit(self.training_dataset[category])
            self.training_reps[category] = kmenas.cluster_centers_
            
       
    def get_batch(self, batch_size, verbose):
        #"""Create batch of n pairs, half same class, half different class"""
        #randomly sample several classes to use in the batch
        n_classes = np.size(self.training_categories)
        selected_classes = rng.randint(low = 0, high = n_classes, size = batch_size)
    
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

        
    def make_oneshot_task(self, testing_validation_windows, train_with_all):       
        n_classes = np.size(self.testing_categories)
        chosen_classes = rng.choice(range(n_classes), size = (testing_validation_windows,),replace = False)       

        true_category = self.testing_categories[chosen_classes[0]]
        ex1, ex2 = rng.choice(self.testing_instances_count[true_category],replace=False,size=(2,))
        
        test_pair = np.asarray([self.testing_dataset[true_category][ex1, :]]*testing_validation_windows*self.number_of_reps).reshape(testing_validation_windows*self.number_of_reps, self.number_of_features)
        
        support_set = np.zeros((testing_validation_windows * self.number_of_reps, self.number_of_features))
        
        for i in range(testing_validation_windows):
            if train_with_all == False:
                # Testing similarity testing vs testing
                current_category = self.testing_categories[chosen_classes[i]]
                index = rng.randint(0, self.testing_instances_count[current_category])
                support_set[i, :] = self.testing_dataset[current_category][index, :]
            else:
                current_category = self.training_categories[chosen_classes[i]]

                if hasattr(self, 'training_reps'):
                    for j in range(self.number_of_reps):
                        support_set[i * self.number_of_reps + j, :] = self.training_reps[current_category][j, :]
                else:
                    # Testing Classification testing vs training
                    index = rng.randint(0, self.training_instances_count[current_category])
                    support_set[i, :] = self.training_dataset[current_category][index, :]
        
        #support_set[0,:] = self.testing_dataset[true_category][ex2,:]
       
        support_set = support_set.reshape(testing_validation_windows * self.number_of_reps, self.number_of_features)
        targets = np.zeros((testing_validation_windows * self.number_of_reps,))
        targets[0] = 1
        
        if hasattr(self, 'training_reps'):
            for j in range(self.number_of_reps):
                targets[j] = 1
            
        targets, test_pair, support_set = shuffle(targets, test_pair, support_set)
        
        pairs = [test_pair,support_set]
    
        return pairs, targets 


    def test_oneshot(self, model, testing_batch_size, testing_validation_windows, train_with_all, verbose):
        n_correct = 0
        
        if verbose:
            print("\nEvaluating model on {} random {} way one-shot learning tasks ...".format(testing_batch_size, testing_validation_windows))
            
        for i in range(testing_batch_size):
            inputs, targets = self.make_oneshot_task(testing_validation_windows, train_with_all)
            probs = model.predict(inputs)
            if targets[np.argmax(probs)] == 1:
                n_correct+=1
         
        accuracy = (100.0*n_correct / testing_batch_size)
        if verbose:
            print("Got an accuracy of {}%  on {} way one-shot learning accuracy".format(accuracy, testing_validation_windows))
    
        return accuracy

    def test_oneshot_new_classes_vs_all(self, model, testing_batch_size, verbose):
        n_correct_not_training_th_60 = 0
        
        for i in range(testing_batch_size):
            n_classes = np.size(self.testing_categories)
            true_category_index = rng.choice(range(n_classes),size=(1,),replace=False)       
             
            true_category = self.testing_categories[true_category_index[0]]
            ex1, ex2 = rng.choice(self.testing_instances_count[true_category],replace=False,size=(2,))
            
            training_count = np.size(self.training_categories)            
            test_pair = np.asarray([self.testing_dataset[true_category][ex1, :]]*training_count * self.number_of_reps).reshape(training_count * self.number_of_reps, self.number_of_features)
            
            support_set = np.zeros((training_count * self.number_of_reps, self.number_of_features))
            
            for k in range(training_count):
                current_category = self.training_categories[k]
                if hasattr(self, 'training_reps'):
                    for j in range(self.number_of_reps):
                        support_set[k * self.number_of_reps + j, :] = self.training_reps[current_category][j, :]
                else:                   
                    index = rng.randint(0, self.training_instances_count[current_category])
                    support_set[k,:] = self.training_dataset[current_category][index, :]
                
            support_set = support_set.reshape(training_count * self.number_of_reps, self.number_of_features)
            pairs = [test_pair,support_set]
            probs = model.predict(pairs)
            
            #Check if its in the training set 
            if np.argmax(probs) < 0.6:
                n_correct_not_training_th_60 += 1
            
        accuracy_not_in_training_60 = (100.0*n_correct_not_training_th_60 / testing_batch_size)
        
        accuracy = -1
        if verbose:
            print("Got an accuracy of {}% new class classified as new attack".format(accuracy_not_in_training_60))
        
        return accuracy, accuracy_not_in_training_60 
    
            
    def test_oneshot_adding_labels(self, model, testing_batch_size, reps_from_all = False, verbose= False):
        n_correct_not_training = {}
        thresholds = [60, 65, 70, 75, 80, 85, 90, 95]
        for th in thresholds:
            n_correct_not_training[th] = 0

        n_correct_labels = 0
        
        testing_categories_count = len(self.testing_categories)
        new_labels = {}
        if reps_from_all:
            new_labels_all = {}
        
        categories_counters = {}
        overall_count = 0
        
        for t in range(testing_categories_count): 
            categories_counters[self.testing_categories[t]] = 0
            overall_count += self.testing_instances_count[self.testing_categories[t]]
        
        for outer in range(overall_count):
            # randomly choose category to test that has remaining instances
            while True:
                current_testing_category = self.testing_categories[rng.randint(0, testing_categories_count)]
                if current_testing_category in categories_counters.keys():
                    break
                            
            current_testing_index = categories_counters[current_testing_category]
            categories_counters[current_testing_category] = categories_counters[current_testing_category] + 1
            if categories_counters[current_testing_category] == self.testing_instances_count[current_testing_category]:
                del categories_counters[current_testing_category] 
            
            training_count = np.size(self.training_categories)     
            sub_pair_count = training_count * self.number_of_reps 
            total_pair_count = sub_pair_count + len(new_labels.keys())
                
            test_pair = np.asarray([self.testing_dataset[current_testing_category][current_testing_index, :]]*total_pair_count).reshape(total_pair_count, self.number_of_features)
                
            support_set = np.zeros((total_pair_count, self.number_of_features))
            pair_id = 0
                
            for tr in range(training_count):
                current_category = self.training_categories[tr]
                for r in range(self.number_of_reps):
                    support_set[pair_id, :] = self.training_reps[current_category][r, :]
                    pair_id += 1
                    
            for k in new_labels.keys():
                support_set[pair_id, :] = new_labels[k]
                pair_id += 1

            support_set = support_set.reshape(total_pair_count, self.number_of_features)
            pairs = [test_pair, support_set]
            probs = model.predict(pairs)
            
            #Check if its in the training set 
            if np.argmax(probs[0: sub_pair_count]) < 0.6:
                for th in thresholds:
                    if  np.argmax(probs[0: sub_pair_count]) < th:
                        n_correct_not_training[th] = n_correct_not_training[th] + 1
                if len(new_labels.keys()) > 0:
                    correct_prediction = np.argmax(probs) 
                    if correct_prediction >= sub_pair_count:
                        # matched to new label 
                        if current_testing_category in [*new_labels][correct_prediction - sub_pair_count]:
                            n_correct_labels += 1
           
                new_labels[current_testing_category + str(uuid.uuid4())] = self.testing_dataset[current_testing_category][current_testing_index, :]
                if reps_from_all:
                    new_labels_all[current_testing_category + str(uuid.uuid4())] = self.testing_dataset[current_testing_category][current_testing_index, :]


            # process new labels (not more than 3 from each category)
            keys_list =  [*new_labels]
            for t in range(testing_categories_count): 
                labeles = list(filter(lambda x: self.testing_categories[t] in x, keys_list))
                if len(labeles) > 0 and len(labeles) % 10 == 0:
                    kmenas = KMeans(n_clusters=self.number_of_reps)
                    if reps_from_all:
                        labeles_temp = list(filter(lambda x: self.testing_categories[t] in x, [*new_labels_all]))
                        kmenas.fit([new_labels_all[x] for x in labeles_temp])
                    else:
                        kmenas.fit([new_labels[x] for x in labeles])
                       
                    for l in labeles:
                        del new_labels[l]       
                    
                    for l in range(self.number_of_reps):
                        new_labels[self.testing_categories[t] + str(uuid.uuid4())] = kmenas.cluster_centers_[l]
                        
        accuracy_not_in_training = {}
        for th in thresholds:
            accuracy_not_in_training[th] = (100.0 * n_correct_not_training[th] / overall_count)
        
        #accuracy_not_in_training_60 = (100.0*n_correct_not_training_th_60 / overall_count)
        accuracy_labels = (100.0 * n_correct_labels / overall_count)

        if verbose:
            print("Got an accuracy of {}%".format(accuracy_not_in_training[60]))
            print("Got an accuracy of {}%".format(accuracy_labels))
            
        return accuracy_not_in_training, accuracy_labels
    
    
