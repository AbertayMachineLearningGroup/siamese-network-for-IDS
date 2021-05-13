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
from sklearn import preprocessing
import uuid
import csv
import math 

class DatasetHandler:
    def __init__(self, path, dataset_name, verbose = True):
        self.number_of_reps = 1
        self.dataset_name = dataset_name
        if dataset_name == 'kdd' or dataset_name == 'nsl-kdd':
            self.dataset = pd.read_csv(path, header=None)
            if dataset_name == 'kdd':  
                self.dataset[41] = self.dataset[41].str[:-1]
            self.dataset[42] = ''        #to add class (DoS, U2R, ....)
            self.dataset = self.dataset.values
            
            self.add_kdd_main_classes(verbose)
        elif dataset_name == 'SCADA' or dataset_name == 'SCADA_Reduced':
           self.dataset = pd.read_csv(path)
           self.dataset = self.dataset.dropna().values
        elif dataset_name == 'CICIDS' or dataset_name == 'CICIDS2':
           normal_path = path + '/biflow_Monday-WorkingHours_Fixed.csv'
           hulk_path = path + '/new_biflow_Wednesday-WorkingHours_Hulk.csv'
           slowloris_path = path + '/new_biflow_Wednesday-WorkingHours_slowloris.csv'
           ddos_path = path +'/new_biflow_Friday-WorkingHours_DDoS.csv'
           FTP_path = path + '/new_biflow_Tuesday-WorkingHours_FTP.csv'
           heartbleed_path = path + '/new_biflow_Wednesday-WorkingHours_Heartbleed.csv'
           portscan_path = path + '/new_biflow_Friday-WorkingHours_PortScan.csv'
           SSH_path = path + '/new_biflow_Tuesday-WorkingHours_SSH.csv'
           
           col_to_drop = ['Unnamed: 0', 'ip_src', 'ip_dst', 'num_src_flows', 'src_ip_dst_prt_delta']

           self.dataset_dictionary = {}  
           self.dataset_dictionary['normal'] = pd.read_csv(normal_path).drop(col_to_drop, axis=1).values
           standard_scaler = preprocessing.StandardScaler()
           self.dataset_dictionary['normal'] = standard_scaler.fit_transform(self.dataset_dictionary['normal'])
          
           if dataset_name == 'CICIDS2':
               self.dataset_dictionary['heartbleed'] = pd.read_csv(heartbleed_path).drop(col_to_drop, axis=1).values
               self.dataset_dictionary['heartbleed'] = standard_scaler.transform(self.dataset_dictionary['heartbleed'])
              
                self.dataset_dictionary['ddos'] = pd.read_csv(ddos_path).drop(col_to_drop, axis=1).values
               self.dataset_dictionary['ddos'] = standard_scaler.transform(self.dataset_dictionary['ddos'])

               self.dataset_dictionary['portscan'] = pd.read_csv(portscan_path).drop(col_to_drop, axis=1).values
               self.dataset_dictionary['portscan'] = standard_scaler.transform(self.dataset_dictionary['portscan'])
           
           if dataset_name == 'CICIDS':
               self.dataset_dictionary['hulk'] = pd.read_csv(hulk_path).drop(col_to_drop, axis=1).values
               self.dataset_dictionary['hulk'] = standard_scaler.transform(self.dataset_dictionary['hulk'])

               self.dataset_dictionary['slowloris'] = pd.read_csv(slowloris_path).drop(col_to_drop, axis=1).values
               self.dataset_dictionary['slowloris'] = standard_scaler.transform(self.dataset_dictionary['slowloris'])

           self.dataset_dictionary['ftp'] = pd.read_csv(FTP_path).drop(col_to_drop, axis=1).values
           self.dataset_dictionary['ftp'] = standard_scaler.transform(self.dataset_dictionary['ftp'])
            
           self.dataset_dictionary['ssh'] = pd.read_csv(SSH_path).drop(col_to_drop, axis=1).values
           self.dataset_dictionary['ssh'] = standard_scaler.transform(self.dataset_dictionary['ssh'])
#           

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
        if self.dataset_name == 'kdd' or self.dataset_name == 'nsl-kdd':
            temp = np.unique(self.dataset[:, 42])
            temp[0], temp[1] = temp[1], temp[0]
        elif self.dataset_name == 'CICIDS' or self.dataset_name == 'CICIDS2':
            temp = [*self.dataset_dictionary.keys()]
        elif self.dataset_name == 'SCADA' or self.dataset_name == 'SCADA_Reduced':
            temp = np.unique(self.dataset[:, 12])
            temp[0], temp[6] = temp[6], temp[0]
            if self.dataset_name == 'SCADA_Reduced':
                temp = list(temp)
                temp.remove('7 Floating objects')
                temp.remove('2 Floating objects')
                temp.remove('Plastic bag')
                temp.remove('Sensor Failure')
                temp.remove('Blocked measure 1')
                temp.remove('Blocked measure 2')
                temp.remove('Humidity')
                temp.remove('Person htting low intensity')
                temp.remove('Person htting med intensity')
                temp.remove('Person htting high intensity')
                temp.remove('Wrong connection')
                
        return temp
    
    def encode_split(self, training_categories, testing_categories, max_instances_count = -1, k_fold = 0, verbose = True):
        self.training_categories = training_categories
        self.testing_categories = testing_categories
        
        if self.dataset_name == 'kdd' or self.dataset_name == 'nsl-kdd':
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
        self.dataset_all = {}
        self.dataset_all_count = {}
        if verbose and max_instances_count != -1:
            print('! Restricting the numebr of instances from each class to {}'.format(max_instances_count))
        
        if training_categories == testing_categories:
            print('\nTraining:Testing 80%:20%\n')
            for category in training_categories:
                if self.dataset_name == 'CICIDS' or self.dataset_name == 'CICIDS2':
                    temp = self.dataset_dictionary[category]
                elif self.dataset_name == 'kdd' or self.dataset_name == 'nsl-kdd':
                    temp = self.dataset_features[self.dataset[:, 42] == category , :]
                elif self.dataset_name == 'SCADA' or self.dataset_name == 'SCADA_Reduced':
                    temp = self.dataset[self.dataset[:, 12] == category, 0: 10]
                    
                temp_size = np.size(temp, axis = 0)
                if max_instances_count != -1:
                    temp_size = min(temp_size, max_instances_count)
                     
                self.dataset_all[category] = temp
                self.dataset_all_count[category] = np.size(temp, axis = 0)
                print(category)
                print(self.dataset_all_count[category])
             
        else:
            for training in training_categories:
                if self.dataset_name == 'kdd' or self.dataset_name == 'nsl-kdd':
                    temp = self.dataset_features[self.dataset[:, 42] == training , :]
                elif self.dataset_name == 'SCADA' or self.dataset_name == 'SCADA_Reduced':
                    temp = self.dataset[self.dataset[:, 12] == training , 0: 10]
                elif self.dataset_name == 'CICIDS' or self.dataset_name == 'CICIDS2':
                     temp = self.dataset_dictionary[training]


                temp_size = np.size(temp, axis = 0)
                if max_instances_count != -1:
                    temp_size = min(temp_size, max_instances_count)
                
                self.dataset_all[training] = temp
                self.dataset_all_count[training] = np.size(temp, axis = 0)
                
                self.training_dataset[training] = temp[0:temp_size, :]
                self.training_instances_count[training] = temp_size
            
            for testing in testing_categories:
                if self.dataset_name == 'kdd' or self.dataset_name == 'nsl-kdd':
                    temp = self.dataset_features[self.dataset[:, 42] == testing , :]
                elif self.dataset_name == 'SCADA' or self.dataset_name == 'SCADA_Reduced':
                    temp = self.dataset[self.dataset[:, 12] == testing , 0: 10]
                elif self.dataset_name == 'CICIDS'  or self.dataset_name == 'CICIDS2':
                    temp = self.dataset_dictionary[testing]
                     
                temp_size = np.size(temp, axis = 0)
                if max_instances_count != -1:
                    temp_size = min(temp_size, max_instances_count)      
                
                self.dataset_all[testing] = temp
                self.dataset_all_count[testing] = np.size(temp, axis = 0)
                
                self.testing_dataset[testing] = temp[0:temp_size, :]
                self.testing_instances_count[testing] = temp_size
        
        if self.dataset_name == 'kdd' or self.dataset_name == 'nsl-kdd': 
            self.number_of_features = np.size(self.dataset_features, axis = 1)
            del self.dataset
        elif self.dataset_name == 'SCADA' or self.dataset_name == 'SCADA_Reduced':     
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
            
    def generate_training_representitives_of_50_percent(self, number_of_reps, verbose = False):
        self.training_reps = {};
        self.number_of_reps = number_of_reps
        for category in self.training_categories:
            kmenas = KMeans(n_clusters=number_of_reps)
            kmenas.fit(self.dataset_all[category][:self.dataset_all_count[category]//2, :])
            self.training_reps[category] = kmenas.cluster_centers_
            
    def load_batch(self, batch_size, file_name):
        print(file_name)
        pairs = [np.zeros((batch_size, self.number_of_features)) for i in range(2)]
        targets = np.zeros((batch_size,))
        temp_file = pd.read_csv(file_name, header=None).values
        
        for i in range(batch_size):
            if np.size(temp_file, axis = 0) == i:
                print('break at {}'.format(i))
                break
            temp = temp_file[i, :]
            
            pairs[0][i, :] = self.dataset_all[temp[0].strip()][int(temp[1]), :].reshape(self.number_of_features)
            pairs[1][i, :] = self.dataset_all[temp[2].strip()][int(temp[3]), :].reshape(self.number_of_features)
            targets[i] = temp[0].strip() == temp[2].strip()
            
        return pairs, targets

    def append_to_confusion_matrix(self, cm, key1, key2):
        if key1 not in cm: 
            cm[key1] = 0
        if key2 not in cm: 
            cm[key2] = 0   
        cm[key1] += 1
        cm[key2] += 1

        return cm
    
    def write_accuracies(self, filename, accuracy_prob, accuracy_first_pair, accuracy_voting):
        with open(filename, "a") as file_writer:
            file_writer.write('accuracy probs ,' + str(accuracy_prob) + ',' + 'accuracy_with_one_pair,' +  str(accuracy_first_pair) + ',accuracy_voting,' + str(accuracy_voting) + "\n")

    def write_diff_pairs_accuracies(self, filename, accuracy_pairs):
        with open(filename, "a") as file_writer:
            file_writer.write('accuracy with differnt number of pairs\n')
            w = csv.DictWriter(file_writer, accuracy_pairs.keys())
            w.writeheader()
            w.writerow(accuracy_pairs)

    def write_confusion_matrix(self, filename, matrix, text):
        with open(filename, "a") as file_writer:
            file_writer.write(text+'\n')
            w = csv.DictWriter(file_writer, matrix.keys())
            w.writeheader()
            w.writerow(matrix)
            
    def evaluate_classisfication(self, file_name, model, testing_batch_size, no_of_classes, classes, output_file):
        print(no_of_classes)
        temp_file = pd.read_csv(file_name, header=None).values
        n_correct = 0
        n_correct_first_pair = 0
        n_correct_variable_pairs = {}
        n_correct_variable_pairs[5] = 0
        n_correct_variable_pairs[10] = 0
        n_correct_variable_pairs[15] = 0
        n_correct_variable_pairs[20] = 0
        n_correct_variable_pairs[25] = 0
        
        n_correct_voting = 0
        mis_classified_first_pair = {}
        
        mis_classified_voting_5 = {}
        mis_classified_voting_10 = {}
        mis_classified_voting_15 = {}
        mis_classified_voting_20 = {}
        mis_classified_voting_25 = {}
        mis_classified_voting = {}
        
        mis_classified_prob = {}
        
        for i in range(testing_batch_size):
            if np.size(temp_file, axis = 0) == i:
                print('break at {}'.format(i))
                testing_batch_size = i+1
                break
            
            votes = np.zeros((no_of_classes,1))
            
            temp_line = temp_file[i, :]
            probs = np.zeros((no_of_classes,1))
            test_pair = np.asarray([self.dataset_all[temp_line[0].strip()][int(temp_line[1]), :]]*no_of_classes).reshape(no_of_classes, self.number_of_features)

            for mm in range(30):
                temp = temp_line[2 + mm*(2*no_of_classes) :2 + (mm+1)*(2*no_of_classes)]
                
                support_set_1 = np.zeros((no_of_classes, self.number_of_features)) 
                
                targets = np.zeros((no_of_classes,))   
                for ci in range(no_of_classes):
                    support_set_1[ci,:] = self.dataset_all[temp[2*ci].strip()][int(temp[2*ci +1]), :]
                    targets[ci] = temp_line[0].strip() != temp[2*ci].strip()
                    
                modes_probs = model.predict([test_pair,support_set_1])

                votes[modes_probs == modes_probs[np.argmin(modes_probs)]] += 1
                probs += modes_probs

                if mm == 0:
                    if targets[np.argmin(probs)] == 0:
                        n_correct_first_pair+=1
                    key = temp_line[0].strip() + '_' + str(np.argmin(probs))
                    key_temp = temp_line[0].strip() + '_' + str(classes[np.argmin(probs)])
                    mis_classified_first_pair = self.append_to_confusion_matrix(mis_classified_first_pair, key, key_temp)
                    
                    
                if (mm + 1) in n_correct_variable_pairs:
                    if targets[np.argmax(votes)] == 0:
                        n_correct_variable_pairs[mm+1]+=1

                    key = temp_line[0].strip() + '_' + str(np.argmax(votes))
                    key_temp = temp_line[0].strip() + '_' + str(classes[np.argmax(votes)])
                    if (mm + 1) == 5:
                        mis_classified_voting_5 = self.append_to_confusion_matrix(mis_classified_voting_5, key, key_temp)
                    elif (mm + 1) == 10:
                        mis_classified_voting_10 = self.append_to_confusion_matrix(mis_classified_voting_10, key, key_temp)
                    elif (mm + 1) == 15:
                        mis_classified_voting_15 = self.append_to_confusion_matrix(mis_classified_voting_15, key, key_temp)
                    elif (mm + 1) == 20:
                        mis_classified_voting_20 = self.append_to_confusion_matrix(mis_classified_voting_20, key, key_temp)
                    elif (mm + 1) == 25:
                        mis_classified_voting_25 = self.append_to_confusion_matrix(mis_classified_voting_25, key, key_temp)
   
            probs/=30

            if targets[np.argmin(probs)] == 0:
                n_correct+=1

            key = temp_line[0].strip() + '_' + str(np.argmin(probs))
            key_temp = temp_line[0].strip() + '_' + str(classes[np.argmin(probs)])
            mis_classified_prob = self.append_to_confusion_matrix(mis_classified_prob, key, key_temp)                
                
            if targets[np.argmax(votes)] == 0:
                n_correct_voting += 1
            key = temp_line[0].strip() + '_' + str(np.argmax(votes))
            key_temp = temp_line[0].strip() + '_' + str(classes[np.argmax(votes)])
            mis_classified_voting = self.append_to_confusion_matrix(mis_classified_voting, key, key_temp)
                
                
        accuracy_pairs = {}
        for key in n_correct_variable_pairs:
            accuracy_pairs[key] = n_correct_variable_pairs[key]/testing_batch_size
        
        print(accuracy_pairs)
        accuracy = (100.0*n_correct / testing_batch_size)
        print("Got an accuracy of {}%".format(accuracy))
        
        accuracy_first_pair = (100.0*n_correct_first_pair / testing_batch_size)
        print("Got an accuracy of {}% using first pair".format(accuracy_first_pair))
        
        accuracy_voting = (100.0*n_correct_voting / testing_batch_size)
        
        
        self.write_accuracies(output_file, accuracy, accuracy_first_pair, accuracy_voting)
        self.write_diff_pairs_accuracies(output_file, accuracy_pairs)
        
        self.write_confusion_matrix(output_file, mis_classified_prob, 'misclassified using probs (30 pairs)')
        
        self.write_confusion_matrix(output_file, mis_classified_voting_5, 'misclassified using voting (5 pairs)')
        self.write_confusion_matrix(output_file, mis_classified_voting_10, 'misclassified using voting (10 pairs)')
        self.write_confusion_matrix(output_file, mis_classified_voting_15, 'misclassified using voting (15 pairs)')
        self.write_confusion_matrix(output_file, mis_classified_voting_20, 'misclassified using voting (20 pairs)')
        self.write_confusion_matrix(output_file, mis_classified_voting_25, 'misclassified using voting (25 pairs)')
        
        self.write_confusion_matrix(output_file, mis_classified_voting, 'misclassified using voting (30 pairs)')
        
        self.write_confusion_matrix(output_file, mis_classified_first_pair, 'misclassified using first pair')
        return accuracy, accuracy_first_pair, mis_classified_prob, accuracy_pairs, accuracy_voting, mis_classified_voting
    
    def evaluate_zero_day_new(self, file_name, model, testing_batch_size, no_of_classes, index_of_zero_day, training_classes, output_file):
        print(file_name)
        temp_file = pd.read_csv(file_name, header=None).values
#        output_file_org = output_file

        number_of_instances_arr = [1, 5, 10, 15, 20, 25, 30]
#        number_of_instances_arr = [30]
        for number_of_instances in number_of_instances_arr:
            with open(output_file, "a") as file_writer:
                file_writer.write('----------------------------------'+ '\n')
                file_writer.write('----------------------------------'+ '\n')                
                file_writer.write('NUMBER OF PAIRS FROM EACH CLASS = {}'.format(number_of_instances)+ '\n')
                
#            output_file = output_file_org.replace('.csv', '{}.csv'.format(number_of_instances))
            n_correct_voting = {}
            conf_matix = {}
            thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

            for th in thresholds:
                n_correct_voting[th] = 0
                conf_matix [th] = {}
                
            no_of_known_classes = no_of_classes-1
          
            for i in range(testing_batch_size):
                votes = {}
                for th in thresholds:
                    votes[th] = np.zeros((no_of_known_classes,1))
                    
                if np.size(temp_file, axis = 0) == i:
                    print('break at {}'.format(i))
                    testing_batch_size = i+1
                    break
                
                temp_line = temp_file[i, :]
                
                test_pair = np.asarray([self.dataset_all[temp_line[0].strip()][int(temp_line[1]), :]]*no_of_known_classes).reshape(no_of_known_classes, self.number_of_features)
    
                for mm in range(number_of_instances):
                    temp = temp_line[2 + mm*(2*no_of_classes) :2 + (mm+1)*(2*no_of_classes)]
                    
                    support_set_1 = np.zeros((no_of_known_classes, self.number_of_features)) 
                    
                    targets = np.zeros((no_of_known_classes,))   
                    index_ci = 0
                    for ci in range(no_of_classes):
                        if ci != index_of_zero_day:
                            support_set_1[index_ci,:] = self.dataset_all[temp[2*ci].strip()][int(temp[2*ci +1]), :]
                            targets[index_ci] = temp_line[0].strip() != temp[2*ci].strip()    
                            index_ci += 1
                            
                       
                    modes_probs = model.predict([test_pair,support_set_1])
                    for th in thresholds:
                        if modes_probs[np.argmin(modes_probs)] < th:
                            votes[th][modes_probs == modes_probs[np.argmin(modes_probs)]] += 1
                    
                for th in thresholds:
                    predicted_class = -1
                    if votes[th][np.argmax(votes[th])] >= (math.ceil(number_of_instances/3)):
                        predicted_class = np.argmax(votes[th])
                    
                    if (predicted_class == -1 and np.all(targets) == 1) or (predicted_class >= 0 and targets[predicted_class] == 0):
                        #correctly predicted
                        n_correct_voting[th] += 1
                    
                    key = temp_line[0].strip() + '_' + str(predicted_class)
                    key_temp = key
                    if predicted_class != -1:
                        key_temp = temp_line[0].strip() + '_' + str(training_classes[predicted_class])
                        
                    if key not in conf_matix[th]:
                        conf_matix[th][key] = 0
                    conf_matix[th][key] += 1
                    
                    if key_temp not in conf_matix[th]:
                        conf_matix[th][key_temp] = 0
                    conf_matix[th][key_temp] += 1
                    
            
            accuracy = {}
            for th in thresholds:
                accuracy[th] = n_correct_voting[th]/testing_batch_size
            
            with open(output_file, "a") as file_writer:
                file_writer.write('accuracy = ,' + str(accuracy)+ '\n')
                
            for th in thresholds:
                self.write_confusion_matrix(output_file, conf_matix[th], 'threshold = {}\n'.format(th))
                
            
        return accuracy, conf_matix
    
    
 