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
        elif dataset_name == 'STA':
           normal_path = path + '/11jun.10percentNormalno_syn.csv'
           dos_path = path + '/14jun.10percentAttackno_syn.csv'
           ddos_path = path + '/15jun.10percentAttackno_syn.csv'
            
           self.dataset_dictionary = {}  
           self.dataset_dictionary['normal'] = pd.read_csv(normal_path, header = None).values
           self.dataset_dictionary['dos'] = pd.read_csv(dos_path, header = None).values
           self.dataset_dictionary['ddos'] = pd.read_csv(ddos_path, header = None).values
        elif dataset_name == 'SCADA' or dataset_name == 'SCADA_Reduced':
           self.dataset = pd.read_csv(path)
           self.dataset = self.dataset.dropna().values
        elif dataset_name == 'CICIDS' or dataset_name == 'CICIDS2':
           normal_path = path + '/biflow_Monday-WorkingHours_Fixed.csv'
           hulk_path = path + '/new_biflow_Wednesday-WorkingHours_Hulk.csv'
           golden_path = path + '/new_biflow_Wednesday-WorkingHours_GoldenEye.csv'
           slowloris_path = path + '/new_biflow_Wednesday-WorkingHours_slowloris.csv'
           slowhttps_path = path + '/new_biflow_Wednesday-WorkingHours_Slowhttptest.csv'
           ddos_path = path +'/new_biflow_Friday-WorkingHours_DDoS.csv'
           FTP_path = path + '/new_biflow_Tuesday-WorkingHours_FTP.csv'
           heartbleed_path = path + '/new_biflow_Wednesday-WorkingHours_Heartbleed.csv'
           botnet_path = path + '/new_biflow_Friday-WorkingHours_Botnet.csv'
           portscan_path = path + '/new_biflow_Friday-WorkingHours_PortScan.csv'
           SSH_path = path + '/new_biflow_Tuesday-WorkingHours_SSH.csv'
           
           col_to_drop = ['Unnamed: 0', 'ip_src', 'ip_dst', 'num_src_flows', 'src_ip_dst_prt_delta']
#           col_to_drop = col_to_drop + ['bwd_num_pkts',
#             'bwd_mean_iat',
#             'fwd_std_iat',
#             'bwd_std_iat',
#             'fwd_max_iat',
#             'bwd_max_iat',
#             'bwd_mean_offset',
#             'fwd_max_pkt_len',
#             'fwd_num_bytes',
#             'bwd_num_bytes',
#             'bwd_num_psh_flags']
           
           self.dataset_dictionary = {}  
           self.dataset_dictionary['normal'] = pd.read_csv(normal_path).drop(col_to_drop, axis=1).values
           standard_scaler = preprocessing.StandardScaler()
           self.dataset_dictionary['normal'] = standard_scaler.fit_transform(self.dataset_dictionary['normal'])
           
#           self.dataset_dictionary['dos'] = pd.read_csv(hulk_path).append(pd.read_csv(golden_path)).append(pd.read_csv(slowloris_path)).append(pd.read_csv(slowhttps_path)).drop(col_to_drop, axis=1).values
#           self.dataset_dictionary['dos'] = standard_scaler.transform(self.dataset_dictionary['dos'])

          
           if dataset_name == 'CICIDS2':
               self.dataset_dictionary['heartbleed'] = pd.read_csv(heartbleed_path).drop(col_to_drop, axis=1).values
               self.dataset_dictionary['heartbleed'] = standard_scaler.transform(self.dataset_dictionary['heartbleed'])
               self.dataset_dictionary['ddos'] = pd.read_csv(ddos_path).drop(col_to_drop, axis=1).values
               self.dataset_dictionary['ddos'] = standard_scaler.transform(self.dataset_dictionary['ddos'])

#               self.dataset_dictionary['botnet'] = pd.read_csv(botnet_path).drop(col_to_drop, axis=1).values
#               self.dataset_dictionary['botnet'] = standard_scaler.transform(self.dataset_dictionary['botnet'])
    
               self.dataset_dictionary['portscan'] = pd.read_csv(portscan_path).drop(col_to_drop, axis=1).values
               self.dataset_dictionary['portscan'] = standard_scaler.transform(self.dataset_dictionary['portscan'])
           
           if dataset_name == 'CICIDS':
               self.dataset_dictionary['hulk'] = pd.read_csv(hulk_path).drop(col_to_drop, axis=1).values
               self.dataset_dictionary['hulk'] = standard_scaler.transform(self.dataset_dictionary['hulk'])
##
#           self.dataset_dictionary['golden'] = pd.read_csv(golden_path).drop(col_to_drop, axis=1).values
#           self.dataset_dictionary['golden'] = standard_scaler.transform(self.dataset_dictionary['golden'])
#
               self.dataset_dictionary['slowloris'] = pd.read_csv(slowloris_path).drop(col_to_drop, axis=1).values
               self.dataset_dictionary['slowloris'] = standard_scaler.transform(self.dataset_dictionary['slowloris'])
##
#           self.dataset_dictionary['slowhttps'] = pd.read_csv(slowhttps_path).drop(col_to_drop, axis=1).values
#           self.dataset_dictionary['slowhttps'] = standard_scaler.transform(self.dataset_dictionary['slowhttps'])

#           self.dataset_dictionary['ddos'] = pd.read_csv(ddos_path).drop(col_to_drop, axis=1).values
#           self.dataset_dictionary['ddos'] = standard_scaler.transform(self.dataset_dictionary['ddos'])
            
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
        elif self.dataset_name == 'STA' or self.dataset_name == 'CICIDS' or self.dataset_name == 'CICIDS2':
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
                if self.dataset_name == 'STA' or self.dataset_name == 'CICIDS' or self.dataset_name == 'CICIDS2':
                    temp = self.dataset_dictionary[category]
                elif self.dataset_name == 'kdd' or self.dataset_name == 'nsl-kdd':
                    temp = self.dataset_features[self.dataset[:, 42] == category , :]
                elif self.dataset_name == 'SCADA' or self.dataset_name == 'SCADA_Reduced':
                    temp = self.dataset[self.dataset[:, 12] == category, 0: 10]
                    
                temp_size = np.size(temp, axis = 0)
                if max_instances_count != -1:
                    temp_size = min(temp_size, max_instances_count)
                     
                testing_start_index = int((1 - 0.2 * (k_fold + 1)) * temp_size)
                testing_end_index = int(testing_start_index + (0.2 * temp_size))
                self.dataset_all[category] = temp
                self.dataset_all_count[category] = np.size(temp, axis = 0)
                print(category)
                print(self.dataset_all_count[category])
                #self.training_dataset[category] = np.append(temp[0:testing_start_index, :], temp[testing_end_index: temp_size, :], axis = 0)
                #self.testing_dataset[category] = temp[testing_start_index:testing_end_index, :]
                #self.training_instances_count[category] = np.size(self.training_dataset[category], axis = 0)
                #self.testing_instances_count[category] = np.size(self.testing_dataset[category], axis = 0)                
        else:
            if self.dataset_name == 'STA':
                print('ERROR! Cannot apply this to STA dataset')
                return
            
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
    
    
    def evaluate_zero_day_detection(self, file_name, model, testing_batch_size, zero_day_file):
        temp_file = pd.read_csv(file_name, header=None).values
        temo_zero_day_file = pd.read_csv(zero_day_file, header=None).values
        
        n_correct_not_training = {}
        n_correct_not_normal = {}
        n_corrent_labeled_any = 0
        n_corrent_labeled_all = 0
        n_corrent_labeled_one_pair = 0
        n_corrent_labeled_avg_5 = 0
        
        accuracy = {}
        accuracy_not_normal = {}
        
        thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        for th in thresholds:
            n_correct_not_training[th] = 0
            n_correct_not_normal[th] = 0
       
        mis_classified = {}
        
        for i in range(testing_batch_size):
            temp_line = temp_file[i, :]
            
            zero_day_line = temo_zero_day_file[i,:]
            
            test_pair = np.asarray([self.dataset_all[temp_line[0].strip()][int(temp_line[1]), :]]*5).reshape(5, self.number_of_features)
            support_set_zero_day = np.zeros((5, self.number_of_features)) 

            for k in range(30):
                support_set_zero_day[k,:] = self.dataset_all[temp_line[0].strip()][zero_day_line[k], :]
            
            zero_day_probs = model.predict([test_pair, support_set_zero_day])
            
            test_pair = np.asarray([self.dataset_all[temp_line[0].strip()][int(temp_line[1]), :]]*4).reshape(4, self.number_of_features)

            for mm in range(1):
                temp = temp_line[2 + mm*10 :10 + mm*10]
                
                support_set_1 = np.zeros((4, self.number_of_features)) 
                support_set_1[0,:] = self.dataset_all[temp[0].strip()][int(temp[1]), :]                
                support_set_1[1,:] = self.dataset_all[temp[2].strip()][int(temp[3]), :]                
                support_set_1[2,:] = self.dataset_all[temp[4].strip()][int(temp[5]), :]
                support_set_1[3,:] = self.dataset_all[temp[6].strip()][int(temp[7]), :]
                
                modes_probs = model.predict([test_pair,support_set_1])
                for th in thresholds:
                    if modes_probs[np.argmin(modes_probs)] > (th):
                        n_correct_not_training[th] = n_correct_not_training[th] + 1
                    if modes_probs[0] > (th):
                        n_correct_not_normal[th] = n_correct_not_normal[th] + 1
                        
                if modes_probs[np.argmin(modes_probs)] < 0.05:
                    key = str(np.argmin(modes_probs))
                    if key not in mis_classified:
                        mis_classified[key] = 0
                    
                    mis_classified[key] += 1
                
                if zero_day_probs[0] < modes_probs[np.argmin(modes_probs)]:
                    n_corrent_labeled_one_pair += 1
                
                if any(i < modes_probs[np.argmin(modes_probs)] for i in zero_day_probs):
                    n_corrent_labeled_any += 1
                    
                if all(i < modes_probs[np.argmin(modes_probs)] for i in zero_day_probs):
                    n_corrent_labeled_all += 1    
                 
                if np.average(zero_day_probs) < modes_probs[np.argmin(modes_probs)]:
                    n_corrent_labeled_avg_5 += 1
        
        for th in thresholds:
            accuracy[th] = (100.0*n_correct_not_training[th] / testing_batch_size)
            print("Got an accuracy of {}%".format(accuracy[th]))
          
            accuracy_not_normal[th] = (100.0*n_correct_not_normal[th] / testing_batch_size)
            print("Got an accuracy of {}%".format(accuracy_not_normal[th]))
        
        acc_labeled_one_pair = n_corrent_labeled_one_pair*100/testing_batch_size
        acc_labeled_any = n_corrent_labeled_any*100/testing_batch_size
        acc_labeled_all = n_corrent_labeled_all*100/testing_batch_size
        acc_labeled_average_5 = n_corrent_labeled_avg_5*100/testing_batch_size
        
        return accuracy, accuracy_not_normal, mis_classified, acc_labeled_one_pair, acc_labeled_any, acc_labeled_all, acc_labeled_average_5
        

            
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
    
    def get_validation_batch(self, batch_size, verbose):
        #"""Create batch of n pairs, half same class, half different class"""
        #randomly sample several classes to use in the batch
        n_classes = np.size(self.testing_categories)
        selected_classes = np.tile(rng.randint(low = 0, high = n_classes, size = int(batch_size/2)), 2)
    
        #initialize 2 empty arrays for the input image batch
        pairs = [np.zeros((batch_size, self.number_of_features)) for i in range(2)]
        
        #initialize vector for the targets, and make one half of it '1's, so 2nd half of batch has same class
        targets = np.zeros((batch_size,))
        targets[batch_size//2:] = 1
    
        for i in range(batch_size):
            current_class = self.testing_categories[selected_classes[i]]
            
            idx_1 = rng.randint(0, self.testing_instances_count[current_class])
            pairs[0][i, :] = self.testing_dataset[current_class][idx_1, :].reshape(self.number_of_features)
            
            if i >= batch_size // 2:
                class_2 = current_class  
            else: 
                #add a random number to the category modulo n classes to ensure 2nd image has
                # ..different category
                class_2 = self.testing_categories[(selected_classes[i] + rng.randint(1, n_classes)) % n_classes]
            
            idx_2 = rng.randint(0, self.testing_instances_count[class_2])
            pairs[1][i, :] = self.testing_dataset[class_2][idx_2, :].reshape(self.number_of_features)
    
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
            if probs[np.argmax(probs)] < 0.6:
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
            if probs[np.argmax(probs[0: sub_pair_count])] < 0.6:
                for th in thresholds:
                    if  probs[p.argmax(probs[0: sub_pair_count])] < th:
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
    
    
