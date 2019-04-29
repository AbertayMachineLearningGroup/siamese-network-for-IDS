#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:33:14 2019

@author: hananhindy
"""

import os
import pandas as pd
import numpy as np
import itertools
import random

total_number_of_pairs = 30000
number_of_classes_for_training = 4
comb_index = 3

path = '/home/hananhindy/Dropbox/SiameseNetworkDatasetFiles/DatasetProcessedFiles/kddcup.data_10_percent_corrected'
dataset = pd.read_csv(path, header=None)
dataset[41] = dataset[41].str[:-1]
dataset[42] = ''
dataset = dataset.values


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
    print('"{}" has {} instances'.
          format(key, np.size(dataset[dataset[:,41] == key, :], axis=0)))
        
    dataset[dataset[:, 41] == key, 42] = base_classes_map[key]

original_classes = ['normal', 'dos', 'probe', 'r2l', 'u2r']
classes = ['normal', 'dos', 'probe', 'r2l', 'u2r']

all_conbinations = list(itertools.combinations(original_classes, number_of_classes_for_training))
    
instances_count = {}

for c in classes:
    instances_count[c] = np.size(dataset[dataset[:,42] == c, :], axis=0)
    print('"{}" has {} instances'.
          format(c, instances_count[c]))

if number_of_classes_for_training != len(original_classes):
    classes = list(all_conbinations[comb_index])

#Similar Pairs classes
total_number_of_similar_pairs = total_number_of_pairs//2
total_number_of_dis_similar_pairs = total_number_of_pairs//2

total_number_of_similar_pair_per_class = total_number_of_similar_pairs//len(classes)
total_number_of_classification_pairs_per_class = total_number_of_pairs//len(classes)

training_file_name = 'kdd_{}_{}_Training_Pairs.csv'.format(total_number_of_pairs, comb_index)
validation_file_name = 'kdd_{}_{}_Validation_Pairs.csv'.format(total_number_of_pairs, comb_index)

if os.path.exists(training_file_name):
    os.remove(training_file_name)
  
if os.path.exists(validation_file_name):
    os.remove(validation_file_name)          

generate = ['t', 'v']
if number_of_classes_for_training != len(original_classes):
    generate = ['t']
    
for s in generate:
    for c in classes:
        if number_of_classes_for_training == len(original_classes):
            half_instances = instances_count[c]//2
        else:
            half_instances = instances_count[c]
            
        print('Hald instances of {} is {}'.format(c, half_instances))
        
        used_pairs = set()
        
        if s == 't':
            current_range = range(half_instances)
            file_name = training_file_name
        else:
            current_range = range(half_instances, instances_count[c])
            file_name = validation_file_name
        
        numbers = list(current_range)
        
        if c != 'u2r':
            for i in range(total_number_of_similar_pair_per_class):
                while True:
                    pair = random.sample(numbers, 2)
                    # Avoid generating both (1, 2) and (2, 1)
                    pair = tuple(sorted(pair))
                    if pair not in used_pairs:
                        used_pairs.add(pair)
                        with open(file_name, "a") as file_writer:
                            file_writer.write("{}, {}, {}, {}\n".format(c, pair[0],c, pair[1]))
                        break
        else:
            print('repeating pairs')
            all_pairs = list(itertools.combinations(numbers, 2))
            j = 0
            for i in range(total_number_of_similar_pair_per_class):
                with open(file_name, "a") as file_writer:
                    file_writer.write("{}, {}, {}, {}\n".format(c, all_pairs[j][0], c, all_pairs[j][1]))
                j = j + 1
                if j == len(all_pairs):
                    j = 0
                
all_conbinations = list(itertools.combinations(classes, 2))
total_number_dis_similar_per_combination = total_number_of_dis_similar_pairs//len(all_conbinations)
for s in generate:
    for comb in all_conbinations:
        if number_of_classes_for_training == len(original_classes):
            half_instances_c1 = instances_count[comb[0]]//2
            half_instances_c2 = instances_count[comb[1]]//2        
        else:
            half_instances_c1 = instances_count[comb[0]]
            half_instances_c2 = instances_count[comb[1]]
        
    
        print('Half instances of {} is {}'.format(comb[0], half_instances_c1))
        print('Half instances of {} is {}'.format(comb[1], half_instances_c2))
        
        if s == 't':
            current_range_c1 = range(half_instances_c1)
            current_range_c2 = range(half_instances_c2)
            file_name = 'kdd_{}_{}_Training_Pairs.csv'.format(total_number_of_pairs, comb_index)
        else:
            current_range_c1 = range(half_instances_c1, instances_count[comb[0]])
            current_range_c2 = range(half_instances_c2, instances_count[comb[1]])
            file_name = 'kdd_{}_{}_Validation_Pairs.csv'.format(total_number_of_pairs, comb_index)

            
        numbers_c1 = list(current_range_c1)
        numbers_c2 = list(current_range_c2)
        
        used_pairs = set()
        for i in range(total_number_dis_similar_per_combination):
            while True:
                pair = tuple([random.sample(numbers_c1, 1)[0], random.sample(numbers_c2, 1)[0]])
                if pair not in used_pairs:
                    used_pairs.add(pair)
                    with open(file_name, "a") as file_writer:
                        file_writer.write("{}, {}, {}, {}\n".format(comb[0], pair[0], comb[1], pair[1]))
                    break


if number_of_classes_for_training == len(original_classes):
    #generate classification testing instances
    list_c1 = list(range(instances_count[classes[1]]//2, instances_count[classes[1]]))
    list_c0 = list(range(instances_count[c]//2, instances_count[classes[0]]))
    list_c2 = list(range(instances_count[classes[2]]//2, instances_count[classes[2]]))
    list_c3 = list(range(instances_count[classes[3]]//2, instances_count[classes[3]]))
    list_c4 = list(range(instances_count[classes[4]]//2, instances_count[classes[4]]))
    file_name = 'kdd_{}_{}_Classification_Pairs.csv'.format(total_number_of_pairs, comb_index)
    if os.path.exists(file_name):
        os.remove(file_name)   
        
    for c in classes:
        for i in range(total_number_of_classification_pairs_per_class):
            used_pairs = set()
            pair_temp = []
            
            while True:
                pair_temp.append(random.sample(range(instances_count[c]//2, instances_count[c]), 1)[0])
                for mm in range(30):
                    pair_temp.append(random.sample(list_c0,1)[0])
                    pair_temp.append(random.sample(list_c1,1)[0])
                    pair_temp.append(random.sample(list_c2,1)[0])
                    pair_temp.append(random.sample(list_c3,1)[0])
                    pair_temp.append(random.sample(list_c4,1)[0])
                    
                pairs = tuple(pair_temp)
#                pairs = tuple([random.sample(range(instances_count[c]//2, instances_count[c]), 1)[0], 
#                               random.sample(list_c0,1)[0], 
#                               random.sample(list_c1,1)[0], 
#                               random.sample(list_c2,1)[0],
#                               random.sample(list_c3,1)[0], 
#                               random.sample(list_c4,1)[0]])
          
                if pairs not in used_pairs:
                    used_pairs.add(pairs)
                    str_pair = c + ',' + str(pairs[0]) + ','
                    for mm in range(30):
                        for mmm in range(5):
                            str_pair += classes[mmm] + ',' + str(pairs[1 + mm*5+mmm]) + ','
                    with open(file_name, "a") as file_writer:
#                        file_writer.write("{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(c, pairs[0],
#                                                                    classes[0], pairs[1],
#                                                                    classes[1], pairs[2],
#                                                                    classes[2], pairs[3],
#                                                                    classes[3], pairs[4],
#                                                                    classes[4], pairs[5]
#                                                                    ))
                        file_writer.write("{}\n".format(str_pair))
                    break                   
        
            
else:
    list_c0 = list(range(instances_count[classes[0]]))
    list_c1 = list(range(instances_count[classes[1]]))
    list_c2 = list(range(instances_count[classes[2]]))
    list_c3 = list(range(instances_count[classes[3]]))
    
    total_number_of_zero_day_pairs_per_class = total_number_of_pairs//(len(original_classes) - len(classes))
    file_name = 'kdd_{}_{}_Zero_Day_Pairs.csv'.format(total_number_of_pairs, comb_index)
    testing_classes = [item for item in original_classes if item not in classes]
    for c in testing_classes:
        for i in range(total_number_of_zero_day_pairs_per_class):
            used_pairs = set()
            pair_temp = []
            
            while True:
                pair_temp.append(random.sample(range(instances_count[c]//2), 1)[0])
                for mm in range(30):
                    pair_temp.append(random.sample(list_c0,1)[0])
                    pair_temp.append(random.sample(list_c1,1)[0])
                    pair_temp.append(random.sample(list_c2,1)[0])
                    pair_temp.append(random.sample(list_c3,1)[0])
                    
                pairs = tuple(pair_temp)           
#            while True:
#                pairs = tuple([random.sample(range(instances_count[c]), 1)[0], 
#                               random.sample(list_c0,1)[0], 
#                               random.sample(list_c1,1)[0], 
#                               random.sample(list_c2,1)[0]])
          
                if pairs not in used_pairs:
                    used_pairs.add(pairs)
                    str_pair = c + ',' + str(pairs[0]) + ','
                    for mm in range(30):
                        for mmm in range(number_of_classes_for_training):
                            str_pair += classes[mmm] + ',' + str(pairs[1 + mm*number_of_classes_for_training+mmm]) + ','
                    with open(file_name, "a") as file_writer:
                        file_writer.write("{}\n".format(str_pair))
#                        file_writer.write("{}, {}, {}, {}, {}, {}, {}, {}\n".format(c, pairs[0],
#                                                                    classes[0], pairs[1],
#                                                                    classes[1], pairs[2],
#                                                                    classes[2], pairs[3]
#                                                                    ))
                        
                    str_pair_2 = ','.join(map(str,random.sample(range(instances_count[c]//2, instances_count[c]), 5)))
                    
                    with open('Same_Class' + file_name, "a") as file_writer:
                        file_writer.write("{}\n".format(str_pair_2))    
                    break
                    