#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:36:48 2019

@author: hananhindy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import matplotlib.ticker as ticker

def add_kdd_main_classes(dataset):
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
    
    return dataset

dataset_name = 'SCADA'

random_state = 0
test_size = 0.5

if dataset_name == 'kdd' or dataset_name == 'nsl-kdd':
    if dataset_name == 'kdd':
        path = '/home/hananhindy/Dropbox/SiameseNetworkDatasetFiles/DatasetProcessedFiles/kddcup.data_10_percent_corrected'
    else:
        path = '/home/hananhindy/Dropbox/SiameseNetworkDatasetFiles/DatasetProcessedFiles/KDDTrain+.txt'
        
    dataset = pd.read_csv(path, header=None)
    
    if dataset_name == 'kdd':  
        dataset[41] = dataset[41].str[:-1]
        
    dataset[42] = ''
    dataset = dataset.values
    dataset = add_kdd_main_classes(dataset)
    label_encoder_1 = LabelEncoder()
    label_encoder_2 = LabelEncoder()
    label_encoder_3 = LabelEncoder()
    one_hot_encoder = OneHotEncoder(categorical_features = [1,2,3])
        
    dataset[:, 1] = label_encoder_1.fit_transform(dataset[:, 1])
    dataset[:, 2] = label_encoder_2.fit_transform(dataset[:, 2])        
    dataset[:, 3] = label_encoder_3.fit_transform(dataset[:, 3])
    X = one_hot_encoder.fit_transform(dataset[:, :-2]).toarray()
            
    y = dataset[:, 42]
elif dataset_name == 'SCADA':
    path = '/home/hananhindy/Dropbox/SiameseNetworkDatasetFiles/DatasetProcessedFiles/SCADA_dataset_processed.csv'
    dataset = pd.read_csv(path)
    dataset = dataset.dropna()
    
    X = dataset.iloc[:, 0: 10].values
    y = dataset.iloc[:, 12].values
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
knn_classifier = KNeighborsClassifier(n_neighbors  = 5)
knn_classifier.fit(X_train, y_train)

classes = list(knn_classifier.classes_)

no_of_classes = len(classes)

y_single_prob_correct_more_75 = np.zeros((no_of_classes))
y_single_prob_correct_less_75 = np.zeros((no_of_classes))
y_single_prob_incorrect_more_75 = np.zeros((no_of_classes))
y_single_prob_incorrect_less_75 = np.zeros((no_of_classes))
y_multiple_prob_including_correct = np.zeros((no_of_classes))
y_multiple_prob_excluding_correct = np.zeros((no_of_classes))

total = np.zeros((no_of_classes)) 

classes_counts = {}
for i in range(no_of_classes):
    classes_counts[classes[i]] = np.zeros((no_of_classes))

for i in range(np.size(X_test, axis = 0)):
    index_of_test_class = classes.index(y_test[i])
    total[index_of_test_class] += 1
    
    probs = list(knn_classifier.predict_proba(X_test[i,:].reshape(1, -1))[0,:])
        
    if list(probs == probs[np.argmax(probs)]).count(True) == 1:
        #Unique predicted class
        if classes[np.argmax(probs)] == y_test[i]:
            if  probs[np.argmax(probs)] >= 0.75:
                y_single_prob_correct_more_75[index_of_test_class] += 1
            else:
                y_single_prob_correct_less_75[index_of_test_class] += 1
        else:
            classes_counts[y_test[i]][probs==probs[np.argmax(probs)]] += 1
            if probs[np.argmax(probs)] >= 0.75:
                y_single_prob_incorrect_more_75[index_of_test_class] += 1
            else:
                y_single_prob_incorrect_less_75[index_of_test_class] += 1     
    else:
        # Multiple predictions
        if probs[index_of_test_class] == probs[np.argmax(probs)]:
            # Correct class in predictions
            y_multiple_prob_including_correct[index_of_test_class] += 1
        else:
            classes_counts[y_test[i]][probs == probs[np.argmax(probs)]] += (1/probs.count(probs[np.argmax(probs)]))
            y_multiple_prob_excluding_correct[index_of_test_class] += 1
        
            
ind = np.arange(len(classes)) 

y_single_prob_correct_more_75 = y_single_prob_correct_more_75 * 100 / total
y_single_prob_correct_less_75 = y_single_prob_correct_less_75 * 100 / total

y_single_prob_incorrect_more_75 = y_single_prob_incorrect_more_75 * 100 / total
y_single_prob_incorrect_less_75 = y_single_prob_incorrect_less_75 * 100 / total

y_multiple_prob_including_correct = y_multiple_prob_including_correct * 100 / total
y_multiple_prob_excluding_correct = y_multiple_prob_excluding_correct * 100 / total

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 14}

plt.rc('font', **font)

bar_width = 0.5

#Plot how instances are divided
plt.clf()
p1 = plt.bar(ind, y_single_prob_correct_more_75, bar_width, color='#3288bd')
p2 = plt.bar(ind, y_single_prob_correct_less_75, bar_width, bottom=y_single_prob_correct_more_75, color = '#99d594')
p3 = plt.bar(ind, y_multiple_prob_including_correct, bar_width, bottom=y_single_prob_correct_more_75+y_single_prob_correct_less_75, color = '#e6f598')
p4 = plt.bar(ind, y_single_prob_incorrect_less_75, bar_width, bottom=y_single_prob_correct_more_75+y_single_prob_correct_less_75+y_multiple_prob_including_correct, color = '#fee08b')
p5 = plt.bar(ind, y_single_prob_incorrect_more_75, bar_width, bottom=y_single_prob_correct_more_75+y_single_prob_correct_less_75+y_multiple_prob_including_correct+y_single_prob_incorrect_less_75, color = '#fc8d59')
p6 = plt.bar(ind, y_multiple_prob_excluding_correct, bar_width, bottom=y_single_prob_correct_more_75+y_single_prob_correct_less_75+y_multiple_prob_including_correct+y_single_prob_incorrect_more_75+y_single_prob_incorrect_less_75, color = '#d53e4f')
plt.xticks(ind, classes, rotation=90)
plt.yticks(np.arange(0, 110, 10))
plt.xlabel('Class', font)
plt.ylabel('%', font)

plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]),
           ('ONE KNN Class - Correct - Prob >= 0.75', 'ONE KNN Class - Correct - Prob < 0.75', 
            'Multiple KNN Classes - Correct Included', 
            'ONE KNN Class - Incorrect - Prob < 0.75', 'ONE KNN Class - Incorrect - Prob >= 0.75', 
            'Multiple KNN Classes - Correct Not Included'), 
           loc='lower left', bbox_to_anchor= (0.0, 1.01),
          ncol=3, fancybox=True, shadow=True)

#plt.title(dataset_name + ' KNN-based class predictions')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
fig.savefig(dataset_name + '_KNN_predictions.pdf', type='pdf',  bbox_inches='tight')

t = np.zeros((no_of_classes))

plt.clf()
bar_col = ['#a6cee3','#1f78b4','#b2df8a', '#1a1a1a', '#67001f', '#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#ffff99','#b15928','#cab2d6','#6a3d9a']
for i in range(len(classes)):
    temp = []
    for j in range(len(classes)):
        if i == 0:
            if sum(classes_counts[classes[j]]) != 0:
                classes_counts[classes[j]] = classes_counts[classes[j]] * 100 / sum(classes_counts[classes[j]])
        temp.append(classes_counts[classes[j]][i])
        
    plt.bar(ind, temp, bar_width, bottom=t, color = bar_col[i])
    t += temp
    
plt.xticks(ind, classes, rotation=90)
plt.yticks([])
plt.legend(classes,
           loc='lower left', bbox_to_anchor= (0.0, 1.01),
          ncol=5, fancybox=True, shadow=True)
plt.xlabel('Class', font)
#plt.title(dataset_name + ' KNN-based wrong classes distributions')

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
fig.savefig(dataset_name + '_Wrong_neighbouring_classes_disctribution.pdf', type='pdf',  bbox_inches='tight')


plt.clf()
temp = []
for i in range(len(classes)):
    temp.append(np.count_nonzero(classes_counts[classes[i]]))

plt.bar(ind, temp, bar_width, color = '#3288bd')    
    
plt.xticks(ind, classes, rotation=90)
plt.yticks(np.arange(0, no_of_classes + 1, 1))
plt.xlabel('Class', font)
plt.ylabel('Count of wrong classes that were equally neighbouring during testing', font)
#plt.title(dataset_name + ' KNN-based number of confused classes')

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
fig.savefig(dataset_name + '_Count_wrong_neighbouring_classes.pdf', type='pdf',  bbox_inches='tight')
