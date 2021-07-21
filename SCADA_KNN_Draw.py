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

random_state = 0
test_size = 0.5

path = '/home/hananhindy/Dropbox/SiameseNetworkDatasetFiles/DatasetProcessedFiles/SCADA_dataset_processed.csv'
dataset = pd.read_csv(path)
dataset = dataset.dropna()
dataset = dataset.values

original_classes = np.unique(dataset[:, 12])
m = []
for c in original_classes:
    if len(m) == 0:
        m = dataset[dataset[:,12] == c, :][0:1000,:]
    else:
        m = np.append(m, dataset[dataset[:,12] == c, :][0:1000:,:], axis = 0)

dataset = pd.DataFrame(m)

X = dataset.iloc[:, 0: 10].values
y = dataset.iloc[:, 12].values
    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
knn_classifier = KNeighborsClassifier(n_neighbors  = 5)
knn_classifier.fit(X_train, y_train)

classes = list(knn_classifier.classes_)

y_single_prob_correct_more_75 = np.zeros((14))
y_single_prob_correct_less_75 = np.zeros((14))
y_single_prob_incorrect_more_75 = np.zeros((14))
y_single_prob_incorrect_less_75 = np.zeros((14))
y_multiple_prob_including_correct = np.zeros((14))
y_multiple_prob_excluding_correct = np.zeros((14))

total = np.zeros((14)) 

classes_counts = {}
for i in range(14):
    classes_counts[classes[i]] = np.zeros((14))

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

#Plot how instances are divided
plt.clf()
p1 = plt.bar(ind, y_single_prob_correct_more_75, 0.35, color='#3288bd')
p2 = plt.bar(ind, y_single_prob_correct_less_75, 0.35, bottom=y_single_prob_correct_more_75, color = '#99d594')
p3 = plt.bar(ind, y_multiple_prob_including_correct, 0.35, bottom=y_single_prob_correct_more_75+y_single_prob_correct_less_75, color = '#e6f598')
p4 = plt.bar(ind, y_single_prob_incorrect_less_75, 0.35, bottom=y_single_prob_correct_more_75+y_single_prob_correct_less_75+y_multiple_prob_including_correct, color = '#fee08b')
p5 = plt.bar(ind, y_single_prob_incorrect_more_75, 0.35, bottom=y_single_prob_correct_more_75+y_single_prob_correct_less_75+y_multiple_prob_including_correct+y_single_prob_incorrect_less_75, color = '#fc8d59')
p6 = plt.bar(ind, y_multiple_prob_excluding_correct, 0.35, bottom=y_single_prob_correct_more_75+y_single_prob_correct_less_75+y_multiple_prob_including_correct+y_single_prob_incorrect_more_75+y_single_prob_incorrect_less_75, color = '#d53e4f')
plt.xticks(ind, classes, rotation=30)
plt.yticks(np.arange(0, 110, 10))
plt.xlabel('Class')
plt.ylabel('%')
plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]),
           ('ONE KNN Class - Correct - Prob >= 0.75', 'ONE KNN Class - Correct - Prob < 0.75', 
            'Multiple KNN Classes - Correct Included', 
            'ONE KNN Class - Incorrect - Prob < 0.75', 'ONE KNN Class - Incorrect - Prob >= 0.75', 
            'Multiple KNN Classes - Correct Not Included'), 
           loc='lower left', bbox_to_anchor= (0.0, 1.01),
          ncol=3, fancybox=True, shadow=True)

plt.title('SCADA KNN-based class predictions')
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
fig.savefig('SCADA_KNN_predictions.pdf', type='pdf',  bbox_inches='tight')

t = np.zeros((14))

plt.clf()
bar_col = ['#a6cee3','#1f78b4','#b2df8a', '#1a1a1a', '#67001f', '#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#ffff99','#b15928','#cab2d6','#6a3d9a']
for i in range(len(classes)):
    temp = []
    for j in range(len(classes)):
        if i == 0:
            if sum(classes_counts[classes[j]]) != 0:
                classes_counts[classes[j]] = classes_counts[classes[j]] * 100 / sum(classes_counts[classes[j]])
        temp.append(classes_counts[classes[j]][i])
        
    plt.bar(ind, temp, 0.35, bottom=t, color = bar_col[i])
    t += temp
    
plt.xticks(ind, classes, rotation=50)
plt.yticks([])
plt.legend(classes,
           loc='lower left', bbox_to_anchor= (0.0, 1.01),
          ncol=5, fancybox=True, shadow=True)
plt.xlabel('Class')
plt.title('SCADA KNN-based wrong classes distributions')

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
fig.savefig('SCADA_Wrong_neighbouring_classes_disctribution.pdf', type='pdf',  bbox_inches='tight')


plt.clf()
temp = []
for i in range(len(classes)):
    temp.append(np.count_nonzero(classes_counts[classes[i]]))

plt.bar(ind, temp, 0.35, color = '#3288bd')    
    
plt.xticks(ind, classes, rotation=50)
plt.yticks(np.arange(0, 15, 1))
plt.xlabel('Class')
plt.ylabel('Count of wrong classes that were equally neighbouring during testing')
plt.title('SCADA KNN-based number of confused classes')

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
plt.show()
fig.savefig('SCADA_Count_wrong_neighbouring_classes.pdf', type='pdf',  bbox_inches='tight')
