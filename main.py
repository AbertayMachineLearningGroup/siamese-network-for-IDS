#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:19:17 2018

@author: hananhindy
"""
import argparse
import itertools
from dataset_processor import DatasetHandler
from siamese_net import SiameseNet

# Helper Functions
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# End Helper Functions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type = str2bool, help = 'If true, prints will be displayed')
    parser.add_argument('--path', help = 'Path of the dataset csv or files directory')
    parser.add_argument('--network_id', help = 'Siamese Network ID ')
    parser.add_argument('--batch_size', help = 'Number of randomly generated pairs for training')
    parser.add_argument('--testing_batch_size', help = 'Number of randomly generated pairs for testing')
    parser.add_argument('--nruns', help = 'Number of independent runs')
    parser.add_argument('--niterations', help = 'Number of training iterations')

    parser.add_argument('--evaluate_every', help = 'Interval for testing')
    parser.add_argument('--max_from_class', help = 'Max count of instances to use from each class')

    parser.add_argument('--comb_index', help = 'combinaion index')

    parser.add_argument('--train_with_all', type=str2bool, help='bool train and test with all classes')
    parser.add_argument('--test_vs_all', type=str2bool, help = '')
    parser.add_argument('--save_best', type=str2bool, help = 'Save the best accuracy model')
    parser.add_argument('--dataset_name', help = 'Specify the dataset name ')

    # Defaults 
    evaluate_every = 10      # interval for evaluating 
    loss_every = 50         # interval for printing loss (iterations)
    batch_size = 250
    testing_batch_size = 250    #how mahy tasks to validate on?
    niterations = 1000 
    nruns = 10
    test_vs_all = False
    verbose = True
    current_combination_index = 0
    train_with_all = True
    save_best = False
    
    N_way = 2 # how many classes for testing one-shot tasks>

    dataset_name = 'kdd'
    
    # End Defaults
    
    args = parser.parse_args()  

    if args.dataset_name != None:
        dataset_name  = args.dataset_name
        
    if dataset_name == 'kdd':
        path = '/home/hananhindy/Downloads/kddcup.data_10_percent_corrected'
        network_id = 'kdd_0'
        max_from_class = 1200
    elif dataset_name == 'STA':
        path = '/media/hananhindy/MyFiles/GitHub/phd/STA2018_DatasetPreprocessed/'
        network_id = 'STA_0'
        max_from_class = 642
        
    if args.path != None:
        path = args.path
    
    if args.network_id != None:
        network_id = args.network_id
        
    if args.verbose != None:
        verbose = args.verbose        
        
    if args.comb_index != None:
        current_combination_index = int(args.comb_index)
    
    if args.max_from_class != None:
        max_from_class = int(args.max_from_class)     
    
    if args.train_with_all != None:
        train_with_all = args.train_with_all
    
    if args.evaluate_every != None:
        evaluate_every = int(args.evaluate_every)
    
    if args.batch_size != None:
        batch_size = int(batch_size)
    
    if args.testing_batch_size != None:
        testing_batch_size = int(args.testing_batch_size)
        
    if args.niterations != None:
        niterations = int(args.niterations)
    
    if args.nruns != None:
        nruns = int(args.nruns)
    
    if args.test_vs_all != None:
        test_vs_all = args.test_vs_all
        
    if args.save_best != None:
        save_best = args.save_best
    

    dataset_handler = DatasetHandler(path, dataset_name, verbose)
    all_classes = list(dataset_handler.get_classes())
    
    if verbose:
        print('\bClasses are:\n{}'.format(all_classes))
    
    if train_with_all:
        training_categories = all_classes
        testing_categories = all_classes
    else:
        number_of_training = 3
        all_conbinations = list(itertools.combinations(all_classes, number_of_training))
    
        training_categories = all_conbinations[current_combination_index]
        testing_categories =  list(set(all_classes) - set(training_categories))

    dataset_handler.encode_split(training_categories, testing_categories, max_from_class, verbose)
    
    if verbose:
        print('!Starting!')
    
    with open('Results.csv', "a") as file_writer:
        file_writer.write(", ".join(training_categories) + "\n")
            
    for run in range(nruns):
        if verbose:
            print("Run {}".format(run))
            
        best_accuracy = -1
        best_accuracy_partial = -1
        
        wrapper = SiameseNet((dataset_handler.number_of_features,), network_id)
        
        for i in range(1, niterations + 1):
            (inputs, targets) = dataset_handler.get_batch(batch_size, verbose)
            
            loss = wrapper.siamese_net.train_on_batch(inputs,targets)
            if verbose and i % loss_every == 0:
                print(loss)
                
            if i % evaluate_every == 0:
                if test_vs_all:
                    val_acc, val_acc_partial = dataset_handler.test_oneshot_new_classes_vs_all(wrapper.siamese_net, testing_batch_size, verbose)
                else:
                    val_acc = dataset_handler.test_oneshot(wrapper.siamese_net, testing_batch_size, len(dataset_handler.testing_categories), verbose)   
                
                if val_acc >= best_accuracy:
                    if save_best:
                        wrapper.siamese_net.save('Best_Acc{}_{}'.format(i, val_acc))
                    best_accuracy = val_acc
                if test_vs_all and val_acc_partial >= best_accuracy_partial:
                    best_accuracy_partial = val_acc_partial
        
        with open('Results.csv', "a") as file_writer:
            file_writer.write(str(best_accuracy) + ',' + str(best_accuracy_partial) + "\n")
                                
