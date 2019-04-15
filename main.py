#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 12:19:17 2018

@author: hananhindy
"""
import itertools
from dataset_processor import DatasetHandler
from siamese_net import SiameseNet
from args_handler import Arguments
import csv

if __name__ == "__main__":
    args = Arguments()
    args.parse()
    
    dataset_handler = DatasetHandler(args.path, args.dataset_name, args.verbose)
    
    all_classes = list(dataset_handler.get_classes())
    if args.verbose:
        print('\nClasses are:\n{}'.format(all_classes))
    
    if args.train_with_all:
        training_categories = all_classes
        testing_categories = all_classes
    else:
        all_conbinations = list(itertools.combinations(all_classes, args.number_of_training_categories))
    
        training_categories = all_conbinations[args.comb_index]
        testing_categories =  list(set(all_classes) - set(training_categories))

    dataset_handler.encode_split(training_categories, testing_categories, args.max_from_class, args.k_fold_number, args.verbose)
    
    if args.number_of_reps > 0:
        dataset_handler.generate_training_representitives(args.number_of_reps, args.verbose)
    
    if args.verbose:
        print('!Starting!')
    
    with open(args.output_file_name, "a") as file_writer:
        file_writer.write("Dataset, {}\n".format(args.dataset_name))
        file_writer.write("Network ID, {}\n".format(args.network_id))
        file_writer.write("Max from class, {}\n".format(args.max_from_class))
        file_writer.write("Training Batch:Testing Batch, {}:{}\n".format(args.batch_size, args.testing_batch_size))
        file_writer.write("No of iterations, {}\n".format(args.niterations))
        file_writer.write("k =, {}\n".format(args.k_fold_number))
        file_writer.write("n_reps =, {}, reps_from_all = {}\n".format(args.number_of_reps, args.reps_from_all))
        file_writer.write("acc_not_in_training, acc_added_labels\n")
        file_writer.write(", ".join(training_categories) + "\n")
    
    for run in range(args.nruns):
        if args.print_loss:
            loss_array = []
            
        if args.verbose:
            print("Run #{}".format(run))
            
        best_accuracy = -1
        best_accuracy_partial = -1
        best_accuracy_labels = -1
        
        best_threashold_acc = {}
        
        wrapper = SiameseNet((dataset_handler.number_of_features,), args.network_id, args.verbose)
        
        for i in range(1, args.niterations + 1):
            (inputs, targets) = dataset_handler.get_batch(args.batch_size, args.verbose)
            
            loss = wrapper.siamese_net.train_on_batch(inputs,targets)
            
            if i % args.loss_every == 0:
                if args.print_loss:
                    loss_array.append(loss) 
                if args.verbose:
                    print(loss)
                
            if i % args.evaluate_every == 0:
                if args.is_add_labels:
                    val_acc_threasholds, val_acc_labels =  dataset_handler.test_oneshot_adding_labels(wrapper.siamese_net, args.testing_batch_size, args.reps_from_all, args.verbose)
                    val_acc = val_acc_threasholds[60]
                    
                elif args.test_vs_all:
                    val_acc, val_acc_partial = dataset_handler.test_oneshot_new_classes_vs_all(wrapper.siamese_net, args.testing_batch_size, args.verbose)
                else:
                    testing_validation_windows = len(dataset_handler.testing_categories)
                    val_acc = dataset_handler.test_oneshot(wrapper.siamese_net, args.testing_batch_size, testing_validation_windows, args.train_with_all, args.verbose)   
                
                if args.is_add_labels:
                    if val_acc >= best_accuracy and val_acc_labels >= best_accuracy_labels:
                        best_accuracy = val_acc
                        best_accuracy_labels = val_acc_labels
                        
                        try:    
                            best_threashold_acc = val_acc_threasholds
                            best_accuracy_labels = val_acc_labels
                        except NameError:
                            print('Name Error 1')
                else:
                    if val_acc >= best_accuracy:
                        best_accuracy = val_acc
                        if args.save_best:
                            wrapper.siamese_net.save('Best_Acc{}_{}'.format(i, val_acc))
                            
                    if args.test_vs_all and val_acc_partial >= best_accuracy_partial:
                        best_accuracy_partial = val_acc_partial
    
        with open(args.output_file_name, "a") as file_writer:
            if args.is_add_labels:
                try:    
                    w = csv.DictWriter(file_writer, best_threashold_acc.keys())
                    w.writeheader()
                    w.writerow(best_threashold_acc)
                    file_writer.write('Labels, ' + str(best_accuracy_labels) + '\n\n')
                except NameError:
                    print('Name Error 2')
            else:
                file_writer.write(str(best_accuracy) + ',' + str(best_accuracy_partial) + "\n")

            if args.print_loss:
                file_writer.write(",".join(map(str,loss_array)) + "\n")
