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
import matplotlib.pyplot as plt
import numpy as np
import time
import os

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
    #networks = ['kdd__2_wd_dropout_2____', 'kdd_1', 'kdd__2_wd_dropout_3', 'kdd_1', 'kdd__2_wd__', 'kdd__2_wd_dropout_2__', 'kdd__2_wd_dropout_3__','kdd__3_wd', 'kdd__4_wd', 'kdd__2_dropout', 'kdd__3_dropout', 'kdd__4_dropout', 'kdd__4_BN_dropout', 'kdd__3_BN_dropout', 'kdd__2_BN_dropout', 'kdd__2_BN', 'kdd__3_BN', 'kdd__4_BN', 'kdd__2_wd_dropout', 'kdd__3_wd_dropout', 'kdd__4_wd_dropout']
    for run in range(args.nruns):
#        with open(args.output_file_name, "a") as file_writer:
#            file_writer.write("Network ID, {}\n".format(args.network_id))        
#        
        #args.network_id = networks[run]
        print(args.network_id)
        
        if args.print_loss:
            loss_array = []
        
        if args.verbose:
            print("Run #{}".format(run))
            
#        best_accuracy = -1
#        best_accuracy_partial = -1
#        best_accuracy_labels = -1
#        it = 0
#        best_threashold_acc = {}
#        
        #dataset_handler.generate_training_representitives_of_50_percent(1)
        wrapper = SiameseNet((dataset_handler.number_of_features,), args.network_id, args.verbose)
        #(inputs1, targets1) = dataset_handler.get_batch(args.batch_size, args.verbose)
        #(inputs_val, targets_val) = dataset_handler.get_validation_batch(args.validation_batch_size, args.verbose)
        #print(args.comb_index)
        (inputs1, targets1) = dataset_handler.load_batch(args.batch_size, '/home/hananhindy/kdd_30000_{}_Training_Pairs.csv'.format(args.comb_index))
        if args.train_with_all:
            (inputs_val, targets_val) = dataset_handler.load_batch(args.batch_size, '/home/hananhindy/kdd_30000_{}_Validation_Pairs.csv'.format(args.comb_index))
        
        #min_validation_loss = 1000
        #max_validation_loss = 0
        loss = np.zeros(args.niterations)
        validation_loss = np.zeros(args.niterations)
        for i in range(1, args.niterations + 1):
            #print('{}'.format(i))
            if args.use_sub_batches:
                 hist = wrapper.siamese_net.fit(inputs1,targets1, batch_size = 3000, validation_data= (inputs_val,targets_val)).history
                 loss[i-1] = hist['loss'][0]
            else:
                 loss[i-1] = wrapper.siamese_net.train_on_batch(inputs1,targets1)
            if args.train_with_all:
                if args.use_sub_batches:
                    validation_loss[i-1] = hist['val_loss'][0]
                else:
                    validation_loss[i-1] = wrapper.siamese_net.test_on_batch(inputs_val, targets_val)
           
            
            
#            if validation_loss[i-1] < min_validation_loss:
#                min_validation_loss = validation_loss[i-1]
#            if validation_loss[i-1] > max_validation_loss and i > 1000:
#                print(i)
#                break
#            if validation_loss[i-1] > max_validation_loss:
#                max_validation_loss = validation_loss[i-1]
#            
            print('{} -> Loss = {} validation loss = {}'.format(i, loss[i-1], validation_loss[i-1]))
        
            if i >= 500 and i%500 == 0:
                if args.train_with_all:
                    plt.clf()
                    training_plot, = plt.plot(loss[0:i])
                    val_plot, = plt.plot(validation_loss[0:i])
                    plt.xlabel('Iteration no')
                    plt.ylabel('Loss')
                    plt.legend([training_plot, val_plot], ['Training Loss', 'Validation Loss'])
                    
                    plt.savefig('loss_0.00006_Run_{}.png'.format(args.network_id))
                    accuracy1, accuracy_first_pair, mis_classified, accuracy_pairs, accuracy_voting = dataset_handler.evaluate_classisfication('/home/hananhindy/kdd_30000_{}_Classification_Pairs.csv'.format(args.comb_index),
                                                 wrapper.siamese_net, 
                                                 30000)
                else:
                    accuracy1, accuracy_not_normal, mis_classified, labeled_one, labeled_any, labeled_all , lablled_avg = dataset_handler.evaluate_zero_day_detection('/home/hananhindy/kdd_30000_{}_Zero_Day_Pairs.csv'.format(args.comb_index),
                                                     wrapper.siamese_net, 
                                                     30000,
                                                     '/home/hananhindy/Same_Classkdd_30000_{}_Zero_Day_Pairs.csv'.format(args.comb_index))
                    #accuracy2 = -1
                    #accuracy_k_menas = -1
                    #accuracy2 = dataset_handler.evaluate_classisfication('/home/hananhindy/Dropbox/SiameseNetworkDatasetFiles/DatasetProcessedFiles/2_kdd_150000_Classification_Pairs.csv',
                    #                                 wrapper.siamese_net, 
                    #                                 150000)
                    
                with open(args.output_file_name, "a") as file_writer:
                    if args.train_with_all:
                        file_writer.write('accuracy ,' + str(accuracy1) + ',' + 'accuracy_with_one_pair,' +  str(accuracy_first_pair) + ',accuracy_voting,' + str(accuracy_voting) + "\n")
                        file_writer.write('accuracy with differnt number of pairs\n')
                        w = csv.DictWriter(file_writer, accuracy_pairs.keys())
                        w.writeheader()
                        w.writerow(accuracy_pairs)
                        
                    else:
                        file_writer.write('accuracy labeled one pair,' + str(labeled_one) + ',' + 'any,' +  str(labeled_any) + ',all,' + str(labeled_all) + ',avg,'+ str(lablled_avg) + "\n")

                        file_writer.write('accuracy dissimilar to all training (found all similarities > x)\n')
                        w = csv.DictWriter(file_writer, accuracy1.keys())
                        w.writeheader()
                        w.writerow(accuracy1)
                        file_writer.write('accuracy dissimilar to normal (found noraml similarity > x)\n')
                        w = csv.DictWriter(file_writer, accuracy_not_normal.keys())
                        w.writeheader()
                        w.writerow(accuracy_not_normal)     

                        
                    file_writer.write('misclassified\n')
                    w = csv.DictWriter(file_writer, mis_classified.keys())
                    w.writeheader()
                    w.writerow(mis_classified)

        wrapper.siamese_net.save(os.path.join('/home/hananhindy/dump_networks/', '{}_{}_{}'.format(args.comb_index, args.network_id, time.strftime("%Y%m%d-%H%M%S"))))
        
        if args.train_with_all:
            plt.clf()
            training_plot, = plt.plot(loss[loss>0])
            val_plot, = plt.plot(validation_loss[validation_loss>0])
            plt.xlabel('Iteration no')
            plt.ylabel('Loss')
            plt.legend([training_plot, val_plot], ['Training Loss', 'Validation Loss'])
            
            plt.savefig('loss_Run_{}_{}.png'.format(args.network_id, time.strftime("%Y%m%d-%H%M%S")))
#        
#            if i % args.loss_every == 0:
#                if args.print_loss:
#                    loss_array.append(loss) 
#                if args.verbose:
#                    print(loss)
#                
#            if i % args.evaluate_every == 0:
#                if args.is_add_labels and args.train_with_all == False:
#                    val_acc_threasholds, val_acc_labels =  dataset_handler.test_oneshot_adding_labels(wrapper.siamese_net, args.testing_batch_size, args.reps_from_all, args.verbose)
#                    val_acc = val_acc_threasholds[60]
#                elif args.test_vs_all:
#                    val_acc, val_acc_partial = dataset_handler.test_oneshot_new_classes_vs_all(wrapper.siamese_net, args.testing_batch_size, args.verbose)
#                else:
#                    testing_validation_windows = len(dataset_handler.testing_categories)
#                    val_acc = dataset_handler.test_oneshot(wrapper.siamese_net, args.testing_batch_size, testing_validation_windows, args.train_with_all, args.verbose)   
#                
#                if args.is_add_labels  and args.train_with_all == False:
#                    if val_acc_labels >= best_accuracy_labels:
#                        best_accuracy = val_acc
#                        best_accuracy_labels = val_acc_labels
#                        it = i
#                        try:    
#                            best_threashold_acc = val_acc_threasholds
#                            best_accuracy_labels = val_acc_labels
#                        except NameError:
#                            print('Name Error 1')
#                else:
#                    if val_acc >= best_accuracy:
#                        best_accuracy = val_acc
#                        if args.save_best:
#                            wrapper.siamese_net.save('Best_Acc{}_{}'.format(i, val_acc))
#                            
#                    if args.test_vs_all and val_acc_partial >= best_accuracy_partial:
#                        best_accuracy_partial = val_acc_partial
#    
#        with open(args.output_file_name, "a") as file_writer:
#            if args.is_add_labels:
#                try:    
#                    w = csv.DictWriter(file_writer, best_threashold_acc.keys())
#                    w.writeheader()
#                    w.writerow(best_threashold_acc)
#                    file_writer.write(str(it) + ',Labels, ' + str(best_accuracy_labels) + '\n\n')
#                except NameError:
#                    print('Name Error 2')
#            else:
#                file_writer.write(str(best_accuracy) + ',' + str(best_accuracy_partial) + "\n")
#
#            if args.print_loss:
#                file_writer.write(",".join(map(str,loss_array)) + "\n")
