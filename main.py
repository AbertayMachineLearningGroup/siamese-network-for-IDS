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

    for run in range(args.nruns):
        print(args.network_id)
        
        if args.print_loss:
            loss_array = []
        
        if args.verbose:
            print("Run #{}".format(run))
            
        wrapper = SiameseNet((dataset_handler.number_of_features,), args.network_id, args.dataset_name, args.verbose)

        if args.network_path != '':
            wrapper.load_saved_model(args.network_path)
            if args.train_with_all or args.test_vs_all:
                dataset_handler.evaluate_classisfication(
                                        'pairs/{}_{}_{}_Classification_Pairs.csv'.format(args.dataset_name, args.batch_size,args.comb_index),
                                         wrapper.siamese_net, 
                                         args.batch_size,
                                         len(all_classes), 
                                         all_classes, args.output_file_name)
            else:
                index_of_zero_day_category = all_classes.index([item for item in all_classes if item not in training_categories][0])
                dataset_handler.evaluate_zero_day_new(
                                        'pairs/{}_{}_{}_Classification_Pairs.csv'.format(args.dataset_name, args.batch_size, args.comb_index),
                                        wrapper.siamese_net, 
                                        args.batch_size,
                                        len(all_classes),
                                        index_of_zero_day_category, 
                                        training_categories, 
                                        args.output_file_name)
        else:
            (inputs1, targets1) = dataset_handler.load_batch(args.batch_size, 'pairs/{}_{}_{}_Training_Pairs.csv'.format(args.dataset_name, args.batch_size, args.comb_index))
            if args.train_with_all:
                (inputs_val, targets_val) = dataset_handler.load_batch(args.batch_size, 'pairs/{}_{}_{}_Validation_Pairs.csv'.format(args.dataset_name, args.batch_size,args.comb_index))
            
            loss = np.zeros(args.niterations)
            validation_loss = np.zeros(args.niterations)
            tr_acc = np.zeros(args.niterations)
            val_acc = np.zeros(args.niterations)
            
            for i in range(1, args.niterations + 1):
                if args.use_sub_batches:
                    if args.train_with_all:
                        hist = wrapper.siamese_net.fit(inputs1,targets1, batch_size = args.batch_size//1000, validation_data= (inputs_val,targets_val)).history
                        val_acc[i-1] = round(hist['val_accuracy'][0]*100, 2)
                    else:
                        hist = wrapper.siamese_net.fit(inputs1,targets1, batch_size = args.batch_size//1000).history
                        
                    loss[i-1] = hist['loss'][0]
                    tr_acc[i-1] = round(hist['accuracy'][0]*100, 2)
                else:
                    loss[i-1] = wrapper.siamese_net.train_on_batch(inputs1,targets1)
                    
                if args.train_with_all:
                    if args.use_sub_batches:
                        validation_loss[i-1] = hist['val_loss'][0]
                    else:
                        validation_loss[i-1] = wrapper.siamese_net.test_on_batch(inputs_val, targets_val)
               
                print('{} -> Loss = {} validation loss = {}'.format(i, loss[i-1], validation_loss[i-1]))
            
                if i >= args.evaluate_every and i%args.evaluate_every == 0:
                    if args.train_with_all or args.test_vs_all:
                        plt.clf()
                        training_plot, = plt.plot(loss[0:i])
                        val_plot, = plt.plot(validation_loss[0:i])
                        plt.xlabel('Iteration no')
                        plt.ylabel('Loss')
                        plt.legend([training_plot, val_plot], ['Training Loss', 'Validation Loss'])
                        
                        plt.savefig('loss_Run_{}.png'.format(args.network_id))
                        dataset_handler.evaluate_classisfication(
                                        'pairs/{}_{}_{}_Classification_Pairs.csv'.format(args.dataset_name, args.batch_size,args.comb_index),
                                         wrapper.siamese_net, 
                                         args.batch_size,
                                         len(all_classes), 
                                         all_classes, args.output_file_name)
                    else:
                        index_of_zero_day_category = all_classes.index([item for item in all_classes if item not in training_categories][0])
                        accuracy_zero_day, conf_mat = dataset_handler.evaluate_zero_day_new('pairs/{}_{}_{}_Classification_Pairs.csv'.format(args.dataset_name, args.batch_size, args.comb_index),
                                                                                                  wrapper.siamese_net, 
                                                                                                  args.batch_size,
                                                                                                  len(all_classes),
                                                                                                  index_of_zero_day_category, 
                                                                                                  training_categories, 
                                                                                                  args.output_file_name)    

    
            wrapper.siamese_net.save(os.path.join('/home/hananhindy/dump_networks/', '{}_{}_{}_{}'.format(args.dataset_name, args.comb_index, args.network_id, time.strftime("%Y%m%d-%H%M%S"))))
            
            if args.train_with_all or args.test_vs_all:
                plt.clf()
                training_plot, = plt.plot(loss[loss>0])
                val_plot, = plt.plot(validation_loss[validation_loss>0])
                plt.xlabel('Iteration no')
                plt.ylabel('Loss')
                plt.legend([training_plot, val_plot], ['Training Loss', 'Validation Loss'])
                
                plt.savefig('loss_Run_{}_{}_{}.png'.format(args.dataset_name, args.network_id, time.strftime("%Y%m%d-%H%M%S")))
                plt.ylim(bottom=0, top=3)  # adjust the top leaving bottom unchanged
                
                plt.savefig('limit_loss_Run_{}_{}.png'.format(args.network_id, time.strftime("%Y%m%d-%H%M%S")))
    
                plt.clf()
                training_plot, = plt.plot(tr_acc[tr_acc>0])
                val_plot, = plt.plot(val_acc[val_acc>0])
                plt.xlabel('Iteration no')
                plt.ylabel('Accuracy')
                plt.legend([training_plot, val_plot], ['Training Acc', 'Validation Acc'])
