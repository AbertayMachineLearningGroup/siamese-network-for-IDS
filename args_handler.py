#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:22:05 2019

@author: hananhindy
"""
import argparse

# Helper Function
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class Arguments:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.verbose = False
        self.batch_size = 30000          #how many pairs to train on
        self.testing_batch_size = 30000           #how many testing instances to validate on
        
        self.parser.add_argument('--verbose', type = str2bool, help = 'If true, prints will be displayed')
        self.parser.add_argument('--batch_size', help = 'Number of randomly generated pairs for training')
        self.parser.add_argument('--testing_batch_size', help = 'Number of randomly generated pairs for testing')

        self.dataset_name = 'kdd'
        self.parser.add_argument('--dataset_name', help = 'Specify the dataset name ')

        self.add_dataset_specific_defaults()
        
        self.parser.add_argument('--path', help = 'Path of the dataset csv or files directory')
        self.parser.add_argument('--network_id', help = 'Siamese Network ID ')
      
        
        self.train_with_all = True
        self.test_vs_all = False
        
        self.parser.add_argument('--train_with_all', type=str2bool, help='bool train and test with all classes')
        self.parser.add_argument('--test_vs_all', type=str2bool, help = '')
        
        self.evaluate_every = 500     # interval for evaluating 
        
        self.parser.add_argument('--evaluate_every', help = 'interval for evaluating ')

        self.niterations = 1000 
        self.nruns = 1
        
        self.parser.add_argument('--niterations', help = 'Number of training iterations')
        self.parser.add_argument('--nruns', help = 'Number of independent runs')

        self.number_of_training_categories = 4
        self.comb_index = 0
        
        self.parser.add_argument('--number_of_training_categories', help = 'Number of categories used in training')
        self.parser.add_argument('--comb_index', help = 'combinaion index')

        self.output_file_name = 'Result.csv'
        self.parser.add_argument('--output', help = 'Specify the output file name ')

        self.use_sub_batches = False
        self.parser.add_argument('--use_sub_batches', type=str2bool, help = 'Use 3000 per batch')

        self.network_path= ''
        self.parser.add_argument('--network_path')
               
    def parse(self):
        args = self.parser.parse_args() 
        if args.verbose != None:
            self.verbose = args.verbose 
        if args.batch_size != None:
            self.batch_size = int(args.batch_size)
        if args.testing_batch_size != None:
            self.testing_batch_size = int(args.testing_batch_size)
            
            
        if args.dataset_name != None:
            self.dataset_name  = args.dataset_name
            self.add_dataset_specific_defaults()
        
        if args.path != None:
            self.path = args.path
        if args.network_id != None:
            self.network_id = args.network_id

        if args.train_with_all != None:
            self.train_with_all = args.train_with_all
        if args.test_vs_all != None:
            self.test_vs_all = args.test_vs_all
            
        
        if args.evaluate_every != None:
            self.evaluate_every = int(args.evaluate_every)
        
        if args.niterations != None:
            self.niterations = int(args.niterations)
        if args.nruns != None:
            self.nruns = int(args.nruns)
        
        if args.number_of_training_categories != None:
            self.number_of_training_categories = int(args.number_of_training_categories)
        if args.comb_index != None:
            self.comb_index = int(args.comb_index)
            print(self.comb_index)
    
        if args.output != None:
            self.output_file_name = args.output     
        
        if args.use_sub_batches != None:
            self.use_sub_batches = args.use_sub_batches
        
        if args.network_path!= None:
            self.network_path = args.network_path
                       
    
    def add_dataset_specific_defaults(self):
        if self.dataset_name == 'kdd':
            self.path = '/home/hananhindy/Dropbox/SiameseNetworkDatasetFiles/DatasetProcessedFiles/kddcup.data_10_percent_corrected'
        elif self.dataset_name == 'SCADA' or self.dataset_name == 'SCADA_Reduced':
            self.path = '/home/hananhindy/Dropbox/SiameseNetworkDatasetFiles/DatasetProcessedFiles/SCADA_dataset_processed.csv'
        elif self.dataset_name == 'CICIDS' or self.dataset_name == 'CICIDS2':
            self.path = '/home/hananhindy/CICIDS'
        elif self.dataset_name == 'nsl-kdd':
            self.path= '/home/hananhindy/Dropbox/SiameseNetworkDatasetFiles/DatasetProcessedFiles/KDDTrain+.txt'
