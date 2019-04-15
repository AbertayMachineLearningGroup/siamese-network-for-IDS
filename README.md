# Siamese Network Usage for Learning from Small Dataset and for Zero-day attack detection

This repository builds, trains and tests Siamese Network model.

## Used Datasets:
- kddcup.data_10_percent [here](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- STA2018 [here](https://github.com/elud074/STA2018)
- SCADA [here](https://www.sciencedirect.com/science/article/pii/S2352340917303402)

All processed files can be downloaded from [here](https://www.dropbox.com/sh/8y9jni9einfjnyd/AADcSqNs4cG0sQfy2Cias4tfa?dl=0)

## To run the script with the default parameters:
The default of the script is to run with 80:20 training:testing on KDD dataset and all the categories are used in training and testing. 

In this case, the network  is evaluated based on learning from **small datasets**.

To run (remember to add the parameters as specified in the below tables as needed):
- **kdd**: python main.py --path {}
- **STA2018**: python main.py --path {} --dataset_name STA 
- **SCADA**: python main.py --path {} --dataset_name SCADA 


## Script Arguments
1. General Parameters

| Argument       | Usage        				 	     | Default       |  Values and Notes	          |
| ---------------|:-------------------------------------:|:-------------:|:-------------------|
| --verbose      | If true, prints will be displayed     | True 		 | 	 				  |
| --path         | Path of the dataset files / directory |               | **kdd**: path to the kddcup.data_10_percent_corrected <br> **STA**: path to the directory generated from STA preprocessing <br> **SCADA**: path to the processed csv file|
| --dataset_name | Specify the dataset name              | kdd           | kdd <br> STA <br> SCADA  |
| --batch_size   | Number of randomly generated pairs for training | 250 |                   |
| --testing_batch_size | Number of randomly generated pairs for testing | 250 ||
| --max_from_class | Number of instaqnces to use from class (used to limit the pool of instances to select from) | -1 | -1: no max <br> Recommended values **kdd**: 1200 <br> **STA**: 642 <br> **SCADA**: 157 | 
| --niterations | Number of training iterations | 1000 ||
| --nruns | Number of independent runs | 10 ||
| --evaluate_every | Interval for testing       | 10 || 
| --network_id   | Siamese Network Architecture          | kdd_0 <br> STA_0 | kdd_0 : 95, 70, 47, 23, 5 <br> kdd_1 : 98, 79, 59, 39, 20, 5 <br> kdd_2: 101, 84, 67, 51, 24, 17, 5 <br> kdd_3: 103, 89, 74, 59, 44, 30, 15, 5 <br> STA_0: 510, 450, 390, 330, 270, 210, 150, 90, 30, 3 <br> STA_1: 480, 360, 240, 120, 3 <br> STA_2: 500, 400, 300, 200, 100, 3 <br> STA_3: 514, 428, 342, 257, 171, 86, 3 <br> STA_4: 525, 450, 375, 300, 225, 150, 75, 3 <br> STA_5 533, 467, 400, 333, 267, 200, 133, 65, 3 <br> SCADA_0: 20, 14 | 
| --save_best | Save the best accuracy model | False || 
| --k | Choose k for cross validation | 0 | 0, 4 (i.e. 5-folds) | 
| --output  | The output file name | Result.csv ||

<br>

2. Zero-day attack related arguments

| Argument       | Usage         | Default       |  Values and notes          |
| ---------------|:-------------:|:-------------:|:--------------------|
| --train_with_all | If true, all classes are used in training and testing <br> If false, X will be used in training and the rest will be used in testing | True | If using **STA**, cannot be false as only 3 classes are there |
| --comb_index    | In case of not using all classes, this specify the index of the combination to be used in training  | 0 (i.e. first combination) | Not applicable with **STA** |
| --number_of_training_categories | The number of categories to be used for training when train_with_all is false | 3 ||
| --test_vs_all   | If true, the testing classes are validated against training (zero-day detection) | False | Should be true if train with all is false to test one shot learning |

3. Categories Representatives
| Argument       | Usage         | Default       |  Values and notes          |
| ---------------|:-------------:|:-------------:|:--------------------|
| --number_of_reps | Number of representatives to use per category | 0  | No reps are used by default |
| --is_add_labels | To add labels for Zero-Day detected attacks | False | |


