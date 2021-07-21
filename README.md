# Siamese Network Usage for Learning from Small Dataset and for Zero-day attack detection

This repository builds, trains and tests Siamese Network model.
The pairs and trained models are added to the repository for reproducibility.

## Citation
To cite this code, please use the following papers;

```
@article{hindy2020leveraging,
  title={Leveraging Siamese Networks for One-Shot Intrusion Detection Model},
  author={Hindy, Hanan and Tachtatzis, Christos and Atkinson, Robert and Brosset, David and Bures, Miroslav and Andonovic, Ivan and Michie, Craig and Bellekens, Xavier},
  journal={arXiv preprint arXiv:2006.15343},
  year={2020}
}

````

````
@inproceedings{10.1145/3437984.3458842,
author = {Hindy, Hanan and Tachtatzis, Christos and Atkinson, Robert and Bayne, Ethan and Bellekens, Xavier},
title = {Developing a Siamese Network for Intrusion Detection Systems},
year = {2021},
isbn = {9781450382984},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3437984.3458842},
doi = {10.1145/3437984.3458842},
booktitle = {Proceedings of the 1st Workshop on Machine Learning and Systems},
pages = {120–126},
numpages = {7},
keywords = {Few-Shot Learning, Artificial Neural Network, Machine Learning, NSL-KDD, Intrusion Detection, CICIDS2017, Siamese Network, KDD Cup'99},
location = {Online, United Kingdom},
series = {EuroMLSys '21}
}
````

## Usage Scenarios

TBU

## Used Datasets:
- CICIDS2017 [here](https://www.unb.ca/cic/datasets/ids-2017.html)
- kddcup.data_10_percent [here](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- NSL-KDD [here](https://www.unb.ca/cic/datasets/nsl.html)
- SCADA [here](https://www.sciencedirect.com/science/article/pii/S2352340917303402)

All processed files can be downloaded from [here](https://www.dropbox.com/sh/8y9jni9einfjnyd/AADcSqNs4cG0sQfy2Cias4tfa?dl=0)

All pairs can be downloaded from [here](https://www.dropbox.com/sh/8y9jni9einfjnyd/AADcSqNs4cG0sQfy2Cias4tfa?dl=0)

## To run the script with the default parameters:
To run (remember to add the parameters as specified in the below tables as needed):
- **CICIDS2017**: python main.py --path {} --dataset_name CICIDS 
- **kdd**: python main.py --path {} --dataset_name kdd 
- **nsl**: python main.py --path {} --dataset_name nsl-kdd 
- **SCADA**: python main.py --path {} --dataset_name SCADA 


## Script Arguments
1. General Parameters

| Argument       | Usage        				 	     | Default       |  Values and Notes	          |
| ---------------|:-------------------------------------:|:-------------:|:-------------------|
| --verbose      | If true, prints will be displayed     | True 		 | 	 				  |
| --path         | Path of the dataset files / directory |               | 					  |
| --dataset_name | Specify the dataset name              | kdd           | CICIDS <br>kdd <br> nsl-kdd <br> SCADA  |
| --batch_size   | Number of randomly generated pairs for training | 30000 |                   |
| --testing_batch_size | Number of randomly generated pairs for testing | 30000 | |
| --niterations | Number of training iterations | 1000 ||
| --evaluate_every | Interval for testing       | 500 || 
| --output  | The output file name | Result.csv ||
| --network_path  | The path of trained network (used to test a trained network and not train from scratch)|  | |

<br>

2. One-Shot related arguments

| Argument       | Usage         | Default       |  Values and notes          |
| ---------------|:-------------:|:-------------:|:--------------------|
| --train_with_all | If true, all classes are used in training and testing <br> If false, K will be used in training and the rest will be used in testing | True |  |
| --test_vs_all   | If true, the testing classes are tested against all classes | False | Should be true if train with all is false to test one shot learning |
| --comb_index    | In case of not using all classes, this specify the index of the combination to be used in training  | 0 (i.e. first combination) |  |
| --number_of_training_categories | The number of categories to be used for training when train_with_all is false | 3 ||


