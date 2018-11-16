# Siamese Network Usage for Learning from Small Dataset and for Zero-day attack detection

This repo build a siamese network model, train and test it.
Datasets:
- KDD 10%
- Water SCADA System
- ST. Andrews STA2018 

Script Parameters and Defaults
| Argument      | Usage         | Default       |  Additional Options |
| ------------- |:-------------:|:-------------:|:-------------------:|
| --verbose     | Print verbose | True          |                     |
| --path        |               |               |                     |
| --network_id  | Siamese Network Architecture | kdd_0 | kdd_0 : 95, 70, 47, 23, 5 - kdd_1 : 98, 79, 59, 39, 20, 5 - kdd_2: 101, 84, 67, 51, 24, 17, 5 - kdd_3: 103, 89, 74, 59, 44, 30, 15, 5 | 
| --batch_size    | Number of randomaly gerenated pairs for training | 250 | | 
| --testing_batch_size | Number of randomaly gerenated pairs for testing | 250 ||
| --nruns | Number of indeoendant runs | 10 ||
| --niterations | Number of training iterations | 1000 ||
| --evaluate_every | Interval for testing | 10 || 
| --max_from_class | Max count to take from class | 1200 ||
| --comb_index     | | 0| Should be set with the combination index when train with all is set to false |
| --train_with_all | | True ||
| --test_vs_all || False | Should be true if train with all is false to test one shot learning |
| --save_best | Save best accuracy model | False || 

