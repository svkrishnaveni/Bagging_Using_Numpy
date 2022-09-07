#!/usr/bin/env python
'''
This script contains various functions used in this project
Author: Sai Venkata Krishnaveni Devarakonda
Date: 04/11/2022
'''

import utilities
str_path_1b_program = './data_2c2d3c3d_program.txt'

#splitting data into train and test data sets
str_path_1b_program = './data_2c2d3c3d_program.txt'
features,targets = utilities.Load_data(str_path_1b_program)
arr3d_train_features = features[0:90]
arr3d_test_features = features[90:120]
arr1d_train_targets = targets[0:90]
arr1d_test_targets = targets[90:120]

trainfeatures,trainlabels,testfeature,testlabels = arr3d_train_features,arr1d_train_targets,arr3d_test_features,arr1d_test_targets
bagging_times = [10,50,100]
for i in bagging_times:
    train_pred = utilities.bagging(trainfeatures, trainlabels, 4, i, trainfeatures)
    test_pred = utilities.bagging(trainfeatures, trainlabels, 4, i, testfeature)

    train_accuracy = utilities.accuracy(trainlabels, train_pred)
    test_accuracy = utilities.accuracy(testlabels, test_pred)
    print('Train Accuracy for bagging with ' +str(i)+' bootstrap samples is ' +str(train_accuracy))
    print('Test Accuracy for bagging with ' +str(i)+' bootstrap samples is ' +str(test_accuracy))
    print('\n')