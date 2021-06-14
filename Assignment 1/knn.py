import numpy as np
import glob
import os 
import pandas as pd
import stats
from sklearn.metrics import accuracy_score
import random

class KNN_with_cv:
    def __init__(self, k_max = 30, k_folds = 10):
        self.k_max = k_max
        self.k_folds = k_folds

    def data_loader(self, train_data_folds = [], train_labels_folds=[] , test_data=[] ,test_labels = [] , data_dir = 'knn-dataset/'):
        self.train_data_folds = train_data_folds
        self.train_labels_folds = train_labels_folds
        self.test_data = test_data
        self.test_labels = test_labels

        filenames = glob.glob(os.path.join(data_dir+"*.csv"))
        train_data_files = []
        train_label_files = []

        for filename in filenames:
            if('trainData' in filename):
                train_data_files.append(filename)
            if('trainLabels' in filename):
                train_label_files.append(filename)
            
        train_data_files = sorted(train_data_files)
        train_label_files = sorted(train_label_files)

        for i in range(len(train_data_files)):
            self.train_data_folds.append(np.genfromtxt(train_data_files[i], delimiter=','))
            self.train_labels_folds.append(np.genfromtxt(train_label_files[i], delimiter=','))

        self.test_data = np.genfromtxt(os.path.join(data_dir+'testData.csv'), delimiter=',')
        self.test_labels = np.genfromtxt(os.path.join(data_dir+'testLabels.csv'), delimiter=',')

    def cross_validation(self, train_data_list, train_labels_list, k, k_fold):
        val_data = train_data_list[k_fold]
        val_labels = train_labels_list[k_fold]
        train_x = train_data_list[:k_fold] + train_data_list[k_fold+1:]
        train_y = train_labels_list[:k_fold] + train_labels_list[k_fold+1:]
        train_data = np.concatenate(train_x, axis = 0)
        train_labels = np.concatenate(train_y, axis = 0)
        
        preds = []
        for i,row in enumerate(val_data):
            #print(i)
            instance = row.reshape(1,-1)
            idx = np.argpartition(np.sqrt(np.sum((train_data-instance)**2,axis = 1)), k)
            #print(k)
            #print(idx)
            knn_labels = train_labels[idx[:k]]
            try:
                pred = stats.mode(knn_labels)
            except:
                pred = random.choice(knn_labels)
            preds.append(pred)
        
        return(accuracy_score(val_labels,preds))


    def evaluate(self, k):
        train_X = np.concatenate(self.train_data_folds, axis = 0)
        train_y = np.concatenate(self.train_labels_folds, axis = 0)
        test_X = self.test_data
        test_y = self.test_labels
        preds = []
        for i,row in enumerate(test_X):
            instance = row.reshape(1,-1)
            idx = np.argpartition(np.sqrt(np.sum((train_X-instance)**2,axis = 1)), k)
            knn_labels = train_y[idx[:k]]
            try:
                pred = stats.mode(knn_labels)
            except:
                pred = random.choice(knn_labels)
            preds.append(pred)
        
        return(accuracy_score(test_y,preds))

    def fit(self):
        accuracy_list = []
        for k in range(self.k_max):
            accuracy_k_fold_list = []
            for k_fold in range(self.k_folds): 
                accuracy_k_fold = self.cross_validation(self.train_data_folds, self.train_labels_folds, k+1, k_fold)
                accuracy_k_fold_list.append(accuracy_k_fold)
            accuracy_k = np.mean(accuracy_k_fold_list)
            accuracy_list.append(accuracy_k)

        k_optimized = accuracy_list.index(max(accuracy_list)) + 1
        return {'accuracy_list': accuracy_list, 'k_optimized': k_optimized}
