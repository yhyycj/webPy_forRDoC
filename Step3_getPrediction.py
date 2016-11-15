# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 15:39:06 2016

@author: 310149083
"""

import json
import random
import pandas as pd
import numpy as np
import scipy
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

def modelEnsemble(predRlt, ensemble):
    if ensemble == 'vote':
        for row_i in range(predRlt.shape[0]):
            preds = predRlt.values[row_i][2:].tolist()
            if len(set(preds)) == len(preds):
                print 'all predictions are different for', predRlt['fileName'][row_i]
            mode = scipy.stats.mode(preds).mode[0]
            predRlt['Score'][row_i] = mode
    elif ensemble == 'worst':
        for row_i in range(predRlt.shape[0]):
            preds = predRlt.values[row_i][2:].tolist()
            for s in ['SEVERE', 'MODERATE', 'MILD', 'ABSENT']:
                if s in preds:
                    predRlt['Score'][row_i] = s
                    break
    return predRlt
    
def sampleDown(matrix):
    while True:
        type_counts = pd.pivot_table(matrix, values = 'fileName', index = 'Score', aggfunc = np.count_nonzero)
        print type_counts
        tar_class = raw_input('target class?')
        if tar_class == '':
            break
        keep_num = raw_input('sample down to ___ size?')
        rowsToKeep = matrix[matrix.Score != tar_class]
        sampledRows = matrix[matrix.Score == tar_class].sample(n = int(keep_num))
        matrix = pd.concat([rowsToKeep, sampledRows])
    return matrix
    
def sampleUp(matrix):
    while True:
        type_counts = pd.pivot_table(matrix, values = 'fileName', index = 'Score', aggfunc = np.count_nonzero)
        print type_counts
        tar_class = raw_input('target class?')
        if tar_class == '':
            break
        keep_num = raw_input('sample up to ___ size?')
        rowsToKeep = matrix[matrix.Score != tar_class]
        magnify_times = int(keep_num)*1.0/type_counts[tar_class]
        while magnify_times > 1:
            rowsToKeep = pd.concat([rowsToKeep, matrix[matrix.Score == tar_class]])
            magnify_times -= 1
        magnify_num = type_counts[tar_class] * magnify_times
        sampledUps = matrix[matrix.Score == tar_class].sample(n = int(magnify_num))
        matrix = pd.concat([rowsToKeep, sampledUps])
    return matrix
    
def getPredict(matrix_train_csv, matrix_test_csv, models = ['gnb'], ensemble = 'vote', balance = 'sampleDown', toBinary = None):
    if matrix_train_csv.endswith('.txt'):
        seperator = '\t'
    else:
        seperator = ','
    
    train = pd.read_csv(matrix_train_csv, sep = seperator)
    # deal with unbalance class 
    if balance is None:
        X_train = train.drop(['fileName', 'Score'], axis = 1)
        y_train = train['Score']
    elif balance == 'sampleDown':
        train_down = sampleDown(train)
        X_train = train_down.drop(['fileName', 'Score'], axis = 1)
        y_train = train_down['Score']
    elif balance == 'sampleUp':
        train_up = sampleUp(train)
#        train_up.to_csv('sampleUp.csv', index = False)
        X_train = train_up.drop(['fileName', 'Score'], axis = 1)
        y_train = train_up['Score']
    
    test = pd.read_csv(matrix_test_csv, sep = seperator)
    X_test = test.drop(['fileName', 'Score'], axis = 1)
    y_test = test['Score']
    
    if toBinary is not None:
        allValues = y_train.unique()
        mapdict = dict.fromkeys(allValues, 0)
        for tb in toBinary:
            mapdict[tb] = 1
        y_train.replace(mapdict, inplace = True)
        y_test.replace(mapdict, inplace = True)
    
    predRlt = pd.DataFrame({'fileName': test[test.columns[0]].tolist(), 'Score': ['' for i in range(test.shape[0])]})
    
    if 'all' in models or 'gnb' in models:
        try:
            gnb = GaussianNB()
            y_train_gnb_pred = gnb.fit(X_train, y_train).predict(X_train)
            y_test_gnb_pred = gnb.fit(X_train, y_train).predict(X_test)
            print 'success to build gnb model'
            print("[training] Number of mislabeled points out of a total %d points : %d" % (X_train.shape[0], (y_train != y_train_gnb_pred).sum()))
            print '[training] mean_absolute_error:', mean_absolute_error(y_train.replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}), pd.DataFrame(y_train_gnb_pred).replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}))
            print("[testing] Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_test_gnb_pred).sum()))
            print '[training] mean_absolute_error:', mean_absolute_error(y_test.replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}), pd.DataFrame(y_test_gnb_pred).replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}))
            predRlt['gnb'] = y_test_gnb_pred
        except:
            print 'fail to build gnb model'
        
    if 'all' in models or 'lgr' in models:
        try:
            g = LogisticRegression(C = 1000, random_state = 0, penalty = 'l1')
            y_train_lr_pred = lgr.fit(X_train, y_train).predict(X_train)
            y_test_lr_pred = lgr.fit(X_train, y_train).predict(X_test)
            print 'success to build lgr model'
            print("[training] Number of mislabeled points out of a total %d points : %d" % (X_train.shape[0], (y_train != y_train_lr_pred).sum()))
            print '[training] mean_absolute_error:', mean_absolute_error(y_train.replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}), pd.DataFrame(y_train_lr_pred).replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}))
            print("[testing] Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_test_lr_pred).sum()))
            print '[training] mean_absolute_error:', mean_absolute_error(y_test.replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}), pd.DataFrame(y_test_lr_pred).replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}))
            predRlt['lgr'] = y_test_lr_pred
        except:
            print 'fail to build lgr model'
  
    if 'all' in models or 'lr' in models:
        mapdict = dict.fromkeys(y_train.unique(), 0)
        mapdict['MILD'] = 1
        mapdict['MODERATE'] = 2
        mapdict['SEVERE'] = 3        
        y_train_numeric = y_train.replace(mapdict)
        try:            
            lr = LinearRegression()
            y_train_lr_pred = lr.fit(X_train, y_train_numeric).predict(X_train)
            y_test_lr_pred = lr.fit(X_train, y_train_numeric).predict(X_test)
            y_test_lr_pred_class = [0] * y_test_lr_pred.shape[0]
            for i in range(y_test_lr_pred.shape[0]):
                if i <= 0.5: 
                    y_test_lr_pred_class[i] = 'ABSENT'
                elif i <= 1.5:
                    y_test_lr_pred_class[i] = 'MILD'
                elif i <= 2.5:
                    y_test_lr_pred_class[i] = 'MODERATE'
                else:
                    y_test_lr_pred_class[i] = 'SEVERE'
            print 'success to build lr model'
            print("[training] Number of mislabeled points out of a total %d points : %d" % (X_train.shape[0], (y_train != y_train_lr_pred).sum()))
            print '[training] mean_absolute_error:', mean_absolute_error(y_train.replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}), y_train_lr_pred)
            print("[testing] Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_test_lr_pred_class).sum()))
            print '[training] mean_absolute_error:', mean_absolute_error(y_test.replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}), y_test_lr_pred)
            predRlt['lr'] = y_test_lr_pred_class
        except:
            print 'fail to build lgr model'
            
    if 'all' in models or 'mnb' in models:
        try:
            mnb = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
            y_train_mnb_pred = mnb.fit(X_train, y_train).predict(X_train)
            y_test_mnb_pred = mnb.fit(X_train, y_train).predict(X_test)
            print 'success to build mnb model'
            print("[training] Number of mislabeled points out of a total %d points : %d" % (X_train.shape[0], (y_train != y_train_mnb_pred).sum()))
            print '[training] mean_absolute_error:', mean_absolute_error(y_train.replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}), pd.DataFrame(y_train_mnb_pred).replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}))
            print("[testing] Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_test_mnb_pred).sum()))
            print '[training] mean_absolute_error:', mean_absolute_error(y_test.replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}), pd.DataFrame(y_test_mnb_pred).replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}))
            predRlt['mnb'] = y_test_mnb_pred
        except:
            print 'fail to build mnb model'
        
    if 'all' in models or 'rf' in models:
        try:
            ranFor = RandomForestClassifier()
            y_train_rf_pred = ranFor.fit(X_train, y_train).predict(X_train)
            y_test_rf_pred = ranFor.fit(X_train, y_train).predict(X_test)
            print 'success to build rf model'
            print("[training] Number of mislabeled points out of a total %d points : %d" % (X_train.shape[0], (y_train != y_train_rf_pred).sum()))
            print '[training] mean_absolute_error:', mean_absolute_error(y_train.replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}), pd.DataFrame(y_train_rf_pred).replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}))
            print("[testing] Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_test_rf_pred).sum()))
            print '[training] mean_absolute_error:', mean_absolute_error(y_test.replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}), pd.DataFrame(y_test_rf_pred).replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}))
            predRlt['rf'] = y_test_rf_pred
        except:
            print 'fail to build rf model'
            
    if 'all' in models or 'svm' in models or 'svc' in models:
        try:
            clf = SVC()
            y_train_svc_pred = clf.fit(X_train, y_train).predict(X_train)
            y_test_svc_pred = clf.fit(X_train, y_train).predict(X_test)
            print 'success to build SVC model'
            print("[training] Number of mislabeled points out of a total %d points : %d" % (X_train.shape[0], (y_train != y_train_svc_pred).sum()))
            print '[training] mean_absolute_error:', mean_absolute_error(y_train.replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}), pd.DataFrame(y_train_svc_pred).replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}))
            print("[testing] Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_test_svc_pred).sum()))
            print '[training] mean_absolute_error:', mean_absolute_error(y_test.replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}), pd.DataFrame(y_test_svc_pred).replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}))
            predRlt['svc'] = y_test_rf_pred
        except:
            print 'fail to build svc model'
    
    if 'all' in models or 'tree' in models:
        try:
            clf = tree.DecisionTreeClassifier()
            y_train_tree_pred = clf.fit(X_train, y_train).predict(X_train)
            y_test_tree_pred = clf.fit(X_train, y_train).predict(X_test)
            print 'success to build tree model'
            print("[training] Number of mislabeled points out of a total %d points : %d" % (X_train.shape[0], (y_train != y_train_tree_pred).sum()))
            print '[training] mean_absolute_error:', mean_absolute_error(y_train.replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}), pd.DataFrame(y_train_tree_pred).replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}))
            print("[testing] Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_test_tree_pred).sum()))
            print '[training] mean_absolute_error:', mean_absolute_error(y_test.replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}), pd.DataFrame(y_test_tree_pred).replace({'ABSENT':0, 'MILD':1, 'MODERATE':2, 'SEVERE':3}))
            predRlt['tree'] = y_test_tree_pred
        except:
            print 'fail to build tree model'
            
    
    #ensembling
    if ensemble:
        finalPred = modelEnsemble(predRlt, ensemble)
    else:
        finalPred = predRlt
    return finalPred
    
if __name__ == '__main__':
    # ensembledRlts = getPredict(matrix_train_csv = r'..\rlts\Matrix_Train_328.txt', matrix_test_csv = r'..\rlts\Matrix_Validate_105.txt', models = ['all'], ensemble = 'vote')
    ensembledRlts = getPredict(matrix_train_csv = r'..\rlts\Matrix_Train_328.txt', matrix_test_csv = r'..\rlts\Matrix_Validate_105.txt', models = ['lr'], ensemble = 'vote', balance = None, toBinary = None)
    json.dump(dict(ensembledRlts[['fileName', 'Score']].values), open(r'..\rlts\linearRegression.json', 'w'), indent = 4)