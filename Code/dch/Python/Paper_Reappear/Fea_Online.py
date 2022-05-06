# -*- coding:utf-8 -*-

"""
作者: Gary Ding
日期: 2021年11月20日
"""

import pandas as pd
import numpy as np
import csv

from imblearn.combine import SMOTEENN
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score, \
    classification_report
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE

# In this project, we need to transform the feature Matrix.
def tryPatientInfo(feaMatr):
    row=feaMatr.shape[0]
    col=feaMatr.shape[1]
    feaM=np.ones((row,col-1)) # the final feature Matrix

    for i in range(row):
        for j in range(3):
            feaM[i][j]=feaMatr[i][j]

    for i in range(row):
        feaM[i][3]=feaMatr[i][4]/feaMatr[i][3]

    for i in range(row):
        feaM[i][4]=feaMatr[i][5]/feaMatr[i][3]

    return feaM

# Form the feature Matrix and the target Label
def getMatrix(result):
    # Remove description
    row=result.shape[0]-1
    col=result.shape[1]-2

    feaMatr = np.ones((row, col))
    lab=[] # the final label list

    # all feature Matrix
    for i in range(row):
        for j in range(col):
            feaMatr[i][j] = result[i + 1][j + 1]

    for i in range(row):
        for j in range(1):
            lab.append(result[i + 1][j + col + 1])  # 获取最后一列的label

    feaM=tryPatientInfo(feaMatr)

    return feaM,lab

# Data imbalance processing
def dataEnhancement(feaMatr,lab):
    # 0. Do nothing
    # 1. SMOTE
    # sm = SMOTEENN({'URTI':58},random_state=42)
    # feaMatr, lab = sm.fit_resample(feaMatr, lab)

    X_train,X_test,y_train,y_test=train_test_split(feaMatr,lab,test_size=0.2, random_state=42)
    return X_train,X_test,y_train,y_test

# Normalization/Standardization
def tryScalar(X_train,X_test):
    # Normalization
    # min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    # X_train = min_max_scaler.fit_transform(X_train)
    # X_test = min_max_scaler.fit_transform(X_test)

    # Standardization
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.fit_transform(X_test)

    return X_train,X_test

# Get the final result
def tryResult(X_train,X_test,y_train,y_test):

    # SVM
    # model = OneVsRestClassifier(svm.LinearSVC(random_state = 0, verbose = 1))
    # model.fit(X_train, y_train)

    # LR
    lr_clf = LogisticRegression(random_state=0, solver='sag', multi_class='ovr', verbose=1)
    lr_clf.fit(X_train, y_train)

    # 1、AUC
    # y_score = model.decision_function(X_test)   #svm
    y_pred_pa = lr_clf.predict_proba(X_test)  # lr
    y_test_oh = label_binarize(y_test, classes=['COPD', 'Healthy', 'Other', 'URTI'])
    # y_test_oh = label_binarize(y_test, classes=[0, 1, 2, 3])

    # 2. For Confusion Matrix
    # y_pred = model.predict_proba(X_test) # svm
    y_pred = lr_clf.predict(X_test)  # lr

    # 3. Model Evaluation
    print()
    print('------------------------------------')
    # print('roc_auc_score：', roc_auc_score(y_test, y_score, average='micro',multi_class="ovr"))  #svm
    print('roc_auc_score：', roc_auc_score(y_test_oh, y_pred_pa, average='micro'))  # lr,lgm
    print('precision_score: ', precision_score(y_test, y_pred, average='micro'))
    print('recall_score: ', recall_score(y_test, y_pred, average='micro'))
    print('f1_score: ', f1_score(y_test, y_pred, average='micro'))
    print('------------------------------------')
    print()

    # 4. Model Report
    target_names = ['COPD', 'Healthy', 'Other', 'URTI']
    print(classification_report(y_test, y_pred, digits=2, target_names=target_names))



if __name__ =='__main__':
    with open('patient_info2.csv','r') as p:
        reader = csv.reader(p)
        result = list(reader) # to UTF compile standard

    feaMatr,lab=getMatrix(np.array(result))
    X_train, X_test, y_train, y_test=dataEnhancement(feaMatr,lab)
    X_train, X_test=tryScalar(X_train, X_test)
    tryResult(X_train, X_test, y_train, y_test)
