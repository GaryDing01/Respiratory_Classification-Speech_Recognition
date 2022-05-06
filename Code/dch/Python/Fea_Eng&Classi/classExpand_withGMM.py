# -*- coding:utf-8 -*-

"""
作者: Gary Ding
日期: 2021年12月20日
"""
import math

import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report,accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks,NearMiss
from sklearn.decomposition import LatentDirichletAllocation

#综合采样
from imblearn.combine import SMOTEENN,SMOTETomek

import lightgbm as lgb

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from loadFeaLab import train_fea,train_lab,test_fea,test_lab
# from loadFeaLab_try2 import train_fea,train_lab,test_fea,test_lab

# Use GMM
# Convert Array from String to float
train_temp=np.ones((len(train_fea),len(train_fea[0])),dtype=float)
for i in range(len(train_fea)):
    for j in range(len(train_fea[0])):
        train_temp[i][j] = float(train_fea[i][j])

test_temp=np.ones((len(test_fea),len(test_fea[0])),dtype=float)
for i in range(len(test_fea)):
    for j in range(len(test_fea[0])):
        test_temp[i][j]=float(test_fea[i][j])

# Rearranging data
all_fea=np.vstack((train_fea,test_fea))
all_lab=train_lab+test_lab

fea_Matrix=train_fea
lab_Cla=train_lab

# # Data imbalance processing
# # SMOTE
# sm = SMOTE({'Healthy':420,'Other':420,'URTI':420},random_state=42)
# train_temp, train_lab = sm.fit_resample(train_temp, train_lab)
# # print('Resampled dataset shape %s' % Counter(y_res))

# #ADASYN
# sm = ADASYN({'Healthy':350,'Other':350,'URTI':350},random_state=42)
# fea_Matrix, lab_Cla = sm.fit_resample(fea_Matrix, lab_Cla)
# # print('Resampled dataset shape %s' % Counter(y_res))

# #BorderlineSMOTE
# sm = BorderlineSMOTE({'linux':2000},random_state=42)
# fea_Matrix, lab_Cla = sm.fit_resample(fea_Matrix, lab_Cla)
# # print('Resampled dataset shape %s' % Counter(y_res))

# Undersampling
# # Tomelinks
# tl = TomekLinks({'COPD':450})
# fea_Matrix, lab_Cla = tl.fit_resample(all_fea, all_lab)

# # NearMiss
# tl = NearMiss({'COPD':450})
# # 采用SMOTE算法对少数类样本进行增强后的数据集
# # fea_Matrix, lab_Cla = tl.fit_resample(all_fea, all_lab)
# fea_Matrix, lab_Cla = tl.fit_resample(fea_Matrix, lab_Cla)

# Use GMM
n_clusters=16

model = GaussianMixture(n_components=n_clusters, covariance_type='full')
model.fit(train_temp)
train_fea = model.predict_proba(train_temp)

model.fit(test_temp)
test_fea = model.predict_proba(test_temp)

# #SMOTEENN
# sm = SMOTEENN({'Healthy':80,'Other':80,'URTI':80},random_state=42)
# fea_Matrix, lab_Cla = sm.fit_resample(fea_Matrix, lab_Cla)
# # print('Resampled dataset shape %s' % Counter(y_res))

# Prepare Data
# X_train, X_test, y_train, y_test = train_test_split(all_fea, all_lab, test_size=0.2,random_state=0)

# Extending MFCC-related operations
X_train=train_fea
y_train=train_lab
X_test=test_fea
y_test=test_lab

# Normalization
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

# Standardization
# std = StandardScaler()
# X_train = std.fit_transform(X_train)
# X_test = std.fit_transform(X_test)

# #LDA
model=LinearDiscriminantAnalysis()
model.fit(X_train, y_train)

# #OneVsResrClassifier
# model = OneVsRestClassifier(svm.LinearSVC(random_state = 0, verbose = 1,max_iter=10000))
# model.fit(X_train, y_train)

# 1、AUC
y_score = model.decision_function(X_test)   #svm
y_test_oh = label_binarize(y_test, classes=['COPD', 'Healthy', 'Other', 'URTI'])

# 2. Prepare for Confusion Matrix
y_pred = model.predict(X_test) # svm

# 3. Model Evaluation
print()
print('------------------------------------')
print('roc_auc_score：', roc_auc_score(y_test_oh, y_score, average='micro'))  #svm
print('accuracy_score：', accuracy_score(y_test, y_pred))  #svm
print('precision_score: ',precision_score(y_test, y_pred,average='micro'))
print('recall_score: ',recall_score(y_test, y_pred,average='micro'))
print('f1_score: ',f1_score(y_test, y_pred,average='micro'))
print('------------------------------------')
print()

# 4. Model Report
target_names=['COPD', 'Healthy', 'Other', 'URTI']
print(classification_report(y_test, y_pred,digits=2,target_names=target_names))
