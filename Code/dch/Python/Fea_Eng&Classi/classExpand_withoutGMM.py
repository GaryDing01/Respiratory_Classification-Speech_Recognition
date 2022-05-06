# -*- coding:utf-8 -*-

"""
作者: Gary Ding
日期: 2021年12月20日
"""
import numpy as np
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize, MinMaxScaler, StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks,NearMiss

#综合采样
from imblearn.combine import SMOTEENN,SMOTETomek

import lightgbm as lgb

from loadFeaLab import train_fea,train_lab,test_fea,test_lab
# from loadFeaLab_try2 import train_fea,train_lab,test_fea,test_lab

# Use GMM
# Convert Array from String to float
all_fea=np.vstack((train_fea,test_fea))
all_lab=train_lab+test_lab

fea_Matrix=train_fea
lab_Cla=train_lab

# # Data imbalance processing
# # SMOTE
# sm = SMOTE({'Healthy':500,'Other':500,'URTI':500},random_state=42)
# train_fea, train_lab = sm.fit_resample(train_fea, train_lab)
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
# # fea_Matrix, lab_Cla = tl.fit_resample(all_fea, all_lab)
# fea_Matrix, lab_Cla = tl.fit_resample(fea_Matrix, lab_Cla)


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
# model = OneVsRestClassifier(svm.LinearSVC(random_state = 0, verbose = 1,max_iter=20000))
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
