# -*- coding:utf-8 -*-

"""
作者: Gary Ding
日期: 2021年12月19日
"""
import csv

import numpy as np

train_path=r"D:\DCH\Tongji\DaSanShang\Speech_Recognition\Assignment_Final\Week15\DCH\train_fea.csv"
test_path=r"D:\DCH\Tongji\DaSanShang\Speech_Recognition\Assignment_Final\Week15\DCH\test_fea.csv"

# After Denoise
# train_path=r"D:\DCH\Tongji\DaSanShang\Speech_Recognition\Assignment_Final\Week17\DCH\train_fea.csv"
# test_path=r"D:\DCH\Tongji\DaSanShang\Speech_Recognition\Assignment_Final\Week17\DCH\test_fea.csv"

def loadFeaLab(list):
    arr=np.array(list)
    arr_fea=arr[:,:-1]
    list_lab=[]
    for i in range(arr.shape[0]):
        list_lab.append(arr[i][-1])
    return arr_fea,list_lab

with open(train_path, 'r') as tr:
    reader = csv.reader(tr)
    result_train = list(reader)
train_fea,train_lab=loadFeaLab(result_train)

with open(test_path, 'r') as te:
    reader = csv.reader(te)
    result_test = list(reader)
test_fea,test_lab=loadFeaLab(result_test)