# -*- coding:utf-8 -*-

"""
作者: Gary Ding
日期: 2021年12月09日
"""

import csv
import random

import numpy as np
from shutil import copy

# load patient-situation.csv
def patientSitu(file):
    with open(file, 'r', encoding='utf-8') as pS:
        reader = csv.reader(pS)
        result_List = list(reader)

        # Preprocessing, only the list of patients and information on their condition is left
        result_Array=np.array(result_List)
        for i in range(6):
            result_Array=np.delete(result_Array,1,axis=1)

    return result_Array.tolist() # return patient situation list

# split train and test set
import os

def splitTrainTest(folder,save_train,save_test):
    filenum=len(os.listdir(folder))
    testnum=filenum//5
    print(testnum)
    trainnum=filenum-testnum
    print(trainnum)
    name_list=list(name for name in os.listdir(folder))
    random_name_list=list(random.choice(name_list) for _ in range(testnum))
    for filename in os.listdir(folder):
        if filename in random_name_list:
            if not os.path.isdir(save_test):
                os.makedirs(save_test)
            copy(os.path.join(folder,filename), save_test)
        else:
            if not os.path.isdir(save_train):
                os.makedirs(save_train)
            copy(os.path.join(folder,filename), save_train)

# Convert audio to tags
def wav2label(folder,pSList):
    t_list=[]
    for file in os.listdir(folder):
        for key in range(101,227):
            if str(key) in file:
                t_list.append(pSList[key-101])
                break
    return t_list

def writeTrTecsv(t_list,csv_path):
    for i in range(len(t_list)):
        with open(csv_path, 'a+', newline='') as cp:
            csv_writer = csv.writer(cp)
            csv_writer.writerow(t_list[i])


# Obtain train set labels and test set labels
if __name__=="__main__":
    root_pS="patient_situation.csv"
    pSList=patientSitu(root_pS)
    pSList[0][0]="101" # Special data preprocessing

    root_Patient=r"D:\DCH\Tongji\DaSanShang\Speech_Recognition\Assignment_Final\Week14\DCH\output"
    splitTrainTest(root_Patient,"train_wav","test_wav")
    train_list=wav2label("train_wav",pSList)
    test_list=wav2label("test_wav",pSList)

    # print(train_list)
    # print(test_list)
    # print(len(os.listdir("train_wav")))
    # print(len(os.listdir("test_wav")))

    writeTrTecsv(train_list,"train_label.csv")
    writeTrTecsv(test_list, "test_label.csv")


