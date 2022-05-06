# -*- coding:utf-8 -*-

"""
作者: Gary Ding
日期: 2021年12月24日
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

# split train test set
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

def newsplitTrainTest(folder,save_train,save_test,pSList):
    # Statistics
    num_COPD=793
    num_Healthy=35
    num_Other=69
    num_URTI=23

    test_COPD=num_COPD//5
    test_Healthy=num_Healthy//5
    test_Other=num_Other//5
    test_URTI=num_URTI//5

    i_COPD=0
    i_Healthy=0
    i_Other=0
    i_URTI=0

    for filename in os.listdir(folder):
        pSituation="Still"

        for key in range(101, 227):
            if str(key) in filename:
                pSituation=pSList[key - 101][1]

        if 'COPD' == pSituation:
            if i_COPD<test_COPD:
                if not os.path.isdir(save_test):
                    os.makedirs(save_test)
                copy(os.path.join(folder, filename), save_test)
                i_COPD+=1
            else:
                if not os.path.isdir(save_train):
                    os.makedirs(save_train)
                copy(os.path.join(folder, filename), save_train)
        elif 'Healthy' == pSituation:
            if i_Healthy<test_Healthy:
                if not os.path.isdir(save_test):
                    os.makedirs(save_test)
                copy(os.path.join(folder, filename), save_test)
                i_Healthy+=1
            else:
                if not os.path.isdir(save_train):
                    os.makedirs(save_train)
                copy(os.path.join(folder, filename), save_train)
        elif 'Other' == pSituation:
            if i_Other<test_Other:
                if not os.path.isdir(save_test):
                    os.makedirs(save_test)
                copy(os.path.join(folder, filename), save_test)
                i_Other+=1
            else:
                if not os.path.isdir(save_train):
                    os.makedirs(save_train)
                copy(os.path.join(folder, filename), save_train)
        elif 'URTI' == pSituation:
            if i_URTI<test_URTI:
                if not os.path.isdir(save_test):
                    os.makedirs(save_test)
                copy(os.path.join(folder, filename), save_test)
                i_URTI+=1
            else:
                if not os.path.isdir(save_train):
                    os.makedirs(save_train)
                copy(os.path.join(folder, filename), save_train)

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
    newsplitTrainTest(root_Patient, "train1_wav", "test1_wav",pSList)
    train_list=wav2label("train1_wav",pSList)
    test_list=wav2label("test1_wav",pSList)

    writeTrTecsv(train_list,"train1_label.csv")
    writeTrTecsv(test_list, "test1_label.csv")


