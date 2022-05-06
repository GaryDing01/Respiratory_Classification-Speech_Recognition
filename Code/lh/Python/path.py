from contextlib import nullcontext
import os
import numpy as np
import csv
import scipy.io as scio

infilelist = []
# for infile in infilelist
# infile[0]: path of .wav
# infile[1]: name of .wav (without suffix_name) 

label_list = []

#################################################################################
# init list 
def init():
    global infilelist
    global label_list

    if os.path.exists('infilelist.npy'):
        infilelist = np.load('infilelist.npy')

    else:
        inpath = '.\\output'
        infilelist = find_files(inpath)
        np.save('infilelist.npy',infilelist)

    csv_file = csv.reader(open('respiratory-sound-database\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\patient_diagnosis.csv'))
    
    for label in csv_file:
        label_list.append(label)

    label_list = np.array(label_list)

    pass

#################################################################################
# return list of all .wav
def get_infilelist():
    global infilelist
    return infilelist

#################################################################################
# return list of all .wav
def get_label_list():
    global label_list
    return label_list

#################################################################################
# find all .wav in path {target_dir}
def find_files(target_dir,target_suffix_dot):
    find_res = []

    for root_path, dirs, files in os.walk(target_dir):
            for file in files:
                file_name, suffix_name = os.path.splitext(file)
                if suffix_name == target_suffix_dot:
                    find_res.append([os.path.join(root_path, file),file_name])

    find_res = np.array(find_res) 

    return find_res

#################################################################################
# generate .mat of all .mfc
def generate_training_file_list_mat():
    global infilelist
    mfc_list = []
    for file in infilelist:
        mfc_list.append([find_label(file[1]),'mfcc\\'+file[1]+'.mfc'])

    mfc_list = np.array(mfc_list,np.object)

    scio.savemat('trainingfile_list.mat', {'trainingfile': mfc_list})
    

#################################################################################
def find_label(name):
    global label_list

    patient = name[0:3]

    for info in label_list:
        if info[0] == patient:
            if info[1] == 'URTI':
                return 1
            elif info[1] == 'COPD':
                return 2
            elif info[1] == 'Healthy':
                return 3
            else:
                return 4

