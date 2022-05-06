import path
import numpy as np
import scipy.io as scio

path.init()

train_list = path.find_files(".\\test_wav",".wav")

label_list = path.get_label_list()

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

mfc_list = []
for file in train_list:
    mfc_list.append([find_label(file[1]),'mfcc\\'+file[1]+'.mfc'])

mfc_list = np.array(mfc_list,np.object)

scio.savemat('testingfile_list.mat', {'testingfile': mfc_list})

pass