import mfc
import path
import scipy.io as scio
from tqdm import tqdm
import numpy as np

path.init()

file_list = path.find_files(".\\mfcc_mel\\mfcc",".mfc")

MFCCs = np.zeros((len(file_list),2), dtype = np.object)

for i in tqdm(range(len(file_list)),position=1,bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}<{remaining}]'):
    nSamples, dim, features = mfc.get_mfcc(file_list[i,0])
    MFCCs[i,0] = features
    MFCCs[i,1] = path.find_label(file_list[i,1])

scio.savemat('mfcc_mel.mat', {'MFCCs': MFCCs})