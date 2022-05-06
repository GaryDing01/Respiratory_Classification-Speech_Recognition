import path
import mfc
from tqdm import tqdm
import numpy

trim_list = path.find_files(".\\mfcc_trim\\mfcc",".mfc")
spec_list = path.find_files(".\\mfcc_specsub\\mfcc",".mfc")

for i in tqdm(range(len(trim_list)),position=1,bar_format='{l_bar}|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}{postfix}|{elapsed}<{remaining}]'):
    nSamples, dim, trim = mfc.get_mfcc(trim_list[i,0])
    nSamples, dim, spec = mfc.get_mfcc(spec_list[i,0])

    ans = spec[:,0:trim.shape[1]]

    numpy.save("mfcc_spec_trim\\"+trim_list[i,1]+".npy",ans)

    pass