import path
import numpy as np
import scipy.io as scio
import specsub
from tqdm import tqdm
import mfcc

path.init()

file_list = path.find_files(".\\respiratory-sound-database\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\audio_and_txt_files",".wav")

specsub.generate(file_list[4,0],file_list[4,1])

pass