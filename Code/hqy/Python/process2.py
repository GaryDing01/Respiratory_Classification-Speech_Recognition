# coding:utf-8

import re
import os
from pydub import AudioSegment

# Get all files under the path of the source folder
sourceFileDir = 'D:\\桌面\\语音识别\\大项目\\archive\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\处理\\txt\\'
audioFileDir='D:\\桌面\\语音识别\\大项目\\archive\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\audio\\'
filenames = os.listdir(sourceFileDir)
print(filenames)
# print(file)
# Iterate through the file
file_list=[]
end_list=list()# Audio cut-off time for each segment
times=0#Audio number
for filename in filenames:
    filepath = sourceFileDir+'\\'+filename
    # View Files
    # print(filepath)
    # file_list.append(filepath)
    # Read each row of data
    line_num=int(1)#first row
    breathe_time=2#Number of breaths
    for line in open(filepath):
        #Get deadline
        if line_num!=breathe_time:
            #print(type(line_num))
            line_num+=1
            continue
        else:
            line_list = line.split()
            end=float(line_list[1])
            end_list.append(end)
            break
# Test selection times
tmp_list=[]
for i in range(1,6):
    for filename in filenames:
        filepath = sourceFileDir + '\\' + filename
        # View Files
        # print(filepath)
        # file_list.append(filepath)
        # Read each row of data
        line_num = int(1)  # first row
        breathe_time = i  # Number of breaths
        for line in open(filepath):
            # Get deadline
            if line_num != breathe_time:
                # print(type(line_num))
                line_num += 1
                continue
            else:
                line_list = line.split()
                end = float(line_list[1])
                tmp_list.append(end)
                break
    print(f'至少包含{i}个呼吸周期的音频有{len(tmp_list)}个')
    tmp_list.clear()

