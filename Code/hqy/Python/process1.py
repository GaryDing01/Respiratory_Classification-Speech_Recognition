# coding:utf-8

import re
import os

# Get all files under the path of the source folder
sourceFileDir = 'D:\\桌面\\语音识别\\大项目\\archive\\Respiratory_Sound_Database\\Respiratory_Sound_Database\\处理\\txt\\'
filenames = os.listdir(sourceFileDir)
print(filenames)
# print(file)
# Iterate through the file
times_list=[]
crackle_list=[]
wheeze_list=list()
times=0
crackle=0
wheeze=0
before=101
now=101
for filename in filenames:
    filepath = sourceFileDir+'\\'+filename
    # print(filename[0:3])
    # Traverse a single file, read lines, write content
    before=now
    now_str=filename[0:3]
    print(now_str)
    now=int(now_str)
    if before<now:
        times_list.append(times)
        crackle_list.append(crackle)
        wheeze_list.append(wheeze)
        times,crackle,wheeze=0,0,0
    for line in open(filepath):
        line_list=line.split()
        times+=1
        crackle+=int(line_list[-2])
        wheeze+=int(line_list[-1])
        print(line_list)
times_list.append(times)
crackle_list.append(crackle)
wheeze_list.append(wheeze)
print(times_list)
print(len(times_list))
print(crackle_list)
print(len(crackle_list))
with open("test.txt", "a") as f:
    for i in range(126):
        str_=str(times_list[i])+ ' '+str(crackle_list[i])+' '+str(wheeze_list[i])+'\n'
        # print(type(str_))
        f.write(str_)

