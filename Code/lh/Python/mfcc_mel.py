from scipy.io import wavfile
import numpy as np
import math
import os
import struct
from tqdm import tqdm
import time
import path

def generate_mfcc(batch):

    infilelist = path.get_infilelist()
    outpath = '.\\mfcc'

    wav_total_num = infilelist.shape[0]

    num_divided = int(wav_total_num/10)
    end = min( (batch+1)*num_divided, wav_total_num)

    for index in tqdm(range(batch*num_divided,end),desc=str(batch)+': ',ncols=55,position=batch,
                    bar_format='{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):

        outfile = outpath + '\\' + infilelist[index,1] + '.mfc'

        if os.path.exists(outfile):
            time.sleep(.1)
            continue
        else:
            mfcc = calculate_MFCC(infilelist[index,0])

            file = open(outfile,'wb')

            file.write(struct.pack('>l', mfcc.shape[1]))
            file.write(struct.pack('>l',0))
            file.write(struct.pack('>h',156))
            file.write(struct.pack('>h',0))

            for j in range(mfcc.shape[1]):
                for i in range(mfcc.shape[0]):
                    file.write(struct.pack('>f',mfcc[i,j]))

            file.close()

        pass

#################################################################################
def calculate_MFCC(filepath):
    sampleRate, amplitude = wavfile.read(filepath)
    length = len(amplitude)
    frameWidth = int(0.025*sampleRate)
    frameShift = int(0.01*sampleRate)
    amplitude = [i/32768 for i in amplitude]

    # Pre-emphasis

    emphasisedAmplitude = []

    for index in range(1,length):
        # formula of  pre-emphasis
        temp = amplitude[index]-0.97*amplitude[index-1]
        emphasisedAmplitude.append(temp)

    # windowing

    # Hanning window function
    def getHanning(n, frameWidth):
        return 0.5-0.5*np.cos((2*np.pi*n)/(frameWidth-1))

    # fill with 0
    zeroCount = (len(emphasisedAmplitude)-frameWidth)%frameShift
    for i in range(frameShift-zeroCount):
        emphasisedAmplitude.append(0)

    windowingFrames = []

    # find the start of each frame
    for start in range(0, len(emphasisedAmplitude)-frameWidth+1, frameShift):
        temp = []
        # window the frames
        for index in range(frameWidth):
            temp.append(emphasisedAmplitude[index+start]*getHanning(index, frameWidth))   
        windowingFrames.append(temp)

    # DFT

    # get DFT of frames in abs
    def DFT(frames):
        dft = []
        # get DFT of each frames in plural
        for frame in frames:
            dft.append(np.fft.rfft(frame))

        ans = []
        for nd in dft:
            temp = []
            for x in nd:
                temp.append(abs(x))
            ans.append(temp)
        return ans

    spectrum = np.array(DFT(windowingFrames))

    x = [i*frameShift/sampleRate for i in range(spectrum.shape[0])]
    y = [i*sampleRate/frameWidth for i in range(spectrum.shape[1])]

    # hz to mel
    def melfrequence(x):
        return 1125*math.log(1+x/700)

    # mel to hz
    def melfrequenceInverse(x):
        return 700*(math.pow(math.e, x/1125)-1)

    # function used in mel filterbanks to get bins
    # m：number of the filterbank（started with 0） 
    # M：amount of filterbanks
    def f(m, M):
        # range of frequnce
        fmax = sampleRate/2
        fmin = 300
        return (frameWidth/sampleRate)*melfrequenceInverse(melfrequence(fmin)+m*(melfrequence(fmax)-melfrequence(fmin))/(M+1))

    # mel filterbanks
    # k：x axis
    # m：number of the filterbank（started with 0）  
    # M：amount of filterbanks
    def getMelFilterbank(k, m, M):
        min = f(m, M)
        mid = f(m+1, M)
        max = f(m+2, M)
        if k<min or k>max:
            return 0
        elif min<=k and k<=mid:
            return (k-min)/(mid-min)
        elif mid<k and k<=max:
            return (max-k)/(max-mid)

    # amount of filterbanks
    M = 26
    for i in range(M):
        temp = []
        temp.append(f(i,M))
        temp.append(f(i+1,M))
        temp.append(f(i+2,M))

    # calculate melspectrum
    def getMelSpectrum():
        ans = []
        for frame in spectrum:
            temp = []
            for m in range(M):
                y = 0
                for k in range(spectrum.shape[1]):
                    y += getMelFilterbank(k, m, M)*frame[k]*frame[k]
                temp.append(y)
            ans.append(temp)
        return ans

    melSpectrum = np.array(getMelSpectrum())

    # log()
    energy = []
    for frame in melSpectrum:
        sum = np.finfo("float").eps
        for i in frame:
            sum += i**2 
        energy.append(math.log(sum))
    energy = np.array(energy)

    melSpectrum = np.log(melSpectrum+np.finfo("float").eps)

    final = np.transpose(melSpectrum)

    return final
