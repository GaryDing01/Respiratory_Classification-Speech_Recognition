import numpy as np
import wave
import nextpow2
import math
from scipy.io import wavfile

def generate(path,name):

    f = wave.open(path)
    params = f.getparams()
    f.close()

    fs, x = wavfile.read(path)

    len_ = 20 * fs // 1000
    len1 = int(len_ * 0.5)
    len2 = len_ - len1

    Expnt = 2.0
    beta = 0.02
    G = 0.9
    win = np.hamming(len_)
    winGain = len2 / sum(win)
    
    # to 2^n
    nFFT = 2 * 2 ** (nextpow2.nextpow2(len_))
    noise_mean = np.zeros(nFFT)
    
    j = 0
    for k in range(1, 6):
        noise_mean = noise_mean + abs(np.fft.fft(win * x[j:j + len_], nFFT))
        j = j + len_

    # estimated noise spectrum
    noise_mu = noise_mean / 5
    
    k = 1
    img = 1j
    x_old = np.zeros(len1)
    Nframes = len(x) // len2 - 1
    xfinal = np.zeros(Nframes * len2)
    
    for n in range(0, Nframes):
        insign = win * x[k-1:k + len_ - 1]
        spec = np.fft.fft(insign, nFFT)
        sig = abs(spec)
    
        theta = np.angle(spec)
        SNRseg = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)
      
        # berouti function
        def berouti(SNR):
            if -5.0 <= SNR <= 20.0:
                a = 4 - SNR * 3 / 20
            else:
                if SNR < -5.0:
                    a = 5
                if SNR > 20:
                    a = 1
            return a   

        alpha = berouti(SNRseg)

        # specsub
        sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt

        for i in range(len(sub_speech)):
            min_spec = beta * noise_mu[i] ** Expnt
            if sub_speech[i]<min_spec:
                sub_speech[i] = min_spec

        sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
        x_phase = (sub_speech ** (1 / Expnt)) * (np.array([math.cos(x) for x in theta]) + img * (np.array([math.sin(x) for x in theta])))

        # IFFT
        xi = np.fft.ifft(x_phase).real
        
        xfinal[k-1:k + len2 - 1] = x_old + xi[0:len1]
        x_old = xi[0 + len1:len_]
        k = k + len2

    wf = wave.open(name+'_specsub.wav', 'wb')
    wf.setparams(params)
    wave_data = (winGain * xfinal).astype(np.short)
    wf.writeframes(wave_data.tostring())
    wf.close()