import numpy as np
import struct

# return nSamples,dim and mfcc of file(.mfc) in {path}
def get_mfcc(path):
    mfcfile = open(path, 'rb')
        
    nSamples = int.from_bytes(mfcfile.read(4), 'big')
    sampPeriod = int.from_bytes(mfcfile.read(4), 'big')*1E-7
    sampSize = int.from_bytes(mfcfile.read(2), 'big')
    dim = int(0.25*sampSize)
    parmKind = int.from_bytes(mfcfile.read(2), 'big')

    # get MFCCs
    features = []
    temp = mfcfile.read(4)
    while temp:
        features.append(struct.unpack_from('>f', temp)[0])
        temp = mfcfile.read(4)

    features = np.array(features)
    features = np.reshape(features, (dim, nSamples), order="F")

    mfcfile.close()

    return nSamples, dim, features