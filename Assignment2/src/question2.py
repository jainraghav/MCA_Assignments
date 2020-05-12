import os
import scipy
import numpy as np
from tqdm import tqdm
from scipy import fftpack
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

class MFCC:
    def __init__(self, flen, fshift, num_melbins):
        self.flen = flen
        self.fshift = fshift
        self.nmbins = num_melbins
    
    def load(self,path):
        sample_freq, raw_seq = wavfile.read(path)
        self.sf = sample_freq
        seq = raw_seq.astype(np.float64)
        self.seq = seq - np.mean(seq)

    def preemphasis(self):
        term_add = self.seq[1:] - 0.97 * self.seq[:-1]
        self.seq = np.append(self.seq[0], term_add)

    def freq_to_mel(self,freq):
        return 1125.0 * np.log(1.0 + freq / 700.0)

    def mel_to_freq(self,mel):
        return 700.0 * (np.exp(mel / 1125.0) - 1.0)

    def iter_bin(self, out, curr_bin, next_bins, backward=False):
        next_bin = next_bins[np.where(next_bins > curr_bin)][0]
        for f in range(int(curr_bin), int(next_bin)):
            if backward:
                out[f] = -1*(f - next_bin) / (next_bin - curr_bin)
            else:
                out[f] = (f - curr_bin) / (next_bin - curr_bin)

    def mel_filterbank(self):
        num_fft = (self.spec.shape[0] - 1) * 2
        lowf = 20
        highf = self.sf // 2
        low_mel = self.freq_to_mel(lowf)
        high_mel = self.freq_to_mel(highf)
        banks = np.linspace(low_mel, high_mel, self.nmbins + 2)
        bins = np.floor((num_fft + 1) * self.mel_to_freq(banks) / self.sf)
        out = np.zeros((self.nmbins, num_fft // 2 + 1))
        for b in range(self.nmbins):
            self.iter_bin(out[b], bins[b], bins[b+1:])
            self.iter_bin(out[b], bins[b+1], bins[b+2:], backward=True)
        return out

    def feats(self,path):
        self.load(path)
        self.preemphasis()
        samples = self.sf // 1000
        window = signal.get_window("hamming", self.flen*samples)
        b, a, spectrogram = signal.spectrogram(self.seq, self.sf, window=window, noverlap=self.fshift*samples, mode="psd")
        self.spec = spectrogram
        banks = self.mel_filterbank()
        fbank_spect = np.dot(banks, self.spec)
        fbank_spect[np.where(fbank_spect == 0)] = np.finfo(dtype=fbank_spect.dtype).eps
        logfbank_spect = np.log(fbank_spect)

        dct_feat = fftpack.dct(logfbank_spect, type=2, axis=0, norm="ortho")
        dct_feat = dct_feat[:13]
        lifter = 1 + 11.0 * np.sin(np.pi * np.arange(13) / 22)
        self.mfcc_feat = lifter[:, np.newaxis] * dct_feat

def main():

    training_dir = "training/"
    saved_mfccs = "mfcc/"

    #frame_length, frame_shift, num_melbins
    p1 = MFCC(25,10,80)
    classes = os.listdir(training_dir)
    for classi in tqdm(classes):
        classfiles = os.listdir(training_dir+classi)
        for file in tqdm(classfiles):
            filepath = training_dir+classi+"/"+file
            # print(filepath)
            p1.feats(filepath)
            savename = saved_mfccs+classi+"/"+file[:-4]+".npy"
            # print(savename)
            # print(p1.mfcc_feat.shape)
            np.save(savename,p1.mfcc_feat)

if __name__ == '__main__':
    main()