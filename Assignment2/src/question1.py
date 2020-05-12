import os
import cv2
import scipy
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import matplotlib.pyplot as plt

class Spectrogram:
	def __init__(self, window_len):
		self.window_len = window_len

	def load(self, path):
		fs, data = wavfile.read(path)
		self.sample_rate = fs
		self.audio_data = data
		self.duration = len(self.audio_data)/fs
	def pad(self, data):
		n = len(data)
		n = 2**(int(np.log2(n))+1)
		if n==len(data):
			return data
		new_data = np.zeros(n)
		new_data[:len(data)] = data
		return new_data
	def fft(self, data):
		data = np.asarray(data,dtype=float)
		nmin = 32
		n = np.arange(nmin)
		k = n[:, None]
		M = np.exp(-2j * np.pi * n * k / nmin)
		X = np.dot(M, data.reshape((nmin, -1)))
		while X.shape[0] < len(data):
			X_even = X[:, :int(X.shape[1]/2)]
			X_odd = X[:, int(X.shape[1]/2):]
			factor = np.exp(-1j * np.pi * np.arange(X.shape[0])/ X.shape[0])[:, None]
			X = np.vstack([X_even + factor * X_odd, X_even - factor * X_odd])
		return np.abs(X.ravel())

	def fspec(self, path):
		self.load(path)
		self.audio_data = self.pad(self.audio_data)
		fft_op = self.fft(self.audio_data)
		# print(fft_op)
		# print(np.fft.fft(self.audio_data))
		nooverlap = self.window_len/2
		st = np.arange(0,len(self.audio_data),nooverlap,dtype=int)
		st = st[st+self.window_len < len(self.audio_data)]
		collect = []
		for si in st:
			collect.append(self.fft(self.audio_data[si:si+self.window_len]))
		spc = np.array(collect).T
		spc = np.where(spc==0, 0.00001, spc)
		self.spec = 10*np.log10(spc)

def main():
	
	training_dir = "training/"
	saved_specs = "specs/"

	classes = os.listdir(training_dir)
	for classi in tqdm(classes):
		classfiles = os.listdir(training_dir+classi)
		p1 = Spectrogram(128)
		for file in tqdm(classfiles):
			filepath = training_dir+classi+"/"+file
			# print(filepath)
			p1.fspec(filepath)
			savename = saved_specs+classi+"/"+file[:-4]+".npy"
			# print(savename)
			np.save(savename,p1.spec)

if __name__ == '__main__':
    main()