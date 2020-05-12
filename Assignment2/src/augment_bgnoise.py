import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from random import seed
from random import randint
seed(1)

bgfolder = "_background_noise_/"
svfold = "augment/"

training_dir = "training/"

def augment(inpt_path,fname,cl):
	value = randint(0,5)
	noises = os.listdir(bgfolder)
	selected = noises[value]
	selected_path = bgfolder + selected
	fs, data = wavfile.read(selected_path)
	fs_o, data_o = wavfile.read(inpt_path)

	if len(data_o)==16000:
		#Augment BG Noise
		data_aug = data_o*np.random.uniform(0.95, 1.25)+data[16000:32000]*np.random.uniform(0.0001, 0.001)
		file_save = svfold + cl + '/' + fname[:-4] + '_' +selected
		# print(file_save)
		wavfile.write(file_save,fs_o,data_aug)

classes = os.listdir(training_dir)
for classi in tqdm(classes):
    classfiles = os.listdir(training_dir+classi)
    for file in tqdm(classfiles):
        filepath = training_dir+classi+"/"+file
        # print(filepath)
        augment(filepath,file,classi)