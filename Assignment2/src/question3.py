import os
import pickle
import sklearn
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from scipy.io import wavfile
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from question2 import MFCC 
from question1 import Spectrogram 

from random import seed
from random import randint
seed(1)

labelmap = {"zero":0, "one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9}

mfcc_feats_folder = "mfcc/"
spec_feats_folder = "specs/"

augfold = "augment/"

p1 = MFCC(25,10,80)

def train(feat_folder,ft='mfcc'):

	classes = os.listdir(feat_folder)
	if ft=='mfcc':
		modelfilename = 'svm_mfcc_noise.sav'
	elif ft=='spec':
		modelfilename = 'svm_spec.sav'
	else:
		print("Wrong Model!")
		return 0

	X_train=[]
	y_train=[]
	for cl in tqdm(classes):
		files = os.listdir(feat_folder+cl)
		files = files[:len(files)//2]
		for fl in tqdm(files):
			filepath = feat_folder+cl+"/"+fl
			feati = np.load(filepath)
			if ft=='mfcc':
				if feati.shape[1]==66:
					X_train.append(feati.ravel())
					y_train.append(labelmap[cl])
			else:
				if feati.shape[1]==254:
					X_train.append(feati.ravel())
					y_train.append(labelmap[cl])

	#Add Augmented datapoints to the data (50-50)
	for cl in tqdm(classes):
		files = os.listdir(augfold+cl)
		files = files[:len(files)//2]
		for fl in tqdm(files):
			filepath = augfold+cl+"/"+fl
			p1.feats(filepath)
			feati = p1.mfcc_feat
			if ft=='mfcc':
				if feati.shape[1]==66:
					X_train.append(feati.ravel())
					y_train.append(labelmap[cl])
			else:
				if feati.shape[1]==254:
					X_train.append(feati.ravel())
					y_train.append(labelmap[cl])

	X_train = np.array(X_train)
	y_train = np.array(y_train)
	print(X_train.shape)
	print(y_train.shape)

	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	pca = PCA(n_components=240).fit(X_train_scaled)
	X_train_pca = pca.transform(X_train_scaled)

	print(sum(pca.explained_variance_ratio_))

	clf = SVC(kernel = 'rbf', gamma='scale')
	clf.fit(X_train, y_train)

	pickle.dump(clf, open(modelfilename, 'wb'))
	print("Done training!")

# train(spec_feats_folder,'spec')
train(mfcc_feats_folder,'mfcc')