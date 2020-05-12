import os
import pickle
import sklearn
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from question2 import MFCC 
from question1 import Spectrogram 

labelmap = {"zero":0, "one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9}

validation_dir = "validation/"

def eval(dir, model='mfcc'):
	classes = os.listdir(validation_dir)

	if model == 'mfcc':
		p1 = MFCC(25,10,80)
		loaded_model = pickle.load(open('svm_mfcc.sav', 'rb'))
	elif model == 'spec':
		p1 = Spectrogram(128)
		loaded_model = pickle.load(open('svm_spec.sav', 'rb'))
	else:
		return "Wrong model"

	X_val=[]
	y_val=[]
	for cl in classes:
		files = os.listdir(validation_dir+cl)
		for fl in files:
			filepath = validation_dir+cl+"/"+fl

			if model == 'mfcc':
				p1.feats(filepath)
				feati = p1.mfcc_feat
				if feati.shape[1]==66:
					X_val.append(feati.ravel())
					y_val.append(labelmap[cl])
			else:
				p1.fspec(filepath)
				feati = p1.spec
				if feati.shape[1]==254:
					X_train.append(feati.ravel())
					y_train.append(labelmap[cl])

			

	X_val = np.array(X_val)
	y_val = np.array(y_val)
	print(X_val.shape)
	print(y_val.shape)

	scaler = StandardScaler()
	X_val_scaled = scaler.fit_transform(X_val)
	pca = PCA(n_components=120).fit(X_val_scaled)
	X_val_pca = pca.transform(X_val_scaled)

	X_pred = loaded_model.predict(X_val)

	acc = accuracy_score(X_pred,y_val)
	prec = precision_score(X_pred,y_val,average='micro')
	rec = recall_score(X_pred,y_val,average='micro')
	f1 = f1_score(X_pred,y_val,average='micro')

	return (acc,prec,rec,f1)

print("Results of SVM (MFCC feats)")
results = eval(validation_dir,model='mfcc')
print("Accuracy: ",results[0])
print("Precision: ",results[1])
print("Recall: ",results[2])
print("F1-Score: ",results[3])
# print("Results of SVM (Spectrogram feats)")
# print(eval(validation_dir,model='spec'))