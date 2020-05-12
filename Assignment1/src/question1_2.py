import sys
import numpy as np
import cv2
import os
from tqdm import tqdm
from pylab import *
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.ndimage import filters
from scipy import spatial

def formula1(x,sigma):
	return -(x*x/(2.*(sigma**2)))

def formula2(x_filter,y_filter,x,y,sigma):
	return (-(2*sigma**2) + (x**2 + y**2) ) * (x_filter*y_filter) * (1/(2*np.pi*sigma**4))

def calc_new(slice_img,k,sigma)
	indexmax = slice_img.argmax()
	z,x,y = np.unravel_index(indexmax,slice_img.shape)
	new_x,new_y = i+x-1,j+y-1
	new_std = k**z*sigma
	return new_x,new_y,new_std

def LoG_convolve(img):
	log_images = []
	for i in range(0,9):
		y = np.power(k,i) 
		sigma_1 = sigma*y
		n = np.ceil(sigma_1*6)

		y,x = np.ogrid[-n//2:n//2+1,-n//2:n//2+1]
		y_filter = np.exp(formula1(y,sigma_1))
		x_filter = np.exp(formula1(x,sigma_1))
		filter_log = formula2(x_filter,y_filter,x,y,sigma_1)

		image = cv2.filter2D(img,-1,filter_log)
		image = np.pad(image,((3,3),(3,3)),'constant')
		image = np.square(image)
		log_images.append(image)
	log_image_list = [i for i in log_images]
	return np.array(log_image_list)

def detect_blob(log_image_np):
	co_ordinates = []
	directions=[-1,-2,0,1,2,3]
	for i in range(1,img.shape[0]):
		for j in range(1,img.shape[1]):
			slice_img = log_image_np[:,i+directions[0]:i+directions[4],j+directions[0]:j+directions[4]]
			if np.amax(slice_img) >= .05: #threshold
				new_x,new_y,new_std = calc_new(slice_img,k,sigma)
				co_ordinates.append((new_x,new_y,new_std))
	
	#Remove redundant blobs using thresholding
	#Check overlap and remove smaller
	return co_ordinates


imagedirpath = "../../images/"
savedirpath = "../../q2_feats/"

donefeats = os.listdir(savedirpath)
donefeats = [x[:-4] for x in donefeats]
print(donefeats)

k = 1.414
sigma = 1.0

for image in tqdm(os.listdir(imagedirpath)):
	if image[:-4] not in donefeats:
		imagepath = imagedirpath + image
		#print(imagepath)

		img = cv2.imread(imagepath,0)
		img = img/255.0

		log_image_np = LoG_convolve(img)
		co_ordinates = list(set(detect_blob(log_image_np)))
		print(len(co_ordinates))

		savepath = savedirpath + image[:-4]
		np.save(savepath,np.array(co_ordinates))
