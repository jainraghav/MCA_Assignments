import sys
import numpy as np
import cv2
import os
from skimage.color import rgb2gray
from skimage.feature import blob_doh
from skimage.feature import peak_local_max
import matplotlib.pyplot as plt

imagepath = "../../images/all_souls_000001.jpg"
print(imagepath)
img = cv2.imread(imagepath,0)
h,w = img.shape
print(h,w)

def SURF(image):
	#Determinant of Hessian
	image_gray = rgb2gray(image)
	blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)
	return blobs_doh

result_matrix = SURF(img)
print(result_matrix)