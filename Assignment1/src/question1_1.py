import sys
import numpy as np
import cv2
from tqdm import tqdm
import os

def correlogram(photo, Cm, K):
    X, Y, t = photo.shape
    colorsPercent = []

    for k in K:
        countColor = 0
        color = [0]*len(Cm)
        
        max_xlimit = int(round(X / 10))
        max_ylimit = int(round(Y / 10))

        for x in range(0, X, max_xlimit):
            for y in range(0, Y, max_ylimit):

                Ci = photo[x][y]
                dist = k
                points = ((x + dist, y + dist), (x + dist, y), 
                    (x + dist, y - dist), (x, y - dist), (x - dist, y - dist), 
                    (x - dist, y), (x - dist, y + dist), (x, y + dist))

                Cn = []
                for i in points:
                    if i[0]<0 or i[0]>=X or i[1]<0 or i[1]>=Y:
                        continue
                    else:
                      Cn.append(i)

                for j in Cn:
                    firsti = j[0]
                    firstj = j[1]
                    Cj = photo[firsti][firstj]
 
                    for m in range(len(Cm)):
                        if np.array_equal(Cm[m], Ci):
                            if np.array_equal(Cm[m], Cj):
                                countColor+=1
                                color[m]+=1

        for i in range(len(color)):
            color[i] = float(color[i]) / countColor
        
        colorsPercent.append(color)

    return colorsPercent

def autoCorrelogram(img):

    Z = np.float32(img.reshape((-1, 3)))

    ret, label, center = cv2.kmeans(Z, 64, None, (3, 10, 1), 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res = np.array(res.reshape((img.shape)))

    #Removing Duplicates
    res = res[np.lexsort(res.T)]
    diff = np.diff(res, axis = 0)
    ui = np.ones(len(res), 'bool')
    ui[1:] = (diff != 0).any(axis = 1)
    colors64 = res[ui]

    result = correlogram(res2, colors64, [1,3,5])
    return result

imagedirpath = "../../images/"
savedirpath = "../../q1_feats/"

#Can run anytime (will resume)
donefeats = os.listdir(savedirpath)
donefeats = [x[:-4] for x in donefeats]
print(donefeats)
for image in tqdm(os.listdir(imagedirpath)):
    if image[:-4] not in donefeats:
        imagepath = imagedirpath + image
        print(imagepath)
        img4auto = cv2.imread(imagepath,1)
        matrix = autoCorrelogram(img4auto)
        #print(matrix)
        savepath = savedirpath + image[:-4]
        np.save(savepath,matrix)
