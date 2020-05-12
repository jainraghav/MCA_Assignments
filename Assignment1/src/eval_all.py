import sys
import numpy as np
import os
import csv
import math

qgt = {}

querypath = "../../train/query/"
gtpath = "../../train/ground_truth/"
q1_feats = "../../q1_feats/"
q2_feats = "../../q2_feats/"
q3_feats = "../../q3_feats/"

all_queries = os.listdir(querypath)
all_gt = os.listdir(gtpath)
for q in all_queries:
	qgt[q[:-10]] = []
	for xx in all_gt:
		if xx.startswith(q[:-10]):
			qgt[q[:-10]].append(xx)

query_input = {}
for q in all_queries:
	with open(querypath+q) as csvfile:
		spamreader = csv.reader(csvfile, delimiter=' ')
		for row in spamreader:
			query_input[q] = row[0]
#print(query_input)

def distance(x,y):
  return sum(abs(a-b)/(1+a+b) for a, b in zip(x, y))

all_traindb = os.listdir(q1_feats)
loaded_feats = []
for ft in all_traindb:
	loaded_feats.append([ft, np.load(q1_feats+ft)])
# print(loaded_feats)

def dist(val):
	return val[2]

import time
times = []
for x,y in query_input.items():
	st_time = time.time()
	print(x)
	gtfiles = qgt[x[:-10]]
	gtimages = []
	for file in gtfiles:
		gt1=[]
		with open(gtpath+file) as csvfile:
			spamreader = csv.reader(csvfile)
			for row in spamreader:
				gt1.append(row[0])
		gtimages.append(gt1)


	query_feat = np.load(q1_feats+y[5:]+'.npy')
	# print("Checking feat size: ",query_feat.shape)

	for i in range(len(loaded_feats)):
		d1 = distance(query_feat[0],loaded_feats[i][1][0])
		d2 = distance(query_feat[1],loaded_feats[i][1][1])
		d3 = distance(query_feat[2],loaded_feats[i][1][2])
		finald = d1+d2+d3
		loaded_feats[i].append(finald)

	loaded_feats.sort(key = dist)
	#print("Results: ")

	index_division = [0,200,500,5063]

	et_time = time.time() -st_time
	times.append(et_time)

	for i in range(len(index_division)-1):
		index_range = (index_division[i],index_division[i+1])
		gtimages_temp = gtimages[i]
		cc=0
		for j in range(index_range[0],index_range[1]):
			if loaded_feats[j][0][:-4] in gtimages_temp:
				cc+=1
		total = index_division[i+1] - index_division[i]
		precision = cc/total
		#print("Precision"+str(i+1)+" : ", precision)
		recall = cc/len(gtimages_temp)
		#print("Recall"+str(i+1)+" : ", recall)

	#print("Average Precision: ")
	ccf=0
	for x in loaded_feats:
		if x[0][:-4] in gtimages[0] or x[0][:-4] in gtimages[1] or x[0][:-4] in gtimages[2]:
			ccf+=1
	lensum = len(gtimages[0]) + len(gtimages[1]) + len(gtimages[2])
	precision = ccf/5063
	#print("Overall Precision"+str(i+1)+" : ", precision)
	recall = ccf/lensum
	#print("Overall Recall"+str(i+1)+" : ", recall)
	#print("================================================")

print("Average time taken for a query: ", sum(times)/33)
	





