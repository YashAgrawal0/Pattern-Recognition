import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import random
import sys

dim = 0
class1_datapoints,class2_datapoints,class3_datapoints = [],[],[]

#===============================================
def input(fileName):
	data_points = []
	tmp = open(fileName,"r")
	
	f = open(fileName,"r")
	for i in f.readlines():
		x = [float(j) for j in i.split()]
		data_points.append(x)

	return np.array(data_points)

def inputTestData(fileName):
	data_points = []
	tmp = open(fileName,"r")
	length = 0

	for i in tmp.readlines():
		length+=1

	f = open(fileName,"r")
	cnt = 0
	for i in f.readlines():
		if cnt >= 0.75*length:
			x = [float(j) for j in i.split()]
			data_points.append(x)
		cnt = cnt + 1
	return np.array(data_points)	
#==============================================	

def findMean(data):
	mean = []
	for i in range(len(data[0])):
		mu = 0.0
		for j in range(len(data)):
			mu = mu + data[j][i]
		mu = mu/len(data)	
		mean.append(mu)	
	return np.array(mean)

#================================================

def output(w,data,name):
	file = open(name,"w")
	for i in range(len(data)):
		val = np.matmul(w,data[i])
		print >> file, val

class1_datapoints = input("DATA/BOVW/train_class1.txt")
class2_datapoints = input("DATA/BOVW/train_class2.txt")
class3_datapoints = input("DATA/BOVW/train_class3.txt")
mean1 = findMean(class1_datapoints)
mean2 = findMean(class2_datapoints)
mean3 = findMean(class3_datapoints)
cov1 = np.cov(np.transpose(class1_datapoints))
cov2 = np.cov(np.transpose(class2_datapoints))
cov3 = np.cov(np.transpose(class3_datapoints))

#======================================  FOR class 1 & 2 ================
Inverse = np.linalg.inv(cov1 + cov2)
w1 = np.matmul(Inverse,mean1-mean2)
#======================================  FOR class 2 & 3 ================
Inverse = np.linalg.inv(cov3 + cov2)
w2 = np.matmul(Inverse,mean2-mean3)
#======================================  FOR class 1 & 3 ================
Inverse = np.linalg.inv(cov1 + cov3)
w3 = np.matmul(Inverse,mean1-mean3)
#========================================================================
	
# #								OUTPUT
	
output(w1,class1_datapoints,"c1_FDA12.txt")
output(w1,class2_datapoints,"c2_FDA12.txt")
output(w2,class2_datapoints,"c2_FDA23.txt")
output(w2,class3_datapoints,"c3_FDA23.txt")
output(w3,class1_datapoints,"c1_FDA13.txt")
output(w3,class3_datapoints,"c3_FDA13.txt")

# #========================================================================


class1_Testdatapoints = input("DATA/BOVW/test_class1.txt")
class2_Testdatapoints = input("DATA/BOVW/test_class2.txt")
class3_Testdatapoints = input("DATA/BOVW/test_class3.txt")

output(w1,class1_Testdatapoints,"c1w1.txt")
output(w2,class1_Testdatapoints,"c1w2.txt")
output(w3,class1_Testdatapoints,"c1w3.txt")
output(w1,class2_Testdatapoints,"c2w1.txt")
output(w2,class2_Testdatapoints,"c2w2.txt")
output(w3,class2_Testdatapoints,"c2w3.txt")
output(w1,class3_Testdatapoints,"c3w1.txt")
output(w2,class3_Testdatapoints,"c3w2.txt")
output(w3,class3_Testdatapoints,"c3w3.txt")