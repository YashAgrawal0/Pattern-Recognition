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
	length = 0
	for i in tmp.readlines():
		length+=1

	f = open(fileName,"r")
	cnt = 0
	for i in f.readlines():
		if cnt == 0.75*length:
			break
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

class1_datapoints = input("DATA/LS/class1.txt")
class2_datapoints = input("DATA/LS/class2.txt")
class3_datapoints = input("DATA/LS/class3.txt")
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


def plotData(class_datapoints,color):
	class_datapointsX,class_datapointsY = zip(*class_datapoints)
	plt.scatter(class_datapointsX,class_datapointsY,c=color)

def plotW(w,l,r,c):
	x,y = [],[]

	for i in range(l,r):
		x.append(i)
		val = float(-1*w[0]*i + c)
		v = float(val/w[1])
		y.append(v)

	plt.plot(x,y,linewidth=2)

def findc1(w,p,q):
	val = float(w[0]*q - w[1]*p)
	v = float(val/w[0])
	return v

def findab(w,p,q,c):
	val = float(w[0]*(c - findc1(w,p,q)*w[1]))
	a = float(val/((w[0]*w[0])+(w[1]*w[1])))
	v = float(c - w[0]*a)
	b = float(v/w[1])
	return a,b

def findb(w,x,c):
	val = float(-1*w[0]*x + c)
	v = float(val/w[1])
	return v


def plotProjection(w,data1,data2):
	for i in range(len(data1)):
		a,b = findab(w,data1[i][0],data1[i][1],c)
		# a = a + 15
		# b = findb(w,a,c)
		plt.plot([a],[b],marker = 'o',markersize=4,c="chocolate")
		plt.plot([a,data1[i][0]],[b,data1[i][1]],linewidth=0.1,c="chocolate")

	for i in range(len(data2)):
		a,b = findab(w,data2[i][0],data2[i][1],c)
		# a = a + 10
		# b = findb(w,a,c)
		plt.plot([a],[b],marker = 'o',markersize=4,c="indigo")
		plt.plot([a,data2[i][0]],[b,data2[i][1]],linewidth=0.1,c="indigo")

	plotData(data1,"yellow")
	plotData(data2,"blue")

l = int(sys.argv[1])
r = int(sys.argv[2])
c = int(sys.argv[3])

w1[0] = -1*w1[0]
w2[0] = -1*w2[0]
w3[0] = -1*w3[0]

plotW(w2,l,r,c)  #ls(-5,30)  NLS(-6,6)
plotProjection(w2,class2_datapoints,class3_datapoints)
plt.savefig("test.png",bbox_inches="tight", pad_inches=0.5)
plt.show()