import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import random
import sys

a = np.zeros(3)
a[0],a[1],a[2] = 1,1,1

def input(fileName):
	length = 0
	temp = open(fileName,"r")
	for i in temp.readlines():
		length += 1
	data = []
	
	f = open(fileName,"r")
	cnt = 0
	Z = []
	for i in f.readlines():
		if cnt == 0.75*length:
			break
		x = [float(j) for j in i.split()]
		y = []
		y.append(1)
		for j in x:
			y.append(j)
		Z.append(y)
		data.append(x)
		cnt = cnt + 1
	return np.array(Z),data


def findMisclassified(x,y):
	Z1 = x.astype(float)
	Z2 = y.astype(float)
	misclassified = []
	global a
	# print a
	for i in range(len(Z1)):
		val = np.matmul(a,Z1[i])
		if val < 0 :
			misclassified.append(Z1[i])
		# print a,Z[i],val
	
	for i in range(len(Z2)):
		val = np.matmul(a,Z2[i])
		if val > 0 :
			misclassified.append(-1*Z2[i])

	return np.array(misclassified)

Z1,data1 = input("DATA/LS/class1.txt")
Z2,data2 = input("DATA/LS/class2.txt")
misclassified = findMisclassified(Z1,Z2)	
ETA = 1
length = 0

data1X,data1Y = zip(*data1)
data2X,data2Y = zip(*data2)

while(len(misclassified)!=0):
	mat = np.zeros(len(misclassified[0]))
	length = len(misclassified)
	# print len(misclassified)
	for i in misclassified:
		mat = mat + i

	mat = ETA*mat
	# ETA = ETA - 0.3*ETA

	a = a + mat
	misclassified = findMisclassified(Z1,Z2)

#==========================================================  PLOT  ==================================

# x1,x2 = [],[]

# for i in range(0,30):
# 	x1.append(i)
# 	temp = float(-1*a[1]*i - a[0])
# 	temp = float(temp/a[2])
# 	x2.append(temp)


# plt.scatter(data1X,data1Y,c="red")
# plt.scatter(data2X,data2Y,c="green")
# plt.plot(x1,x2)
# plt.savefig("perceptron12.png",bbox_inches="tight", pad_inches=0.5)	
# plt.show()

#====================================================================================================

