import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import random
import sys

a1 = np.zeros(3)
a1[0],a1[1],a1[2] = 1,1,1
a2 = np.zeros(3)
a2[0],a2[1],a2[2] = 1,1,1
a3 = np.zeros(3)
a3[0],a3[1],a3[2] = 1,1,1

def input(fileName):
	length = 0
	temp = open(fileName,"r")
	for i in temp.readlines():
		length += 1
	
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
		cnt = cnt + 1
	return np.array(Z)

def inputTestData(fileName):
	length = 0
	temp = open(fileName,"r")
	for i in temp.readlines():
		length += 1
	
	f = open(fileName,"r")
	cnt = 0
	Z = []
	for i in f.readlines():
		if cnt >= 0.75*length:
			x = [float(j) for j in i.split()]
			y = []
			y.append(1)
			for j in x:
				y.append(j)
			Z.append(y)
		cnt = cnt + 1
	return np.array(Z)


def findMisclassified(x,y,a):
	Z1 = x.astype(float)
	Z2 = y.astype(float)
	misclassified = []
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

def finda(z,Z,a):
	misclassified = findMisclassified(z,Z,a)
	ETA = 1
	length = 0

	while(len(misclassified)!=0):
		mat = np.zeros(len(misclassified[0]))
		for i in misclassified:
			mat = mat + i

		# mat = ETA*mat
		# ETA = ETA - 0.3*ETA

		a = a + mat
		misclassified = findMisclassified(z,Z,a)

	return a	

def findClass(z):
	global a1,a2,a3
	clas = np.zeros(3)
	if np.matmul(a1,z) > 0:
		clas[0] += 1
	else:
		clas[1] += 1

	if np.matmul(a2,z) > 0:
		clas[1] += 1
	else:
		clas[2] += 1
	
	if np.matmul(a3,z) > 0:
		clas[0] += 1
	else:
		clas[2] += 1		
	
	mx = -1
	ind = 0

	for i in range(3):
		if clas[i] > mx:
			ind = i
			mx = clas[ind]
			
	return ind


Z1 = input("DATA/LS/class1.txt")
Z2 = input("DATA/LS/class2.txt")
Z3 = input("DATA/LS/class3.txt")
z1 = inputTestData("DATA/LS/class1.txt")
z2 = inputTestData("DATA/LS/class2.txt")
z3 = inputTestData("DATA/LS/class3.txt")

a1 = finda(Z1,Z2,a1)
a2 = finda(Z2,Z3,a2)
a3 = finda(Z1,Z3,a3)

Confusion_Matrix = np.zeros((3,3))

for i in z1:
	c = findClass(i)
	Confusion_Matrix[0][c] += 1
for i in z2:
	c = findClass(i)
	Confusion_Matrix[1][c] += 1
for i in z3:
	c = findClass(i)
	Confusion_Matrix[2][c] += 1


def print_results(ConfusionMatrix):
	print "Confusion Matrix :\n",ConfusionMatrix, "\n"

	l = np.zeros([3])
	for i in range(3):
		for j in range(3):
			l[i] += ConfusionMatrix[i][j]

	accuracy = float((ConfusionMatrix[0][0]+ConfusionMatrix[1][1]+ConfusionMatrix[2][2])/(l[0]+l[1]+l[2]))
	print "Accuracy : ", accuracy*100, "%\n"

	recallC1 = float(ConfusionMatrix[0][0]/l[0])
	recallC2 = float(ConfusionMatrix[1][1]/l[1])
	recallC3 = float(ConfusionMatrix[2][2]/l[2])

	print "Recall :"
	print "Class 1 :   ",recallC1
	print "Class 2 :   ",recallC2
	print "Class 3 :   ",recallC3
	print "Mean Recall : ", (recallC1+recallC2+recallC3)/3
	print "\n"


	precisionC1=float((ConfusionMatrix[0][0])/(ConfusionMatrix[0][0]+ConfusionMatrix[1][0]+ConfusionMatrix[2][0]))
	precisionC2=float((ConfusionMatrix[1][1])/(ConfusionMatrix[0][1]+ConfusionMatrix[1][1]+ConfusionMatrix[2][1]))
	precisionC3=float((ConfusionMatrix[2][2])/(ConfusionMatrix[0][2]+ConfusionMatrix[1][2]+ConfusionMatrix[2][2]))

	print "Precision :"
	print "Class 1 :   ",precisionC1
	print "Class 2 :   ",precisionC2
	print "Class 3 :   ",precisionC3
	print "Mean Precision : ", (precisionC1+precisionC2+precisionC3)/3
	print "\n"

	if (precisionC1+recallC1)==0:
		fmeasureC1=0
	else: fmeasureC1 = float((2*(precisionC1*recallC1))/(precisionC1+recallC1))
	if (precisionC2+recallC2)==0:
		fmeasureC2=0
	else: fmeasureC2 = float((2*(precisionC2*recallC2))/(precisionC2+recallC2))
	if (precisionC3+recallC3)==0:
		fmeasureC3=0
	else: fmeasureC3 = float((2*(precisionC3*recallC3))/(precisionC3+recallC3))

	print "F-Measure :"
	print "Class 1 :   ",fmeasureC1
	print "Class 2 :   ",fmeasureC2
	print "Class 3 :   ",fmeasureC3
	print "Mean F-Measure : ", (fmeasureC1+fmeasureC2+fmeasureC3)/3
	print "\n"

print_results(Confusion_Matrix)

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

