import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from operator import itemgetter
import math
import random
import sys
import os

m = int(sys.argv[1])
n = int(sys.argv[2])

def initializeUtil(observationSeq):
	A = np.zeros([n, n])
	B = np.zeros([n, m])
	PI = np.zeros(n)
	l = len(observationSeq)
	stateLen = np.zeros(n)

	# Initializing stateSequence
	stateSeq = np.zeros(l)
	quotient = int(l)//n
	remainder = int(l)%n
	i = 0
	currState = 0
	while(i < l):
		if(remainder==0):
			stateLen[currState] = quotient
		else: stateLen[currState] = quotient+1

		for k in range(quotient):
			stateSeq[i] = currState
			i += 1
		if(remainder > 0):
			stateSeq[i] = currState
			remainder -= 1
			i += 1
		currState += 1

	#Will always start from first state LEFT-to-RIGHT model
	PI[0] = 1

	# Setting matrix A
	for i in range(n-1):
		A[i][i+1] = 1/stateLen[i]
		A[i][i] = 1 - A[i][i+1]
	A[n-1][n-1] = 1

	# Setting matrix B
	for i in range(l):
		B[int(stateSeq[i])][int(observationSeq[i])] += 1/stateLen[int(stateSeq[i])]

	return A, B, PI

def initialize(observationSeqArr):
	A = np.zeros([n, n])
	B = np.zeros([n, m])
	PI = np.zeros(n)
	l = len(observationSeqArr)

	for i in range(l):
		Atemp, Btemp, PItemp = initializeUtil(observationSeqArr[i])
		A += Atemp
		B += Btemp
		PI += PItemp

	# Take average of all class examples
	A /= l
	B /= l
	PI /= l

	return A, B, PI


def findAlpha(A, B, PI, observationSeq):
	T = len(observationSeq)
	alpha = np.zeros([T, n])

	for j in range(n):
		alpha[0][j] = PI[j]*B[j][int(observationSeq[0])]

	for t in range(T-1):
		for j in range(n):
			for i in range(n):
				alpha[t+1][j] += alpha[t][i]*A[i][j]
			alpha[t+1][j] *= B[j][int(observationSeq[t+1])]

	return alpha

def findBeta(A, B, PI, observationSeq):
	T = len(observationSeq)
	beta = np.zeros([T, n])

	for i in range(n):
		beta[T-1][i] = 1

	t = T-2
	while(t>=0):
		for i in range(n):
			for j in range(n):
				beta[t][i] += A[i][j]*B[j][int(observationSeq[t+1])]*beta[t+1][j]
		t -= 1

	return beta

def findProbability(A, B, PI, observationSeq):
	alpha = findAlpha(A, B, PI, observationSeq)
	# print("findProbability alpha = ")
	# print(alpha)
	probability = 0
	T = len(observationSeq)
	for i in range(n):
		probability += alpha[T-1][i]
	# print("probability = "+str(probability))
	return probability

def HMM(observationSeqArr):
	A, B, PI = initialize(observationSeqArr)

	# print("initialized\nA = ")
	# print(A)
	# print("B = ")
	# print(B)
	# print("PI = ")
	# print(PI)

	L = len(observationSeqArr)
	# Apply EM method here
	probabilitySumOld = 0
	probabilitySumNew = L
	while(abs(probabilitySumOld - probabilitySumNew) > 0.0001):
		zeta = []
		gamma = []
		# print("E-Step")
		# E-Step
		for l in range(L):
			# print("l = "+str(l))
			T = len(observationSeqArr[l])
			alpha = findAlpha(A, B, PI, observationSeqArr[l])
			beta = findBeta(A, B, PI, observationSeqArr[l])

			# print("Alpha = ")
			# print(alpha)
			# print("Beta = ")
			# print(beta)

			tempZeta = np.zeros([T, n, n])
			tempGamma = np.zeros([T, n])
			for t in range(T-1):
				denominator_zeta = 0
				for i in range(n):
					for j in range(n):
						denominator_zeta += alpha[t][i] * A[i][j] * B[j][int(observationSeqArr[l][t+1])] * beta[t+1][j]
				for i in range(n):
					for j in range(n):
						tempZeta[t][i][j] = ( alpha[t][i] * A[i][j] * B[j][int(observationSeqArr[l][t+1])] * beta[t+1][j] ) / denominator_zeta
					
			for t in range(T):
				denominator_gamma = 0
				for i in range(n):
					denominator_gamma += alpha[t][i] * beta[t][i]
				for i in range(n):
					tempGamma[t][i] = ( alpha[t][i] * beta[t][i] ) / denominator_gamma
			zeta.append(tempZeta)
			gamma.append(tempGamma)

		# print("zeta = ")
		# print(zeta)
		# print("gamma = ")
		# print(gamma)

		# print("M-Step")
		# M-Step
		A = np.zeros([n,n])
		B = np.zeros([n,m])
		PI = np.zeros(n)

		for i in range(n):
			for l in range(L):
				PI[i] += gamma[l][0][i]
			PI[i] /= L

		# print("new PI = ")
		# print(PI)

		for i in range(n):
			for j in range(n):
				for l in range(L):
					numerator = 0
					denominator = 0
					T = len(observationSeqArr[l])
					for t in range(T-1):
						numerator += zeta[l][t][i][j]
						denominator += gamma[l][t][i]
					if(denominator < 0.000001):
						denominator = 0.000001
					A[i][j] += numerator/denominator
				A[i][j] /= L

		# print("new A = ")
		# print(A)

		for j in range(n):
			for k in range(m):
				for l in range(L):
					numerator = 0
					denominator = 0
					T = len(observationSeqArr[l])
					for t in range(T):
						if(observationSeqArr[l][t] == k):
							numerator += gamma[l][t][j]
						denominator += gamma[l][t][j]
					if(denominator < 0.000001):
						denominator = 0.000001
					B[j][k] += numerator/denominator
				B[j][k] /= L

		# print("new B = ")
		# print(B)

		# Find new probabilities
		probabilitySumOld = probabilitySumNew
		probabilitySumNew = 0
		# print("L = "+str(L))
		for l in range(L):
			probabilitySumNew += findProbability(A, B, PI, observationSeqArr[l])

		# print("OldP = "+str(probabilitySumOld)+"   NewP = "+str(probabilitySumNew))


	return A, B, PI


def classify(A1, B1, PI1, A2, B2, PI2, A3, B3, PI3, observationSeqArr, ConfusionMatrix, row):
	for l in range(len(observationSeqArr)):
		p1 = findProbability(A1, B1, PI1, observationSeqArr[l])
		p2 = findProbability(A2, B2, PI2, observationSeqArr[l])
		p3 = findProbability(A3, B3, PI3, observationSeqArr[l])
		if(p1 > p2 and p1 > p3):
			ConfusionMatrix[row][0] += 1
		elif(p2 > p3):
			ConfusionMatrix[row][1] += 1
		else:
			ConfusionMatrix[row][2] += 1


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


def main():
	# Take training data from each class
	observationSeqArrTrain1 = []
	observationSeqArrTrain2 = []
	observationSeqArrTrain3 = []

	f = open("Group06/Train/sa_"+str(m)+".txt")
	for i in f.readlines():
		x = [float(k) for k in i.split()]
		observationSeqArrTrain1.append(x)

	f = open("Group06/Train/sai_"+str(m)+".txt")
	for i in f.readlines():
		x = [float(k) for k in i.split()]
		observationSeqArrTrain2.append(x)

	f = open("Group06/Train/sau_"+str(m)+".txt")
	for i in f.readlines():
		x = [float(k) for k in i.split()]
		observationSeqArrTrain3.append(x)


	# Take test data from each class
	observationSeqArrTest1 = []
	observationSeqArrTest2 = []
	observationSeqArrTest3 = []

	f = open("Group06/Test/sa_"+str(m)+".txt")
	for i in f.readlines():
		x = [float(k) for k in i.split()]
		observationSeqArrTest1.append(x)

	f = open("Group06/Test/sai_"+str(m)+".txt")
	for i in f.readlines():
		x = [float(k) for k in i.split()]
		observationSeqArrTest2.append(x)

	f = open("Group06/Test/sau_"+str(m)+".txt")
	for i in f.readlines():
		x = [float(k) for k in i.split()]
		observationSeqArrTest3.append(x)


	# Find HMM for each class
	A1, B1, PI1 = HMM(observationSeqArrTrain1)
	A2, B2, PI2 = HMM(observationSeqArrTrain2)
	A3, B3, PI3 = HMM(observationSeqArrTrain3)
	
	# Classify
	ConfusionMatrix = np.zeros([3,3])
	classify(A1, B1, PI1, A2, B2, PI2, A3, B3, PI3, observationSeqArrTest1, ConfusionMatrix, 0)
	classify(A1, B1, PI1, A2, B2, PI2, A3, B3, PI3, observationSeqArrTest2, ConfusionMatrix, 1)
	classify(A1, B1, PI1, A2, B2, PI2, A3, B3, PI3, observationSeqArrTest3, ConfusionMatrix, 2)
	print_results(ConfusionMatrix)


main()

observationSeqArr = [[1,1,1,1,2], [1,2,2,2,2], [0,1,2,3,3,3,3], [1,1,0,0,0,0]]
# A, B, PI = initialize(observationSeqArr)
# beta = findBeta(A, B, PI, observationSeqArr[1])
# print("A : ")
# print(A)
# print("B : ")
# print(B)
# print("beta : ")
# print(beta)
# prob = findProbability(A, B, PI, observationSeqArr[1])
# print("probability = "+str(prob))

# A, B, PI = HMM(observationSeqArr)