import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from operator import itemgetter
import math
import random
import sys
import os

k = int(sys.argv[1])
dim = 39
data_train = []

def load_all_files_in_folder(folder):
	files = []
	for filename in os.listdir(folder):
		files.append(filename)
	return files

def input_from_folder(folder):
	training_data = []
	files = load_all_files_in_folder(folder)
	for i in range(len(files)):
		data_point = []
		name_file = folder +"/" + str(files[i])
		f=open(name_file,"r")

		for j in f.readlines():
			x = [float(k) for k in j.split()]
			data_point.append(x)
		training_data.append(data_point)
	training_data_np = np.array(training_data)
	return training_data_np

def calculate_dtw(train_ex,test_ex):
	train_ex=np.array(train_ex).astype(float)
	test_ex=np.array(test_ex).astype(float)
	m=len(train_ex)
	n=len(test_ex)
	dp=np.zeros((m+1,n+1))

	for i in range(m+1):
		dp[i][0]=float("inf")
	for j in range(n+1):
		dp[0][j]=float("inf")
	dp[0][0]=0.0
	for i in range(1,m+1):
		for j in range(1,n+1):
			cost=np.linalg.norm(train_ex[i-1]-test_ex[j-1])
			dp[i][j]=cost+min(dp[i-1][j-1],min(dp[i-1][j],dp[i][j-1]))
	return dp[m][n]

def KNN_on_each_test(test_ex):
	dtw_dist = []
	predicted_class1 , predicted_class2, predicted_class3, label = 0,0,0,0
	test_ex=np.array(test_ex).astype(float)

	for i in range(len(data_train)):
		for j in range(len(data_train[i])):
			train_ex=data_train[i][j]
			x=[]
			x.append(calculate_dtw(train_ex,test_ex))
			x.append(i+1)
			dtw_dist.append(x)
	# dtw_dist=np.array(dtw_dist)
	dtw_dist.sort(key=itemgetter(0))

	for i in range(0,k):
		if dtw_dist[i][1]==1:
			predicted_class1+=1
		elif dtw_dist[i][1]==2:
			predicted_class2+=1
		elif dtw_dist[i][1]==3:
			predicted_class3+=1

	if predicted_class1>=predicted_class2 and predicted_class1>=predicted_class3:
		label=1
	elif predicted_class2>=predicted_class1 and predicted_class2>=predicted_class3:
		label=2
	elif predicted_class3>=predicted_class1 and predicted_class3>=predicted_class2:
		label=3

	return label


def bayes_classifier():
	confusionMat=np.zeros((3,3))
	len_train_c1,len_train_c2,len_train_c3=0,0,0
	for i in range(3):
		if i==0:
			data_test=input_from_folder("Group06/Test/sa")
			len_train_c1=len(data_test)
		elif i==1:
			data_test=input_from_folder("Group06/Test/sai")
			len_train_c2=len(data_test)
		elif i==2:
			data_test=input_from_folder("Group06/Test/sau")
			len_train_c3=len(data_test)

		for j in range(len(data_test)):
			label=KNN_on_each_test(data_test[j])

			print "Original Class : ",i+1," File : ",j+1," Predicted Class : ",label
			confusionMat[i][label-1]+=1

	print "Confusion Matrix :\n",confusionMat, "\n"

	accuracy = float((confusionMat[0][0]+confusionMat[1][1]+confusionMat[2][2])/(len_train_c1+len_train_c2+len_train_c3))
	print "Accuracy : ", accuracy*100, "%\n"

	recallC1 = float(confusionMat[0][0]/len_train_c1)
	recallC2 = float(confusionMat[1][1]/len_train_c2)
	recallC3 = float(confusionMat[2][2]/len_train_c3)

	print "Recall :"
	print "Class 1 :   ",recallC1												# Output recall of
	print "Class 2 :   ",recallC2												# the classes
	print "Class 3 :   ",recallC3
	print "Mean Recall : ", (recallC1+recallC2+recallC3)/3
	print "\n"

#------------------------------------------------------------------------------------------------------------

	precisionC1=float((confusionMat[0][0])/(confusionMat[0][0]+confusionMat[1][0]+confusionMat[2][0]))
	precisionC2=float((confusionMat[1][1])/(confusionMat[0][1]+confusionMat[1][1]+confusionMat[2][1]))
	precisionC3=float((confusionMat[2][2])/(confusionMat[0][2]+confusionMat[1][2]+confusionMat[2][2]))

	print "Precision :"
	print "Class 1 :   ",precisionC1
	print "Class 2 :   ",precisionC2											# output Precision
	print "Class 3 :   ",precisionC3											# of the classes
	print "Mean Precision : ", (precisionC1+precisionC2+precisionC3)/3
	print "\n"

#------------------------------------------------------------------------------------------------------------

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
	print "Class 2 :   ",fmeasureC2												# output F-Measure
	print "Class 3 :   ",fmeasureC3												# of the classes
	print "Mean F-Measure : ", (fmeasureC1+fmeasureC2+fmeasureC3)/3
	print "\n"


data_train.append(input_from_folder("Group06/Train/sa"))
data_train.append(input_from_folder("Group06/Train/sai"))
data_train.append(input_from_folder("Group06/Train/sau"))
data_train=np.array(data_train)

# print calculate_dtw(data_train[0][0],data_train[1][0])
# KNN_on_each_test(data_train[2][0])
bayes_classifier()