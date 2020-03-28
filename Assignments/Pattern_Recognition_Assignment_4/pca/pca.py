import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from operator import itemgetter
import math
import random
import sys
import os

l = int(sys.argv[1])

#====================================================================================================================================

def input_all_data(folder):
	data_points = []

	f = open(folder+"/train_class1.txt","r")
	for i in f.readlines():
		x = [float(j) for j in i.split()]
		data_points.append(x)

	f = open(folder+"/train_class2.txt","r")
	for i in f.readlines():
		x = [float(j) for j in i.split()]
		data_points.append(x)

	f = open(folder+"/train_class3.txt","r")
	for i in f.readlines():
		x = [float(j) for j in i.split()]
		data_points.append(x)

	f = open(folder+"/test_class1.txt","r")
	for i in f.readlines():
		x = [float(j) for j in i.split()]
		data_points.append(x)

	f = open(folder+"/test_class2.txt","r")
	for i in f.readlines():
		x = [float(j) for j in i.split()]
		data_points.append(x)

	f = open(folder+"/test_class3.txt","r")
	for i in f.readlines():
		x = [float(j) for j in i.split()]
		data_points.append(x)

	return np.array(data_points)


#====================================================================================================================================

def pca(folder):
	data_points = input_all_data(folder)
	# data = np.zeros([2,2])
	# data = [[1,2],[3,4]]
	# print data
	covar = np.cov(np.transpose(data_points))
	# print covar
	# print covar.shape
	eigen_values, eigen_vectors = np.linalg.eig(covar)
	# print "shape ", eigen_vectors.shape
	# print eigen_values
	# print eigen_vectors
	eigen = []
	for i in range(len(eigen_values)):
		x=[]
		x.append(eigen_values[i])
		x.append(eigen_vectors[i])
		eigen.append(x)

	eigen.sort(key=itemgetter(0),reverse = True)
	x_axis = []
	lambda_i = []
	for i in range(len(eigen)):
		x_axis.append(i+1)
		lambda_i.append(eigen[i][0])

	print lambda_i

	plt.plot(x_axis,lambda_i)
	plt.xlabel("value of l")
	plt.ylabel("lth eigen value")
	plt.show()


	return eigen


def reduce_dimension(eigen,filename):
	data_points=[]
	f = open("old_data/"+filename,"r")
	f1 = open("reduced_dataA/"+str(l)+"/"+filename,"w")
	for i in f.readlines():
		x = [float(j) for j in i.split()]
		data_points.append(x)
	mu = np.mean(data_points, axis=0)
	# print mu
	
	for i in range(len(data_points)):
		yn = data_points[i]-mu
		for j in range(l):
			# print np.array(eigen[j][1])
			# print np.transpose(yn)
			an_i = np.matmul(np.array(eigen[j][1]),np.transpose(yn))
			f1.write(str(an_i)+" ")
		f1.write("\n")
	f1.close()



sorted_eigen = pca("old_data")
reduce_dimension(sorted_eigen,"train_class1.txt")
reduce_dimension(sorted_eigen,"train_class2.txt")
reduce_dimension(sorted_eigen,"train_class3.txt")
reduce_dimension(sorted_eigen,"test_class1.txt")
reduce_dimension(sorted_eigen,"test_class2.txt")
reduce_dimension(sorted_eigen,"test_class3.txt")
