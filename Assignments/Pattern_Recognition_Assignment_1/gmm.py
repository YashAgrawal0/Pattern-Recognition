import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys
from numpy.linalg import inv

k = int(sys.argv[1])

#----------------------------------------------------------------------------------------------------------

def getMean(x1):
	sum1=0
	for i in range(0,len(x1)):
		sum1 += float(x1[i])						# Class is taken as input

	sum1/=float(len(x1))							# Calculated mean value is returned
	return sum1

#----------------------------------------------------------------------------------------------------------

def getCovariance(x,y):
	meanx=getMean(x)
	meany=getMean(y)

	covar=np.zeros((2,2))
	length=float(len(x))

	for i in range(0,len(x)):
		covar[0][0]+=(float(x[i])-meanx)*(float(x[i])-meanx)
		covar[0][1]+=0
		covar[1][0]+=0
		covar[1][1]+=(float(y[i])-meany)*(float(y[i])-meany)

	covar[0][0]/=length 										# Covariance matrix is calculated
	covar[0][1]/=length 										# for the input two classes
	covar[1][0]/=length
	covar[1][1]/=length

	return covar

#----------------------------------------------------------------------------------------------------------

def find_distribution(pi,covar,meanX,meanY,pointX,pointY):
	print len(pointX)
	print len(pointY)
	# mean,point = [],[]
	# point.append([pointX,pointY])
	# mean.append([meanX,meanY])
	# mean = np.transpose(mean)
	# point = np.transpose(point)
	# cov_sqrt = np.sqrt(covar)
	# cov_sqrt_inv = inv(cov_sqrt)
	# tran = np.transpose(np.subtract(point,mean))
	# tran = np.dot(tran,cov_sqrt_inv)
	# tran = np.dot(tran,np.subtract(point,mean))
	# print tran

def find_euc_distance(x1,y1,x2,y2):
	val = (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)
	val = math.sqrt(val)
	return val

def take_mean(x):
	val = 0
	for i in x:
		val = val + float(i)	
	return val/len(x)

Colour = []
for i in range(k):
	x = []
	x.append(random.uniform(0,1))
	x.append(random.uniform(0,1))
	x.append(random.uniform(0,1))
	Colour.append(x)
# Colour=["blue",
# 		"green",
# 		"yellow",
# 		"red",
# 		"darkcyan"
# 		]

x_val = []
y_val = []

f1 = open("NLS_group6/class1.txt","r")
# for j in range(0,60):
# 	f = open("feature/"+str(j),"r")

# 	for i in f.readlines():
# 		x,y = i.split()
# 		x_val.append(x)
# 		y_val.append(y)

for i in f1.readlines():
	x,y = i.split()
	x_val.append(x)
	y_val.append(y)

muX,muY,X,Y=[],[],[],[]

for i in range(0,k):
	x = random.randint(0,len(x_val))
	muX.append(float(x_val[x]))
	muY.append(float(y_val[x]))
	X.append(0)
	Y.append(0)

flag = True

#=========================================================+++++ K Means Loop ++++++======================================

# plt.ion()
while flag:
	flag = False
	points_X,points_Y = [],[]
	for i in range(0,k):
		points_X.append([])
		points_Y.append([])
	for i in range(k):
		X[i] = muX[i]
		Y[i] = muY[i]
		plt.plot(muX[i],muY[i],'bo',markersize=10,marker='v',color='black')

	min_val=0

	for i in range(0,len(x_val)):
		index=0
		min_val = find_euc_distance(float(muX[0]),float(muY[0]),float(x_val[i]),float(y_val[i]))
		for j in range(1,k):
			distance = find_euc_distance(float(muX[j]),float(muY[j]),float(x_val[i]),float(y_val[i]))
			if min_val > distance:
				min_val = distance
				index = j
		points_X[index].append(x_val[i])
		points_Y[index].append(y_val[i])

	plt.title("K-Means Clustering")
	for i in range(k):
		plt.scatter(points_X[i],points_Y[i],s=50,color = (Colour[i][0],Colour[i][1],Colour[i][2]))
		muX[i] = take_mean(points_X[i])
		muY[i] = take_mean(points_Y[i])

	for i in range(k):
		if muX[i] - X[i] !=0 or muY[i]-Y[i]!=0:
			flag = True

	# plt.draw()
	# plt.pause(0.8)
	# plt.clf()
	
#-------------------------------------------------+++ GMM  +++++------------------------------------------------------------

covar = []

for i in range(k):
	covar.append(getCovariance(points_X[i],points_Y[i]))
	
pi_k = []

for i in range(k):
	# print len(points_X[i]),len(x_val)
	pi_k.append(float(len(points_X[i])/float(len(x_val))))

	
find_distribution(pi_k[0],covar[0],muX[0],muY[0],points_X[0],points_Y[0])
old_l_theta = 0
l_theta = 0


