import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import random
import sys
from pylab import figure, show, rand
from matplotlib.patches import Ellipse
from math import pi, cos, sin

k = int(sys.argv[1])
dim = 0

Colour = []
for i in range(k):
	x = []
	x.append(random.uniform(0,1))
	x.append(random.uniform(0,1))
	x.append(random.uniform(0,1))
	Colour.append(x)

#====================================================================================================================================

def input(fileName):
	data_points = []
	
	# for j in range(10):
		# f = open("train/aqueduct/new_feature/"+str(j),"r")
	f = open(fileName,"r")
	# d = 0

	for i in f.readlines():
		x = [float(j) for j in i.split()]
		data_points.append(x)

	global dim
	dim = len(data_points[0])
	
	return np.array(data_points),dim


#====================================================================================================================================
def kMeans(fileName):
	data_points,dim = input(fileName)
	mu = np.zeros((k,dim))

	for i in range(k):
		ind = random.randint(0,len(data_points)-1)
		mu[i] = data_points[ind]

	flag = True
	# plt.ion()
	while(flag):
		flag = False
		cluster = []
		old_mu = np.zeros((k,dim))
		old_mu = np.copy(mu)

		for i in range(k):
			cluster.append([])
			# plt.plot(mu[i][0],mu[i][1],'bo',markersize=10,marker='v',color='black')
	
		for i in range(len(data_points)):
			minimum = np.linalg.norm(data_points[i]-mu[0])
			index = 0
			for j in range(k):
				dist = np.linalg.norm(data_points[i]-mu[j])
				if minimum > dist:
					minimum = dist
					index = j
			cluster[index].append(data_points[i])

		for i in range(k):
			# x,y = zip(*cluster[i])
			# plt.scatter(x,y,s=2,color = (Colour[i][0],Colour[i][1],Colour[i][2]))
			mu[i] = np.mean(cluster[i],axis=0)
		
		if np.linalg.norm(old_mu-mu)!=0:
			flag =True
		# plt.draw()
		# plt.pause(0.8)
		# plt.clf()	

	covar = np.zeros((k,dim,dim))

	for i in range(k):
		covar[i] = np.cov(np.transpose(cluster[i]))
		for j in range(len(covar[i])):
			for x in range(len(covar[i])):
				if j!=x:
					covar[i][j][x] = 0.0


	pi = np.zeros(k)
	for i in range(k):
		pi[i] = float(len(cluster[i]))/float(len(data_points))

	return mu,covar,pi,data_points
#====================================================================================================================================

def find_det(matrix):
	val = 1.0
	for i in range(len(matrix)):
		val *= matrix[i][i]
	return val	

#====================================================================================================================================

def findInverse(matrix):
	x = np.zeros((len(matrix),len(matrix)))

	for i in range(len(matrix)):
		x[i][i] = (1/matrix[i][i])
	return x	

#====================================================================================================================================

def findN(x,mu,sigma):
	for i in range(len(sigma)):
		if sigma[i][i]==0:
			return random.random()
	# print sigma
	det = find_det(sigma)
	inv = findInverse(sigma)

	N = (1.0/math.sqrt(2*math.pi*det))
	N *= math.exp((-1.0/2.0)*np.matmul(np.matmul(np.transpose(x-mu),inv),x-mu))

	return N
#====================================================================================================================================

def GMM(mean,covariance,pi,data_points):
	l_theta_old = 1000000000000
	l_theta = 0.0

	while abs(l_theta - l_theta_old) > 0.01:
		l_theta_old = l_theta
		l_theta = 0.0
		cluster = []

		for i in range(len(data_points)):
			sumK = 0.0
			for j in range(k):
				sumK += pi[j]*findN(data_points[i],mean[j],covariance[j]) 
			l_theta += math.log(sumK)	

		gammaZNK = np.zeros((len(data_points),k))
		
		for i in range(len(data_points)):
			prob=0.0
			for j in range(k):
				gammaZNK[i][j] = (pi[j]*findN(data_points[i],mean[j],covariance[j]))
				prob+=gammaZNK[i][j]
			l_theta += np.log(prob)
			gammaZNK[i]/=prob

		NK = np.zeros(k)

		for j in range(k):
			sumN = 0.0
			for i in range(len(data_points)):
				sumN += gammaZNK[i][j]
			NK[j] = sumN
			
		new_mean = np.zeros((k,dim))

		for j in range(k):
			sumN = np.zeros(dim)
			for i in range(len(data_points)):
				sumN += (gammaZNK[i][j]*data_points[i])
			sumN = sumN/NK[j]
			new_mean[j] = np.copy(sumN)

		new_covariance = np.zeros((k,dim,dim))

		for j in range(k):
			sumN = np.zeros((dim,dim))
			for i in range(len(data_points)):
				mat = np.matmul(np.array(data_points[i]-mean[j]).reshape((dim,1)),np.array(data_points[i]-mean[j]).reshape((1,dim)))
				sumN += (gammaZNK[i][j]*mat)
			sumN = sumN/NK[j]
			for x in range(dim):
				for y in range(dim):
					if x!=y :
						sumN[x][y] = 0.0
			new_covariance[j] = np.copy(sumN)	
		
		new_pi = np.zeros(k)

		for i in range(k):
			new_pi[i] = float(NK[i]/len(data_points))

		covariance = np.copy(new_covariance)
		mean = np.copy(new_mean)
		pi = np.copy(new_pi)

		return mean,covariance,pi,data_points


#====================================================================================================================================


# MEAN,COVARIANCE,PI,data_points = kMeans("Class1")

# GMM(MEAN,COVARIANCE,PI,data_points)

#----------------------------------------------------------------  Lakshman Rekha --------------------------------------------------

# class_mean = []
# class_covar = []
# class_PI = []

# for i in range(3):
# 	mean,covariance,PI=kMeans("dada")	
# 	class_mean.append(mean)
# 	class_covar.append(covariance)
# 	class_PI.append(PI)


def findClass(x, y, MEAN, COVARIANCE, PI):
	l = np.zeros(3)
	data = [x, y]
	# print(data)
	for c in range(3):
		sumOverAllClusters = 0.0
		for clusterNum in range(k):
			sumOverAllClusters += (PI[c][clusterNum])*findN(data, MEAN[c][clusterNum], COVARIANCE[c][clusterNum])
		if(sumOverAllClusters < 0.00000000001):
			sumOverAllClusters = 0.00000000001
		l[c] = math.log(sumOverAllClusters)

	# Belongs to class 1
	if(l[0] > l[1] and l[0] > l[2]):
		return 0
	elif(l[1] > l[2]):
		return 1
	else:
		return 2




MEAN = []
COVARIANCE = []
PI = []
DATA_POINTS = []
for i in range(2):
	mean, cov, pi, data_points = kMeans("LS/train/c"+str(i+1)+"_FDA12.txt")
	# mean_gmm, cov_gmm, pi_gmm, data_points_gmm = kMeans("train_class"+str(i+1)+".txt")
	mean_gmm, cov_gmm, pi_gmm, data_points_gmm = GMM(mean, cov, pi, data_points)
	MEAN.append(mean_gmm)
	COVARIANCE.append(cov_gmm)
	PI.append(pi_gmm)

MEAN = np.array(MEAN)
COVARIANCE = np.array(COVARIANCE)
PI = np.array(PI)


# Plot contours for each cluster
# print(MEAN.shape)
# print(COVARIANCE.shape)
# print(PI.shape)
# print(MEAN)
# print(COVARIANCE)
# print(PI)

# Confusion matrix
ConfusionMatrix = np.zeros([3,3])

l = [0, 0 , 0]

for i in range(3):
	data_points, dim = input("LS/test/c"+str(i+1)+"_FDA12.txt")
	l[i] = len(data_points)
	for j in range(len(data_points)):
		c = findClass(data_points[j][0], data_points[j][1], MEAN, COVARIANCE, PI)
		ConfusionMatrix[i][c] += 1;

print "Confusion Matrix :\n",ConfusionMatrix, "\n"

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



# For training data points
class1X = []
class2X = []
class3X = []
class1Y = []
class2Y = []
class3Y = []

# For general background points
back1X = []
back1Y = []
back2X = []
back2Y = []
back3X = []
back3Y = []


# Finding colors for background points
# Change ranges in for loop for other datasets
# i = -200
# while(i<=1600):
# 	j = 200
# 	while(j<=1600):
# 		c = findClass(i, j, MEAN, COVARIANCE, PI)
# 		if(c == 0):
# 			back1X.append(i)
# 			back1Y.append(j)
# 		elif(c==1):
# 			back2X.append(i)
# 			back2Y.append(j)
# 		else:
# 			back3X.append(i)
# 			back3Y.append(j)
# 		j += 5
# 	i += 5

# # Finding training data points
# train, dim = input("test_class1.txt")
# for i in range(len(train)):
# 	class1X.append(train[i][0])
# 	class1Y.append(train[i][1])

# train, dim = input("test_class2.txt")
# for i in range(len(train)):
# 	class2X.append(train[i][0])
# 	class2Y.append(train[i][1])

# train, dim = input("test_class3.txt")
# for i in range(len(train)):
# 	class3X.append(train[i][0])
# 	class3Y.append(train[i][1])


# plt.xlim([-200,1600])
# plt.ylim([200,1600])
# # Plotting test data points
# plt.scatter(class1X, class1Y, c="red", edgecolor="black", alpha=1)
# plt.scatter(class2X, class2Y, c="green", edgecolor="black", alpha=1)
# plt.scatter(class3X, class3Y, c="blue", edgecolor="black", alpha=1)
# plt.scatter(back1X, back1Y, s=10, c="pink", edgecolor="", alpha=0.3)
# plt.scatter(back2X, back2Y, s=10, c="lightgreen", edgecolor="", alpha=0.3)
# plt.scatter(back3X, back3Y, s=10, c="lightblue", edgecolor="", alpha=0.3)
# plt.show()