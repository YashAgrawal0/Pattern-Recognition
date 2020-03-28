import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import random
import sys

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
	
	for j in range(3):
		# f = open("2c/group06/Train/feature/"+str(j),"r")
		# f = open(fileName,"r")
		f = open("train_class"+str(j+1)+".txt","r")
		d = 0

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
			sigma[i][i] = 1000000
	# print sigma
	det = find_det(sigma)
	inv = findInverse(sigma)

	N = (1/math.sqrt(2*math.pi*det))
	N *= math.exp((-1.0/2.0)*np.matmul(np.matmul(np.transpose(x-mu),inv),x-mu))

	return N
#====================================================================================================================================

def GMM(mean,covariance,pi,data_points):
	l_theta_old = 1000000000000
	l_theta = 0.0
	itr = 1
	iterationsX,loglikely = [],[]

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

		loglikely.append(l_theta)
		iterationsX.append(itr)
		itr += 1
		print abs(l_theta - l_theta_old)

	# plt.plot(iterationsX,loglikely)
	# plt.xlabel("Iterations -->")
	# plt.ylabel("Log_Likelihood -->")
	# plt.savefig("log_vs_itr/class3/k"+str(k)+"_class3.png",bbox_inches="tight", pad_inches=0.5)
	# plt.show()


#====================================================================================================================================


MEAN,COVARIANCE,PI,data_points = kMeans("train_class3.txt")
# print MEAN
GMM(MEAN,COVARIANCE,PI,data_points)

#----------------------------------------------------------------  Lakshman Rekha --------------------------------------------------

# class_mean = []
# class_covar = []
# class_PI = []

# for i in range(3):
# 	mean,covariance,PI=kMeans("dada")	
# 	class_mean.append(mean)
# 	class_covar.append(covariance)
# 	class_PI.append(PI)