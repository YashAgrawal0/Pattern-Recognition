import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import random
import sys

#============================================  Declaration 	=================================

data_points = []
dim = 0
k = int(sys.argv[1]) 		# number of cluster
flag = True
mu = []						# mean array
cluster = []				# cluster array
#==================================================================================================
												# Functions
def find_euc_distance(mean,points):
	val = 0
	for i in range(len(mean)):
		val += (float(mean[i])-float(points[i]))*(float(mean[i])-float(points[i]))
	val = math.sqrt(val)	
	return val

#------------------------------------------------------------------------------------------------------------------------


def find_new_mean(cluster):
	mu = []
	for j in range(len(cluster[0])):
		val = 0
		for i in range(len(cluster)):
			val += float(cluster[i][j])
		val	/= len(cluster)
		mu.append(val)
	return mu

#------------------------------------------------------------------------------------------------------------------------


def check_Convergence(oldMu,mu):
	for i in range(k):
		for j in range(len(mu[i])):
			if mu[i][j]!=oldMu[i][j]:
				return True
	return False

#------------------------------------------------------------------------------------------------------------------------


def find_covar(icluster,imean,covar,ind):
	val = 0
	for j in range(len(icluster[0])):
		val = 0
		for i in range(len(icluster)):
			val += (float(imean[j]) - float(icluster[i][j]))*(float(imean[j]) - float(icluster[i][j]))
		covar[ind][j][j] = val
	return covar	

#------------------------------------------------------------------------------------------------------------------------


def find_Covariance_Matrix(cluster,mean):
	covar = np.zeros((k,dim,dim))
	# for i in range(0,k):
	# 	var = np.var([map(float,j) for j in cluster[i]],axis=0)
	# 	for j in range(0,len(var)):
	# 		covariance[i][j][j]=var[j]
	# return covariance
	for i in range(k):
		covar = find_covar(cluster[i],mean[i],covar,i)
	return covar

def calc_pi():
	PI = []
	length = len(data_points)
	for i in range(k):
		PI.append(len(cluster[i])/float(length))
	return PI	

#------------------------------------------------------------------------------------------------------------------------

def find_Normal(x,mu,covar):
	x = np.array(x).astype(float)
	mu = np.array(mu).astype(float)
	covar = np.array(covar).astype(float)
	# print covar
	# print np.linalg.det(covar)
	# print "\n\n\n\n"

	p = math.exp((-1/2.0)*np.dot((np.transpose(x-mu)),np.dot(inv(covar),x-mu)))
	p/=math.sqrt(2*math.pi)
	p/=(math.sqrt(np.linalg.det(covar)))
	return p

#------------------------------------------------------------------------------------------------------------------------

# def gaussian_mixture_model(mean,covariance,pi):

# 	#Calculate Initial Likelihood
# 	l_theta = 0.0
# 	l_theta_Old = 10000000000000
# 	flag = True
# 	# print covariance
# 	while flag:
# 		gammaZnk=np.zeros((len(data_points),k))
# 		Nk_cap = np.zeros(k)

# 		for i in range(0,len(data_points)):
# 			prob=0.0
# 			for j in range(k):
# 				gammaZnk[i][j] = (pi[j]*find_Normal(data_points[i],mean[j],covariance[j]))
# 				prob+=gammaZnk[i][j]
# 			l_theta += np.log(prob)
# 			gammaZnk[i]/=prob

# 		for i in range(k):
# 			for j in range(len(data_points)):
# 				Nk_cap[i]+=(gammaZnk[j][i])
# 		print Nk_cap
# 		#Maximization Step
# 		for i in range(k):
# 			s=np.zeros((dim,dim))
# 			for j in range(len(data_points)):
# 				arr = np.dot(data_points[j]-mean[i],np.transpose(data_points[j]-mean[i]))
# 				# print arr
# 				s+=(gammaZnk[j][i]* arr)
# 			s/=Nk_cap[i]
# 			for j in range(dim):
# 				for h in range(dim):
# 					if j!=h:
# 						s[j][h]=0.0
# 			covariance[i]=s
		
# 		for i in range(k):
# 			s=np.zeros(dim)
# 			for j in range(len(data_points)):
# 				s+=(gammaZnk[j][i]*data_points[i])
# 			s/=Nk_cap[i]
# 			mean[i]=s

# 		pi = Nk_cap
# 		pi/= len(data_points)

# 		if abs(l_theta - l_theta_Old)<0.0001:
# 			flag = False
# 		print abs(l_theta - l_theta_Old)
# 		l_theta_Old = l_theta

#------------------------------------------------------------------------------------------------------------------------

def find_det(matrix):
	val = 1
	for i in range(len(matrix)):
		val *= matrix[i][i]
	return val	

def find_inv(matrix):
	x = np.zeros((len(matrix),len(matrix)))
	matrix = np.array(matrix).astype(float)
	for i in range(len((matrix))):
		x[i][i] = (1.0/matrix[i][i])
	return x

def calc_theta(mean,covar,pi):
	sumN = 0.0

	for i in range(len(data_points)):
		sumK = 0.0
		for j in range(k):
			sub = np.subtract(np.array(data_points[i]).astype(float),np.array(mean[j]).astype(float))
			value = np.matmul(sub,np.array(find_inv(covar[j])).astype(float))
			val = np.matmul(value,np.array(sub)).astype(float)
			val = pi[j]*math.exp((-1.0/2.0)*val)
			temp = math.sqrt(find_det(covar[j]))
			temp = temp*math.sqrt(2*math.pi)
			val = val/temp
			sumK += val
		sumN += math.log(sumK)
	return sumN

def calc_gamma(mean,covar,pi):
	gamma = np.zeros((len(data_points),k))
	summ = []
		
	for i in range(len(data_points)):
		sumK = 0.0
		for j in range(k):
			sub = np.subtract(np.array(data_points[i]).astype(float),np.array(mean[j]).astype(float))
			value = np.dot(sub,np.array(find_inv(covar[j])).astype(float))
			val = np.dot(value,np.array(np.transpose(sub)))
			val = pi[j]*math.exp((-1.0/2)*val)
			temp = math.sqrt(find_det(covar[j]))
			temp = temp*math.sqrt(2*math.pi)
			val = val/temp
			sumK += val
		summ.append(float(sumK))
	
	for i in range(len(data_points)):
		for j in range(k):
			sub = np.subtract(np.array(data_points[i]).astype(float),np.array(mean[j]).astype(float))
			value = np.dot(sub,np.array(find_inv(covar[j])).astype(float))
			val = np.dot(value,np.array(np.transpose(sub)))
			val = pi[j]*math.exp((-1.0/2)*val)
			temp = math.sqrt(find_det(covar[j]))
			temp = temp*math.sqrt(2*math.pi)
			val = val/temp
			gamma[i][j] = val/summ[i]		
	return gamma	

def cal_new_mean(gamma):
	mu = []

	for j in range(k):
		x = np.zeros((1,dim))
		for i in range(len(data_points)):
			x += float(gamma[i][j])*np.array(data_points[i]).astype(float)
		for i in range(len(x)):
			x[i] = x[i]/len(cluster[j])
		mu.append(x)
	return mu	

def return_2d(matrix):
	matrix = np.array(matrix).astype(float)
	mat = matrix[0]
	x = np.zeros((len(mat),len(mat)))

	for i in range(len(mat)):
		for j in range(len(mat)):
			x[i][j] = float(mat[i])*float(mat[j])
	return x

def cal_new_covar(mean,gamma):
	covar = []

	for j in range(k):
		x = np.zeros((dim,dim))
		for i in range(len(data_points)):
			twoD = return_2d(np.subtract(np.array(data_points[i]).astype(float),np.array(mean[j]).astype(float)))
			x += float(gamma[i][j])*np.array(twoD).astype(float)
		for i in range(len(x)):
			for j in range(len(x)):
				if i!=j:
					x[i][j] = 0.0
				else:
					x[i][j] = x[i][j]/len(cluster[j])

		covar.append(x)

	return covar

def calc_new_pi(cluster):
	x = []
	for i in range(k):
		x.append(float(len(cluster[i]))/float(len(data_points)))
	return x	

def gaussian_mixture(mean,covar,pi):
	flag = True
	l_theta = 100000000000

	x,y = [],[]
	itr =1

	while(flag):
		cluster = []
		l_theta_old = l_theta
		
		for i in range(k):
			cluster.append([])

		# print covar	
		l_theta = calc_theta(mean,covar,pi)
		gammaZNK = calc_gamma(mean,covar,pi)
		# print "\n"
		# print covar
		print gammaZNK
		mean = cal_new_mean(gammaZNK)
		print "\n"
		# print covar
		covar = cal_new_covar(mean,gammaZNK)
		print "\n"
		# print covar
		
		covar = np.array(covar).tolist()
		mean = np.array(mean).tolist()
		
		mu = []
		for i in range(k):
			mu.append(mean[i][0])
		mean = mu
		# print covar
		for i in range(len(data_points)):
			minimum = find_euc_distance(mu[0],data_points[i])
			index = 0
			for j in range(k):
				dist = find_euc_distance(mu[j],data_points[i])
				if minimum > dist:
					minimum = dist
					index = j
			cluster[index].append(data_points[i])

		pi = calc_new_pi(cluster)	
		
		if itr!=1 :
			y.append(l_theta)
			x.append(itr)
		itr+=1
		print l_theta
		if abs(l_theta - l_theta_old) < 0.01 :
			flag = False
	# plt.plot(x,y)
	# plt.show()		
	# print x,y

#------------------------------------------------------------------------------------------------------------------------
		

#===================================================================================================================
#													Coloring the Graph

Colour = []
for i in range(k):
	x = []
	x.append(random.uniform(0,1))
	x.append(random.uniform(0,1))
	x.append(random.uniform(0,1))
	Colour.append(x)

#===================================================================================================================
#														Input File

for j in range(1):
	# f = open("2c/group06/Train/feature/"+str(j),"r")
	f = open("test.txt","r")
	# f = open("1/group06.txt","r")
	for i in f.readlines():
		x = i.split()
		dim = len(x)
		data_points.append(x)

for i in range(k):
	mu.append(data_points[i])
#========================================================   K-MEANS   ==============================================

# plt.ion()
while(flag):
	
	flag = False
	cluster = []
	old_mu = []
	for i in range(k):
		old_mu.append(mu[i])
	for i in range(k):
		cluster.append([])
		# plt.plot(mu[i][0],mu[i][1],'bo',markersize=6,marker='v',color='black')

	for i in range(len(data_points)):
		minimum = find_euc_distance(mu[0],data_points[i])
		index = 0
		for j in range(k):
			dist = find_euc_distance(mu[j],data_points[i])
			if minimum > dist:
				minimum = dist
				index = j
		cluster[index].append(data_points[i])

	for i in range(k):
		x,y = zip(*cluster[i])
		# plt.scatter(x,y,s=10,color = (Colour[i][0],Colour[i][1],Colour[i][2]))
		mu[i] = find_new_mean(cluster[i])

	flag = check_Convergence(old_mu,mu)
	# plt.draw()
	# plt.pause(0.8)
	# plt.clf()

#=========================================================================================================================	

covariance = find_Covariance_Matrix(cluster,mu)
pi = calc_pi()
gaussian_mixture(mu,covariance,pi)