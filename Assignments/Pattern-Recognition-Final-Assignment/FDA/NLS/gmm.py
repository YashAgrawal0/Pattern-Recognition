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
	
	# for j in range(1):
		# f = open("c"+str(j+1)+"_FDA12.txt","r")
	f = open(fileName,"r")
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
	# itr = 1
	# iterationsX,loglikely = [],[]
	# f = open("train/aqueduct/log_vs_itr/k"+str(k)+"_class1.txt","w")

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

		# loglikely.append(l_theta)
		# iterationsX.append(itr)
		# print >> f,itr,l_theta
		# itr += 1
		print abs(l_theta - l_theta_old)

	return mean,covariance,pi
	# plt.plot(iterationsX,loglikely)
	# plt.xlabel("Iterations -->")
	# plt.ylabel("Log_Likelihood -->")
	# plt.savefig("train/aqueduct/log_vs_itr/k"+str(k)+"_class1.png",bbox_inches="tight", pad_inches=0.5)
	# plt.show()


#====================================================================================================================================


# MEAN,COVARIANCE,PI,data_points = kMeans("train/c1_FDA12.txt")

# GMM(MEAN,COVARIANCE,PI,data_points)

#======================================================================================

def findMCP(fileName,MEAN,COVARIANCE,PI):
	mean, cov, pi, data_points = kMeans(fileName)
	mean_gmm, cov_gmm, pi_gmm = GMM(mean, cov, pi, data_points)
	MEAN.append(mean_gmm)
	COVARIANCE.append(cov_gmm)
	PI.append(pi_gmm)	
	return MEAN,COVARIANCE,PI

#========================================================================================================================

def findClass12(data,MEAN, COVARIANCE, PI):			#data -> c1w1,2,3[0]
	l = np.zeros(2)

	sumOverAllClusters = 0.0
	for clusterNum in range(k):
		sumOverAllClusters += (PI[0][clusterNum])*findN(data, MEAN[0][clusterNum], COVARIANCE[0][clusterNum])
	if(sumOverAllClusters < 0.00000000001):
		sumOverAllClusters = 0.00000000001
	l[0] = math.log(sumOverAllClusters)


	sumOverAllClusters = 0.0
	for clusterNum in range(k):
		sumOverAllClusters += (PI[1][clusterNum])*findN(data, MEAN[1][clusterNum], COVARIANCE[1][clusterNum])
	if(sumOverAllClusters < 0.00000000001):
		sumOverAllClusters = 0.00000000001
	l[1] = math.log(sumOverAllClusters)

	if l[0] > l[1]:
		return 0
	else :
		return 1	

def findClass23(data,MEAN, COVARIANCE, PI):			#data -> c2w1,2,3[0]
	l = np.zeros(2)

	sumOverAllClusters = 0.0
	for clusterNum in range(k):
		sumOverAllClusters += (PI[2][clusterNum])*findN(data, MEAN[2][clusterNum], COVARIANCE[2][clusterNum])
	if(sumOverAllClusters < 0.00000000001):
		sumOverAllClusters = 0.00000000001
	l[0] = math.log(sumOverAllClusters)


	sumOverAllClusters = 0.0
	for clusterNum in range(k):
		sumOverAllClusters += (PI[3][clusterNum])*findN(data, MEAN[3][clusterNum], COVARIANCE[3][clusterNum])
	if(sumOverAllClusters < 0.00000000001):
		sumOverAllClusters = 0.00000000001
	l[1] = math.log(sumOverAllClusters)

	if l[0] > l[1]:
		return 1
	else :
		return 2	

def findClass13(data,MEAN, COVARIANCE, PI):			#data -> c3w1,2,3[0]
	l = np.zeros(2)

	sumOverAllClusters = 0.0
	for clusterNum in range(k):
		sumOverAllClusters += (PI[4][clusterNum])*findN(data, MEAN[4][clusterNum], COVARIANCE[4][clusterNum])
	if(sumOverAllClusters < 0.00000000001):
		sumOverAllClusters = 0.00000000001
	l[0] = math.log(sumOverAllClusters)


	sumOverAllClusters = 0.0
	for clusterNum in range(k):
		sumOverAllClusters += (PI[5][clusterNum])*findN(data, MEAN[5][clusterNum], COVARIANCE[5][clusterNum])
	if(sumOverAllClusters < 0.00000000001):
		sumOverAllClusters = 0.00000000001
	l[1] = math.log(sumOverAllClusters)

	if l[0] > l[1]:
		return 0
	else :
		return 2	

#==============================================================================================================


def print_results(ConfusionMatrix):
	print "\n\n\n\n\n\nConfusion Matrix :\n",ConfusionMatrix, "\n"

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

#==============================================================================================================


MEAN = []
COVARIANCE = []
PI = []

MEAN,COVARIANCE,PI = findMCP("train/c1_FDA12.txt",MEAN,COVARIANCE,PI)
MEAN,COVARIANCE,PI = findMCP("train/c2_FDA12.txt",MEAN,COVARIANCE,PI)
MEAN,COVARIANCE,PI = findMCP("train/c2_FDA23.txt",MEAN,COVARIANCE,PI)
MEAN,COVARIANCE,PI = findMCP("train/c3_FDA23.txt",MEAN,COVARIANCE,PI)
MEAN,COVARIANCE,PI = findMCP("train/c1_FDA13.txt",MEAN,COVARIANCE,PI)
MEAN,COVARIANCE,PI = findMCP("train/c3_FDA13.txt",MEAN,COVARIANCE,PI)

MEAN = np.array(MEAN)
COVARIANCE = np.array(COVARIANCE)
PI = np.array(PI)

Confusion_Matrix = np.zeros((3,3))

c1w1,dim = input("test/c1w1.txt")
c1w2,dim = input("test/c1w2.txt")
c1w3,dim = input("test/c1w3.txt")
c2w1,dim = input("test/c2w1.txt")
c2w2,dim = input("test/c2w2.txt")
c2w3,dim = input("test/c2w3.txt")
c3w1,dim = input("test/c3w1.txt")
c3w2,dim = input("test/c3w2.txt")
c3w3,dim = input("test/c3w3.txt")

# for i in range(len(c2w1)):
# 	print findClass12(c2w1[i],MEAN,COVARIANCE,PI)

# # print c1w1[0],c2w1[0],c3w1[0]
for j in range(len(c1w1)):
	x = findClass12(c1w1[j],MEAN,COVARIANCE,PI)
	y = findClass23(c1w2[j],MEAN,COVARIANCE,PI)
	z = findClass13(c1w3[j],MEAN,COVARIANCE,PI)
	clas = np.zeros(3)
	ind = 0
	clas[x] += 1
	clas[y] += 1
	clas[z] += 1
	mx = -1
	for i in range(3):
		if clas[i] > mx:
			ind = i
			mx = clas[i]
	Confusion_Matrix[0][ind] += 1

for j in range(len(c2w1)):
	x = findClass12(c2w1[j],MEAN,COVARIANCE,PI)
	y = findClass23(c2w2[j],MEAN,COVARIANCE,PI)
	z = findClass13(c2w3[j],MEAN,COVARIANCE,PI)
	clas = np.zeros(3)
	ind = 0
	clas[x] += 1
	clas[y] += 1
	clas[z] += 1
	mx = -1
	for i in range(3):
		if clas[i] > mx:
			ind = i
			mx = clas[i]
	Confusion_Matrix[1][ind] += 1

for j in range(len(c3w1)):
	x = findClass12(c3w1[j],MEAN,COVARIANCE,PI)
	y = findClass23(c3w2[j],MEAN,COVARIANCE,PI)
	z = findClass13(c3w3[j],MEAN,COVARIANCE,PI)
	clas = np.zeros(3)
	ind = 0
	clas[x] += 1
	clas[y] += 1
	clas[z] += 1
	mx = -1
	for i in range(3):
		if clas[i] > mx:
			ind = i
			mx = clas[i]
	Confusion_Matrix[2][ind] += 1		

print_results(Confusion_Matrix)