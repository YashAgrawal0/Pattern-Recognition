import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import random
import sys
import os

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
def load_all_files_in_folder(folder):
	files = []
	for filename in os.listdir(folder):
		files.append(filename)
	return files


def input_complete_data(folder):
	data_point = []

	files = load_all_files_in_folder(folder+"/Train/sa")
	for i in range(len(files)):
		name_file = folder +"/Train/sa/" + str(files[i])
		f=open(name_file,"r")

		for j in f.readlines():
			x = [float(k) for k in j.split()]
			data_point.append(x)

	files = load_all_files_in_folder(folder+"/Train/sai")
	for i in range(len(files)):
		name_file = folder +"/Train/sai/" + str(files[i])
		f=open(name_file,"r")

		for j in f.readlines():
			x = [float(k) for k in j.split()]
			data_point.append(x)

	files = load_all_files_in_folder(folder+"/Train/sau")
	for i in range(len(files)):
		name_file = folder +"/Train/sau/" + str(files[i])
		f=open(name_file,"r")

		for j in f.readlines():
			x = [float(k) for k in j.split()]
			data_point.append(x)

	files = load_all_files_in_folder(folder+"/Test/sa")
	for i in range(len(files)):
		name_file = folder +"/Test/sa/" + str(files[i])
		f=open(name_file,"r")

		for j in f.readlines():
			x = [float(k) for k in j.split()]
			data_point.append(x)

	files = load_all_files_in_folder(folder+"/Test/sai")
	for i in range(len(files)):
		name_file = folder +"/Test/sai/" + str(files[i])
		f=open(name_file,"r")

		for j in f.readlines():
			x = [float(k) for k in j.split()]
			data_point.append(x)

	files = load_all_files_in_folder(folder+"/Test/sau")
	for i in range(len(files)):
		name_file = folder +"/Test/sau/" + str(files[i])
		f=open(name_file,"r")

		for j in f.readlines():
			x = [float(k) for k in j.split()]
			data_point.append(x)
	
	dim=len(data_point[0])
	return np.array(data_point),dim


#====================================================================================================================================

def input_folder(folder):
	data_point = []

	files = load_all_files_in_folder(folder)
	for i in range(len(files)):
		name_file = folder +"/" + str(files[i])
		f=open(name_file,"r")

		for j in f.readlines():
			x = [float(k) for k in j.split()]
			data_point.append(x)
	return data_point


def kMeans(folder):
	data_points,dim = input_complete_data(folder)
	# print len(data_points)
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
			# plt.plot(mu[i][0],mu[i][1],'bo',markersize=6,marker='v',color='black')
	
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
			# plt.scatter(x,y,s=1,color = (Colour[i][0],Colour[i][1],Colour[i][2]))
			print "Print Length ",len(cluster[i])
			mu[i] = np.mean(cluster[i],axis=0)
		
		print str(np.linalg.norm(old_mu-mu))
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


#====================================================================================================================================



#----------------------------------------------------------------  Lakshman Rekha --------------------------------------------------

def vector_quantise(folder,mu):
	f1 = open(folder+"_"+str(k)+".txt","w")
	files = load_all_files_in_folder(folder)
	for v in range(len(files)):
		name_file = folder +"/" + str(files[v])
		f=open(name_file,"r")

		for i in f.readlines():
			feature_vector = i.split()
			feature_vector = np.array(feature_vector).astype(float)
			min_dist=np.linalg.norm(feature_vector-mu[0])
			belongs_to = 0
			for j in range(1,k):
				dist = np.linalg.norm(feature_vector-mu[j])
				if(dist<min_dist):
					min_dist = dist
					belongs_to = j
			f1.write(str(belongs_to)+" ")
		f1.write("\n")
	f1.close()

#================================================================================================================================
		
MEAN,COVARIANCE,PI,data_points = kMeans("Group06")


vector_quantise("Group06/Train/sa",MEAN)
vector_quantise("Group06/Train/sai",MEAN)
vector_quantise("Group06/Train/sau",MEAN)

vector_quantise("Group06/Test/sa",MEAN)
vector_quantise("Group06/Test/sai",MEAN)
vector_quantise("Group06/Test/sau",MEAN)