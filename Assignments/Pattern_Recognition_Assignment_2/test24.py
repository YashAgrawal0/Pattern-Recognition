import matplotlib.pyplot as plt
import numpy as np
import math
import random
import sys

k = int(sys.argv[1])

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

val = []

# for j in range(0,60):
# 	f = open("feature/"+str(j),"r")

# 	for i in f.readlines():
# 		x,y = i.split()
# 		x_val.append(x)
# 		y_val.append(y)

columns = 0
rows = 0
f1 = open("RD_group6/class1.txt","r")
for i in f1.readlines():
	x = i.split()
	rows += 1
	val.append(x)

for i in val[0]:
	columns += 1

mu,mu_dash=[],[]
print val[0]

for i in range(k):
	temp,tmp = [],[]
	for j in range(columns):
		x = random.randint(0,rows)
		temp.append(val[x][j])
		tmp.append(0)
	mu.append(temp)
	mu_dash.append(tmp)	

flag = True

plt.ion()
while flag:
	flag = False
	points = []
	for i in range(k):
		tmp = []
		for j in range(columns):
			tmp.append([])
		points.append([])
	
	for i in range(columns):
		for j in range(k):
			mu_dash[i][j] = mu[i][j]

	min_val=0

	for i in range(0,rows):
			index=0
			min_val = find_euc_distance(mu[i],val[i])
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

	plt.draw()
	plt.pause(0.8)
	plt.clf()