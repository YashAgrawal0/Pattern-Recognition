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

# for i in f2.readlines():
# 	x,y = i.split()
# 	x_val.append(x)
# 	y_val.append(y)

muX,muY,X,Y=[],[],[],[]

for i in range(0,k):
	x = random.randint(0,len(x_val))
	muX.append(float(x_val[x]))
	muY.append(float(y_val[x]))
	X.append(0)
	Y.append(0)

flag = True

plt.ion()
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

	plt.draw()
	plt.pause(0.8)
	plt.clf()
