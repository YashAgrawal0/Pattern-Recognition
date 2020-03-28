import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
import random
import sys

def input(fileName):
	data_points = []
	f = open(fileName,"r")
	for i in f.readlines():
		x = [float(j) for j in i.split()]
		data_points.append(x)
		
	return np.array(data_points)

class1X = input("FDA/BOVW/train/c1_FDA13.txt")
class2X = input("FDA/BOVW/train/c3_FDA13.txt")

class1Y = np.zeros(len(class1X))
class2Y = np.zeros(len(class2X))

plt.scatter(class1X,class1Y,c="red")
plt.scatter(class2X,class2Y,c="green")
# plt.xlim(-30,20)
plt.savefig("BOVW13.png",bbox_inches="tight", pad_inches=0.5)
plt.show()