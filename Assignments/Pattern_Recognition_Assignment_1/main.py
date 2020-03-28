#==========================================================================================================

# 												Assignment - 1
#											---------------------

# Files							:	main.py, foo.png

# Test Data & Training Data 	:	class1.txt,class2.txt,class3.txt for differrent datasets

# Authors	 					:	Yash Agrawal(B16120), Aman Jain(B16044), Akhilesh Devrari(B16005)

# Created						:	10th September 2018


# 					** The program is used to test bayes classifier for different datasets **


#-----------------------------------------------------------------------------------------------------------
#			NOTE :   FOR MORE DETAILS ABOUT IMPLEMENTATION OF CODE, A README FILE IS PROVIDED
#-----------------------------------------------------------------------------------------------------------



#********************************************  Headers Included  *******************************************

import numpy as np
from numpy.linalg import inv
import math
import sys
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import matplotlib.lines as mlines

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
		covar[0][0]+=(x[i]-meanx)*(x[i]-meanx)
		covar[0][1]+=(x[i]-meanx)*(y[i]-meany)
		covar[1][0]+=(x[i]-meanx)*(y[i]-meany)
		covar[1][1]+=(y[i]-meany)*(y[i]-meany)

	covar[0][0]/=length 										# Covariance matrix is calculated
	covar[0][1]/=length 										# for the input two classes
	covar[1][0]/=length
	covar[1][1]/=length

	return covar

#----------------------------------------------------------------------------------------------------------

def plotClass(cls,colour,ax,className,position):
	x = []
	y = []

	for i in range(0,len(cls)):										# PlotClass function is used to
		x.append(cls[i][0])											# plot class taken input as argument
		y.append(cls[i][1])											# with given colour
	
	# ax.title(message);	
	temp = ax.scatter(x,y,color=colour,s=1,alpha=0.3,label=className)
	ax.set_xlabel('X1 ---->\n'+position)
	ax.set_ylabel('X2 ---->')
	ax.legend(loc=2,borderpad=0.5,ncol=3,
			  columnspacing=1,borderaxespad=0.1,
			  handletextpad=-0.2,prop={'size': 10},
			  fancybox=True,scatterpoints=1,
			  markerscale=10,shadow=True)

#----------------------------------------------------------------------------------------------------------

def makeClass(cls,c,length):
	file=open(c,"r")
	for i in range(0,length):										# Stores value into array from input
		line=file.readline()										# text file class.txt
		x,y = line.split()
		cls[i][0] = float(x)
		cls[i][1] = float(y)

	return cls	

#----------------------------------------------------------------------------------------------------------

def discriminant(mu1,mu2,x1,x2,covar,pc):							# Calculate value of g(x)
	a=x1-mu1
	b=x2-mu2
	covarInv=inv(covar)
	g = -0.5*float(a*a*covarInv[0][0] + a*b*(covarInv[1][0]+covarInv[0][1]) + covarInv[1][1]*b*b)
	det=np.linalg.det(covar)
	g += math.log(pc) - 0.5*float(math.log(det))

	return g

#----------------------------------------------------------------------------------------------------------

def background(color1X,color2X,color1Y,color2Y,g1,g2,i,j):
	if g1 > g2:
		color1X.append(i)
		color1Y.append(j)											# Background color for two classes
	else:
		color2X.append(i)
		color2Y.append(j)	

#----------------------------------------------------------------------------------------------------------

def backgroud(c1X,c1Y,c2X,c2Y,c3X,c3Y,g1,g2,g3,i,j):
	if g1 > g2 and g1 > g3:
		c1X.append(i)
		c1Y.append(j)
	elif g2>g3:
		c2X.append(i)												# Background color for 3 classes
		c2Y.append(j)												# together
	else:
		c3X.append(i)
		c3Y.append(j)

#----------------------------------------------------------------------------------------------------------

def findTestData(c,length):
	tf1,tf2=[],[]
	f=open(c,"r")
	counter=0
	for i in range(0,length):
		line = f.readline()
		if counter>=0.75*length:									# last 25% data is used as test data
			x,y = line.split()										
			tf1.append(float(x))
			tf2.append(float(y))
		
		counter+=1	

	return tf1,tf2	

#----------------------------------------------------------------------------------------------------------

def set_limit(ax,lowerX,lowerY,upperX,upperY):
	ax.set_xlim([lowerX,upperX])								# Setting limit of graph which is
	ax.set_ylim([lowerY,upperY])								# input as command line argument

#----------------------------------------------------------------------------------------------------------

def plotContour(cf1,cf2,ax,color):
	minX,minY=float('inf'),float('inf')
	maxX,maxY=-float('inf'),-float('inf')
	x,y = np.zeros(len(cf1)),np.zeros(len(cf2))

	for i in range(0,len(cf1)):
		x[i] = cf1[i]
	for i in range(0,len(cf2)):
		y[i] = cf2[i]
																	# Plot contour for the classes
	cov = np.cov(x, y)												
	lambda_, v = np.linalg.eig(cov)
	lambda_ = np.sqrt(lambda_)
	
	for j in xrange(1, 6):
	    ell = Ellipse(xy=(np.mean(x), np.mean(y)),
	                  width=lambda_[0]*j, height=lambda_[1]*j,
	                  angle=np.rad2deg(np.arccos(v[0, 0])),
	                  edgecolor=color)
	    ell.set_facecolor('none')
	    ax.add_artist(ell) 



#==============================================  Main Function  ==============================================

def mainFunction():
	classifierType = sys.argv[1]
	c1 = sys.argv[2]
	c2 = sys.argv[3]												# Class is taken as input dynamically
	c3 = sys.argv[4]												# using command line argument

	f1=open(c1,"r")
	f2=open(c2,"r")
	f3=open(c3,"r")

	# l1,l2,l3 = len(f1.readlines()),len(f1.readlines()),len(f1.readlines())

	l1 = int(len(f1.readlines()))
	l2 = int(len(f2.readlines()))
	l3 = int(len(f3.readlines()))

#------------------------------------------------------------------------------------------------------------

	lengthClass1=int(0.75*l1)
	lengthClass2=int(0.75*l2)										# 75% data is used as a training data
	lengthClass3=int(0.75*l3)	

	class1=np.zeros((lengthClass1,2))
	class2=np.zeros((lengthClass2,2))
	class3=np.zeros((lengthClass3,2))

	class1=makeClass(class1,c1,lengthClass1)
	class2=makeClass(class2,c2,lengthClass2)
	class3=makeClass(class3,c3,lengthClass3)

#------------------------------------------------------------------------------------------------------------

	pc1=float(lengthClass1)/float(lengthClass1+lengthClass2+lengthClass3)
	pc2=float(lengthClass2)/float(lengthClass1+lengthClass2+lengthClass3)
	pc3=float(lengthClass3)/float(lengthClass1+lengthClass2+lengthClass3)
	
	c1f1,c1f2,c2f1,c2f2,c3f1,c3f2=[],[],[],[],[],[]
	t1f1,t1f2,t2f1,t2f2,t3f1,t3f2=[],[],[],[],[],[]

	t1f1,t1f2=findTestData(c1,l1)
	t2f1,t2f2=findTestData(c2,l2)										# test data is stored in tifj array
	t3f1,t3f2=findTestData(c3,l3)										# for ith class and jth feature

	c1f1,c1f2=zip(*class1)
	c2f1,c2f2=zip(*class2)
	c3f1,c3f2=zip(*class3)

#------------------------------------------------------------------------------------------------------------

	c1mu1=getMean(c1f1)
	c1mu2=getMean(c1f2)								
	c1covar=getCovariance(c1f1,c1f2)

	c2mu1=getMean(c2f1)
	c2mu2=getMean(c2f2)
	c2covar=getCovariance(c2f1,c2f2)
	
	c3mu1=getMean(c3f1)
	c3mu2=getMean(c3f2)
	c3covar=getCovariance(c3f1,c3f2)

#---------------------------- For Case 'a' -------------------------------------------------------------------

	if classifierType=='a':
		covar=np.zeros((2,2))
		covarSum = float(c1covar[0][0]+c1covar[0][1]+c1covar[1][0]+c1covar[1][1])
		covarSum+= float(c2covar[0][0]+c2covar[0][1]+c2covar[1][0]+c2covar[1][1])
		covarSum+= float(c3covar[0][0]+c3covar[0][1]+c3covar[1][0]+c3covar[1][1])
		covarSum/=12.0
		covar[0][0]=covarSum
		covar[0][1]=0
		covar[1][0]=0
		covar[1][1]=covarSum
		c1covar=c2covar=c3covar=covar

#------------------------------ For case 'b' ----------------------------------------------------------------
		
	elif classifierType=='b':
		covar=np.zeros((2,2))
		covar[0][0]=float((c1covar[0][0]+c2covar[0][0]+c3covar[0][0])/3.0)
		covar[0][1]=float((c1covar[0][1]+c2covar[0][1]+c3covar[0][1])/3.0)
		covar[1][0]=float((c1covar[1][0]+c2covar[1][0]+c3covar[1][0])/3.0)
		covar[1][1]=float((c1covar[1][1]+c2covar[1][1]+c3covar[1][1])/3.0)
		c1covar=c2covar=c3covar=covar

#---------------------------- For case 'c' ------------------------------------------------------------------

	elif classifierType=='c':
		c1covar[0][1]=c1covar[1][0]=0
		c2covar[0][1]=c2covar[1][0]=0
		c3covar[0][1]=c3covar[1][0]=0

#---------------------------- For case 'd' -----------------------------------------------------------------

	elif classifierType!='d':
		print "Invalid Class Type"
		exit()

#------------------------------------------------------------------------------------------------------------

	confusionMat=np.zeros((3,3))
	
	for i in range(0, len(t1f1)):
		g1=discriminant(c1mu1,c1mu2,t1f1[i],t1f2[i],c1covar,pc1)
		g2=discriminant(c2mu1,c2mu2,t1f1[i],t1f2[i],c2covar,pc2)
		g3=discriminant(c3mu1,c3mu2,t1f1[i],t1f2[i],c3covar,pc3)
		
		if g1>g2 and g1>g3:
			confusionMat[0][0]+=1												# Test Data for class 1
		elif g2>g3:
			confusionMat[0][1]+=1
		else:
			confusionMat[0][2]+=1

#------------------------------------------------------------------------------------------------------------

	for i in range(0, len(t2f1)):
		g1=discriminant(c1mu1,c1mu2,t2f1[i],t2f2[i],c1covar,pc1)
		g2=discriminant(c2mu1,c2mu2,t2f1[i],t2f2[i],c2covar,pc2)
		g3=discriminant(c3mu1,c3mu2,t2f1[i],t2f2[i],c3covar,pc3)
		
		if g1>g2 and g1>g3:
			confusionMat[1][0]+=1												# Test Data for class 2
		elif g2>g3:
			confusionMat[1][1]+=1
		else:
			confusionMat[1][2]+=1

#------------------------------------------------------------------------------------------------------------

	for i in range(0, len(t3f1)):
		g1=discriminant(c1mu1,c1mu2,t3f1[i],t3f2[i],c1covar,pc1)
		g2=discriminant(c2mu1,c2mu2,t3f1[i],t3f2[i],c2covar,pc2)
		g3=discriminant(c3mu1,c3mu2,t3f1[i],t3f2[i],c3covar,pc3)
		
		if g1>g2 and g1>g3:														# Test Data for class 3
			confusionMat[2][0]+=1
		elif g2>g3:
			confusionMat[2][1]+=1
		else:
			confusionMat[2][2]+=1

#---------------------------------------- Giving output on terminal  ---------------------------------------

	print "Confusion Matrix :\n",confusionMat, "\n"

	accuracy = float((confusionMat[0][0]+confusionMat[1][1]+confusionMat[2][2])/(len(t1f1)+len(t2f1)+len(t3f1)))
	print "Accuracy : ", accuracy*100, "%\n"

	recallC1 = float(confusionMat[0][0]/len(t1f1))
	recallC2 = float(confusionMat[1][1]/len(t2f1))
	recallC3 = float(confusionMat[2][2]/len(t3f1))

	print "Recall :"
	print "Class 1 :   ",recallC1												# Output recall of
	print "Class 2 :   ",recallC2												# the classes
	print "Class 3 :   ",recallC3
	print "Mean Recall : ", (recallC1+recallC2+recallC3)/3
	print "\n"

#------------------------------------------------------------------------------------------------------------

	precisionC1=float((confusionMat[0][0])/(confusionMat[0][0]+confusionMat[1][0]+confusionMat[2][0]))
	precisionC2=float((confusionMat[1][1])/(confusionMat[0][1]+confusionMat[1][1]+confusionMat[2][1]))
	precisionC3=float((confusionMat[2][2])/(confusionMat[0][2]+confusionMat[1][2]+confusionMat[2][2]))

	print "Precision :"
	print "Class 1 :   ",precisionC1
	print "Class 2 :   ",precisionC2											# output Precision
	print "Class 3 :   ",precisionC3											# of the classes
	print "Mean Precision : ", (precisionC1+precisionC2+precisionC3)/3
	print "\n"

#------------------------------------------------------------------------------------------------------------

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
	print "Class 2 :   ",fmeasureC2												# output F-Measure
	print "Class 3 :   ",fmeasureC3												# of the classes
	print "Mean F-Measure : ", (fmeasureC1+fmeasureC2+fmeasureC3)/3
	print "\n"

#------------------------------------------------------------------------------------------------------------

	lgX,lgY,pX,pY,lbX,lbY=[],[],[],[],[],[]
	lgX1,lgY1,pX1,pY1=[],[],[],[]											# array initialization for
	pX2,pY2,lbX2,lbY2=[],[],[],[]											# different color background
	lbX3,lbY3,lgX3,lgY3=[],[],[],[]											# of different classes

#------------------------------------------------------------------------------------------------------------

	lowerX = float(sys.argv[5])
	upperX = float(sys.argv[6])
	lowerY = float(sys.argv[7])
	upperY = float(sys.argv[8])
	delX = float(float(upperX-lowerX)/200)
	delY = float(float(upperY-lowerY)/200)

	i=lowerX
	while(i<=upperX):
		j=lowerY
		while(j<=upperY):
			g1=discriminant(c1mu1,c1mu2,i,j,c1covar,pc1)					# getting value of g for each class
			g2=discriminant(c2mu1,c2mu2,i,j,c2covar,pc2)
			g3=discriminant(c3mu1,c3mu2,i,j,c3covar,pc3)
			
			background(lgX1,pX1,lgY1,pY1,g1,g2,i,j)
			background(pX2,lbX2,pY2,lbY2,g2,g3,i,j)							# according to g plot background
			background(lbX3,lgX3,lbY3,lgY3,g3,g1,i,j)
			backgroud(lgX,lgY,pX,pY,lbX,lbY,g1,g2,g3,i,j)
			j+=delY
		i+=delX

#------------------------------------------------------------------------------------------------------------

	fig = plt.figure()
	fig.subplots_adjust(wspace=0.5,hspace=0.4)
	ax1 = fig.add_subplot(2,2,1)
	ax2 = fig.add_subplot(2,2,2)
	ax3 = fig.add_subplot(2,2,3)
	ax4 = fig.add_subplot(2,2,4)
	
	set_limit(ax1,lowerX,lowerY,upperX,upperY)
	set_limit(ax2,lowerX,lowerY,upperX,upperY)
	set_limit(ax3,lowerX,lowerY,upperX,upperY)
	set_limit(ax4,lowerX,lowerY,upperX,upperY)

#------------------------------------------------------------------------------------------------------------

	class1fore = "red"
	class1back = "pink"
	class2fore = "green"
	class2back = "lightgreen"
	class3fore = "blue"
	class3back = "lightblue"

#------------------------------------------------------------------------------------------------------------

	ax1.scatter(lgX1,lgY1,color=class2back)#lightgreen
	ax1.scatter(pX1,pY1,color=class1back)#pink
	
	ax2.scatter(pX2,pY2,color=class1back)#pink
	ax2.scatter(lbX2,lbY2,color=class3back)#lightblue
			
	ax3.scatter(lbX3,lbY3,color=class3back)#lightblue						# plot background with color
	ax3.scatter(lgX3,lgY3,color=class2back)#lightgreen

	ax4.scatter(lgX,lgY,color=class2back)#lightgreen
	ax4.scatter(pX,pY,color=class1back)#pink
	ax4.scatter(lbX,lbY,color=class3back)#lightblue	

#------------------------------------------------------------------------------------------------------------

	plotClass(class1,class2fore,ax1,"class1","(a)")#green
	plotClass(class2,class1fore,ax1,"class2","(a)")#red
	
	plotClass(class1,class2fore,ax4,"class1","(d)")#green
	
	plotClass(class2,class1fore,ax2,"class2","(b)")#red
	plotClass(class3,class3fore,ax2,"class3","(b)")#blue				# plot scattered point of each class
	
	plotClass(class2,class1fore,ax4,"class2","(d)")#red
	
	plotClass(class3,class3fore,ax3,"class3","(c)")#blue
	plotClass(class1,class2fore,ax3,"class1","(c)")#green
	
	plotClass(class3,class3fore,ax4,"class3","(d)")#blue

#------------------------------------------------------------------------------------------------------------

	# plotContour(c1f1,c1f2,ax1,"darkgreen")
	# plotContour(c2f1,c2f2,ax1,"darkred")

	# plotContour(c2f1,c2f2,ax2,"darkred")
	# plotContour(c3f1,c3f2,ax2,"darkblue")							# plot contour of each class

	# plotContour(c3f1,c3f2,ax3,"darkblue")
	# plotContour(c1f1,c1f2,ax3,"darkgreen")

	# plotContour(c1f1,c1f2,ax4,"darkgreen")
	# plotContour(c2f1,c2f2,ax4,"darkred")
	# plotContour(c3f1,c3f2,ax4,"darkblue")

#------------------------------------------------------------------------------------------------------------

	plt.savefig('foo.png',bbox_inches="tight", pad_inches=0)

	ax1.plot(c1mu1, c1mu2, marker='*', markersize=5, color='k')
	ax1.plot(c2mu1, c2mu2, marker='*', markersize=5, color='k')

	ax2.plot(c3mu1, c3mu2, marker='*', markersize=5, color='k')
	ax2.plot(c2mu1, c2mu2, marker='*', markersize=5, color='k')				# plotting subplots of each
																			# pair of classes
	ax3.plot(c1mu1, c1mu2, marker='*', markersize=5, color='k')
	ax3.plot(c3mu1, c3mu2, marker='*', markersize=5, color='k')
	
	ax4.plot(c1mu1, c1mu2, marker='*', markersize=5, color='k')
	ax4.plot(c2mu1, c2mu2, marker='*', markersize=5, color='k')
	ax4.plot(c3mu1, c3mu2, marker='*', markersize=5, color='k')

	plt.show()
#===========================================================================================================

mainFunction()		#Calling Main Function

#------------------------------------------------------- End -----------------------------------------------