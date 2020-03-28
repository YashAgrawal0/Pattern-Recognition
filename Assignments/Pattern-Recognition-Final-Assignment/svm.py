import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import classification_report, confusion_matrix

def input(fileName,data_points):
	tmp = open(fileName,"r")
	length = 0

	for i in tmp.readlines():
		length+=1

	f = open(fileName,"r")
	cnt = 0
	for i in f.readlines():
		if cnt == 0.75*length:
			break
		x = [float(j) for j in i.split()]
		data_points.append(x)
		cnt = cnt + 1
	return data_points,cnt

def inputTestData(fileName,data_points):
	tmp = open(fileName,"r")
	length = 0

	for i in tmp.readlines():
		length+=1

	f = open(fileName,"r")
	cnt = 0
	for i in f.readlines():
		if cnt >= 0.75*length:
			x = [float(j) for j in i.split()]
			data_points.append(x)
		cnt = cnt + 1
	val = int(0.25*length)	
	return data_points,val

def print_results(ConfusionMatrix):
	print "Confusion Matrix :\n",ConfusionMatrix, "\n"

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


	if ConfusionMatrix[0][0]==0 and ConfusionMatrix[1][0]==0 and ConfusionMatrix[2][0] == 0:
		precisionC1 = 0
	else :
		precisionC1=float(float(ConfusionMatrix[0][0])/float(ConfusionMatrix[0][0]+ConfusionMatrix[1][0]+ConfusionMatrix[2][0]))

	if ConfusionMatrix[0][1]==0 and ConfusionMatrix[1][1]==0 and ConfusionMatrix[2][1] == 0:
		precisionC2 = 0
	else :
		precisionC2=float(float(ConfusionMatrix[1][1])/float(ConfusionMatrix[0][1]+ConfusionMatrix[1][1]+ConfusionMatrix[2][1]))

	if ConfusionMatrix[0][0]==0 and ConfusionMatrix[1][0]==0 and ConfusionMatrix[2][0] == 0:
		precisionC3 = 0
	else :
		precisionC3=float(float(ConfusionMatrix[2][2])/float(ConfusionMatrix[0][2]+ConfusionMatrix[1][2]+ConfusionMatrix[2][2]))

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


train_datapoints, test_datapoints = [],[]

train_datapoints,length1 = input("DATA/NLS/class1.txt",train_datapoints)
train_datapoints,length2 = input("DATA/NLS/class2.txt",train_datapoints)
train_datapoints,length3 = input("DATA/NLS/class3.txt",train_datapoints)

TrainDataClass = np.zeros(length1+length2+length3)
for i in range(length1):
	TrainDataClass[i] = 0
for i in range(length2):
	TrainDataClass[i+length1] = 1
for i in range(length3):
	TrainDataClass[i+length1+length2] = 2


test_datapoints,length1 = inputTestData("DATA/NLS/class1.txt",test_datapoints)
est_datapoints,length2 = inputTestData("DATA/NLS/class2.txt",test_datapoints)
test_datapoints,length3 = inputTestData("DATA/NLS/class3.txt",test_datapoints)

TestDataClass = np.zeros(length1+length2+length3)
for i in range(length1):
	TestDataClass[i] = 0
for i in range(length2):
	TestDataClass[i+length1] = 1
for i in range(length3):
	TestDataClass[i+length1+length2] = 2


train_datapoints = np.array(train_datapoints)
test_datapoints = np.array(test_datapoints)

model = svm.SVC(kernel = "rbf", gamma = 100, C=1).fit(train_datapoints,TrainDataClass)
predict = model.predict(test_datapoints)


# scaleX,scaleY = np.meshgrid(np.arange(-6,6,0.05),np.arange(-6,6,0.05))

# res = model.predict (np.c_[scaleX.ravel(), scaleY.ravel()])
# res = res.reshape(scaleX.shape)
# plt.contourf(scaleX, scaleY, res, cmap = plt.cm.Paired, alpha = 1)

mat = confusion_matrix(TestDataClass,predict)
print "\n\n\n"
print_results(mat)

# plt.scatter (train_datapoints[:, 0], train_datapoints[:, 1], c = TrainDataClass, cmap = plt.cm.Paired)
# plt.xlabel ("X-axis")
# plt.ylabel ("Y-axis")
# plt.title ("Gaussian Kernel SVM on Non-Linearly separable data set at gamma = 1")
# plt.savefig("SVM_1_rbf_.png",bbox_inches="tight", pad_inches=0.5)
# plt.show ()







