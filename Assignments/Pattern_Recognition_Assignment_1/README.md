#CS669: Pattern Recognition

##Programming Assignment 1 : Bayes classifier

###Steps :

Run this command: python main.py option_case class1.txt class2.txt class3.txt lower_x_limit upper_x_limit lower_y_limit upper_y_limit

option_case : Option to select which case of covariance matrix is used to calculate decision boundary
				Use one of following options:
				a : Covariance matrix for all the classes is the same and is (σ^2)I
				b : Full Covariance matrix for all the classes is the same and is Σ
				c : Covariance matric is diagonal and is different for each class.
				d : Full Covariance matrix for each class is different

class1.txt class2.txt class3.txt are path to files containing given training and test data. First 75% data in each file is used for training and remaining 25% data is used for testing.

lower_x_limit upper_x_limit lower_y_limit upper_y_limit are limits used to scale plots.
	Suggested values are:
	Linearly separable artificial data : -10 25 -20 15
	Non-linearly separable artificial data : -10 10 -10 10
	Real world data : 0 1000 0 1400
