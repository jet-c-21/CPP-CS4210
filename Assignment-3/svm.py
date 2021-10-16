# -------------------------------------------------------------------------
# AUTHOR: Ta-Wei Chien (Jet)
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import svm
import csv

dbTest = []
X_training = []
Y_training = []
C = [1, 5, 10, 100]
DEGREE = [1, 2, 3]
KERNEL = ["linear", "poly", "rbf"]
DECISION_FUNCTION_SHAPE = ["ovo", "ovr"]
highestAccuracy = 0

# reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
    reader = csv.reader(trainingFile)
    for i, row in enumerate(reader):
        X_training.append(row[:-1])
        Y_training.append(row[-1])

# reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
    reader = csv.reader(testingFile)
    for i, row in enumerate(reader):
        dbTest.append(row)

"""
created 4 nested for loops that will iterate through 
the values of c, degree, kernel, and decision_function_shape
"""
# --> add your Python code here
for c in C:  # iterates over c
    for degree in DEGREE:  # iterates over degree
        for kernel in KERNEL:  # iterates kernel
            for dec_func_shape in DECISION_FUNCTION_SHAPE:  # iterates over decision_function_shape
                """
                Create an SVM classifier that will test 
                all combinations of c, degree, kernel, and decision_function_shape 
                as hyper-parameters. 
                For instance svm.SVC(c=1)
                """
                clf = svm.SVC(C=c,
                              degree=degree,
                              kernel=kernel,
                              decision_function_shape=dec_func_shape,
                              )

                # Fit SVM to the training data
                clf.fit(X_training, Y_training)

                # make the classifier prediction for each test sample and start computing its accuracy
                # --> add your Python code here
                curr_cpc = 0
                for test_sample in dbTest:
                    test_x = test_sample[:-1]
                    test_y = test_sample[-1]
                    class_predicted = clf.predict([test_x])[0]

                    if class_predicted == test_y:
                        curr_cpc += 1

                curr_acc = curr_cpc / len(dbTest)

                """
                check if the calculated accuracy is higher 
                than the previously one calculated. 
                If so, update update the highest accuracy and 
                print it together with the SVM hyper-parameters
                
                Example: 
                Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel=poly, decision_function_shape='ovo'
                """
                # --> add your Python code here
                if curr_acc > highestAccuracy:
                    highestAccuracy = curr_acc
                    msg = f"Highest SVM accuracy so far: {highestAccuracy}, Parameters: C={c}, degree={degree}, " \
                          f"kernel={kernel}, decision_function_shape={dec_func_shape} "
                    print(msg)
