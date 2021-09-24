# -------------------------------------------------------------------------
# AUTHOR: Ta-Wei Chien (Jet)
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

# reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)

# print(db)


cw_dict = {
    'Correct': 0,
    'Wrong': 0,
}

# loop your data to allow each instance to be your test set
for i, instance in enumerate(db):
    # add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    # --> add your Python code here
    X = list()
    Y = list()
    Y_tr_dict = {
        '+': 1,
        '-': 2,
    }
    for j, r in enumerate(db):
        if i == j:
            continue
        X.append([float(r[0]), float(r[1])])

        # transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
        #  feature value to float to avoid warning messages
        # --> add your Python code here
        Y.append(Y_tr_dict[r[2]])

    # print(X)
    # print(Y)

    # store the test sample of this iteration in the vector testSample
    # --> add your Python code here
    testSample = [float(instance[0]), float(instance[1])]

    # fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    # use your test sample in this iteration to make the class prediction. For instance:
    # class_predicted = clf.predict([[1, 2]])[0]
    # --> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    # compare the prediction with the true label of the test instance to start calculating the error rate.
    # --> add your Python code here
    true_label = Y_tr_dict[instance[2]]

    if true_label == class_predicted:
        cw_dict['Correct'] += 1
    else:
        cw_dict['Wrong'] += 1

    # print(class_predicted, true_label)

# print the error rate
# --> add your Python code here
error_rate = cw_dict['Wrong'] / len(db)
error_rate_str = f"Error Rate = {cw_dict['Wrong']} / {len(db)} = {error_rate}"
print(error_rate_str)