# -------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import tree
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
import csv
from pprint import pprint as pp

dbTraining = []
dbTest = []
X_training = []
Y_training = []
classVotes = []  # this array will be used to count the votes of each classifier

# reading the training data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
    reader = csv.reader(trainingFile)
    for i, row in enumerate(reader):
        dbTraining.append(row)

# reading the test data in a csv file
with open('optdigits.tes', 'r') as testingFile:
    reader = csv.reader(testingFile)
    for i, row in enumerate(reader):
        dbTest.append(row)

        # initializing the class votes for each test sample
        classVotes.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

print("Started my base and ensemble classifier ...")
"""
we will create 20 bootstrap samples here (k = 20). 
One classifier will be created for each bootstrap sample
"""


def get_training_xy(train_data) -> (list, list):
    train_x = list()
    train_y = list()
    for r in train_data:
        train_x.append(r[:-1])
        train_y.append(r[-1])
    return train_x, train_y


single_clf_cpc = 0  # single classifier correct predict count
for k in range(20):
    samples_count = len(dbTraining)
    bootstrapSample = resample(dbTraining, n_samples=samples_count, replace=True)
    # populate the values of X_training and Y_training by using the bootstrapSample
    # --> add your Python code here
    X_training, Y_training = get_training_xy(bootstrapSample)
    # print(X_training)
    # print(Y_training)

    # fitting the decision tree to the data
    clf = tree.DecisionTreeClassifier(criterion='entropy',
                                      max_depth=None)  # we will use a single decision tree without pruning it
    clf = clf.fit(X_training, Y_training)

    for i, testSample in enumerate(dbTest):
        """
        1. make the classifier prediction for each test sample and
           update the corresponding index value in classVotes.
           For instance, if your first base classifier predicted 2
           for the first test sample, then classVotes[0,0,0,0,0,0,0,0,0,0]
           will change to classVotes[0,0,1,0,0,0,0,0,0,0].

        2. Later, if your second base classifier predicted 3
           for the first test sample, then classVotes[0,0,1,0,0,0,0,0,0,0]
           will change to classVotes[0,0,1,1,0,0,0,0,0,0]

        3. Later, if your third base classifier predicted 3
           for the first test sample, then classVotes[0,0,1,1,0,0,0,0,0,0]
           will change to classVotes[0,0,1,2,0,0,0,0,0,0]

        4. this array will consolidate the votes of all classifier for all test samples
        """
        pred_cls = clf.predict([testSample[:-1]])[0]
        classVotes[i][int(pred_cls)] += 1  # update class vote

        if k == 0:
            """
            for only the first base classifier, compare the prediction with
            the true label of the test sample here to start calculating its accuracy
            """
            # --> add your Python code here
            truth_label = testSample[-1]
            if pred_cls == truth_label:
                single_clf_cpc += 1

    # for only the first base classifier, print its accuracy here
    if k == 0:
        # --> add your Python code here
        accuracy = single_clf_cpc / len(dbTest)
        print("Finished my base classifier (fast but relatively low accuracy) ...")
        print("My base classifier accuracy: " + str(accuracy))
        print("")
        # break

"""
now, compare the final ensemble prediction (majority vote in classVotes) 
for each test sample with the ground truth label 
to calculate the accuracy of the ensemble classifier (all base classifiers together)
"""
# --> add your Python code here
ens_clf_cpc = 0  # ensemble classifier correct predict count
for cls_vote_record, test_data in zip(classVotes, dbTest):
    ens_clf_pred = cls_vote_record.index(max(cls_vote_record))
    truth_label = int(test_data[-1])

    if ens_clf_pred == truth_label:
        ens_clf_cpc += 1

# printing the ensemble accuracy here
ens_clf_acc = ens_clf_cpc / len(dbTest)
print("Finished my ensemble classifier (slow but higher accuracy) ...")
print("My ensemble accuracy: " + str(ens_clf_acc))
print("")

print("Started Random Forest algorithm ...")

# Create a Random Forest Classifier
"""
n_estimators is the number of decision trees that will be generated by Random Forest. 
The sample of the ensemble method used before
"""
clf = RandomForestClassifier(n_estimators=20)
# Fit Random Forest to the training data
X_training, Y_training = get_training_xy(dbTraining)
clf.fit(X_training, Y_training)

"""
1. make the Random Forest prediction for each test sample. 
   Example: class_predicted_rf = clf.predict([[3, 1, 2, 1, ...]]
2. compare the Random Forest prediction for each test sample with the ground truth label to calculate its accuracy
"""
# --> add your Python code here
rf_clf_cpc = 0  # random forest classifier correct predict count
for testSample in dbTest:
    rf_clf_pred = clf.predict([testSample[:-1]])[0]
    truth_label = testSample[-1]
    if rf_clf_pred == truth_label:
        rf_clf_cpc += 1

# printing Random Forest accuracy here
rf_clf_acc = rf_clf_cpc / len(dbTest)
print("Random Forest accuracy: " + str(rf_clf_acc))
print("Finished Random Forest algorithm (much faster and higher accuracy!) ...")
