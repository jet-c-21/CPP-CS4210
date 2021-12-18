# -------------------------------------------------------------------------
# AUTHOR: Ta-Wei Chien (Jet)
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4200- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
from pprint import pprint as pp

db = []
X = []
Y = []

# reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)
            print(row)

# transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
# --> add your Python code here
ut_x = list()
ut_y = list()
for d in db:
    ut_x.append(d[0:4])
    ut_y.append(d[-1])

x_transform_dict = {
    0: {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3},  # Age
    1: {'Myope': 1, 'Hypermetrope': 2},  # Spectacle Prescription
    2: {'Yes': 1, 'No': 2},  # Tear Production Rate
    3: {'Normal': 1, 'Reduced': 2}  # Tear Production Rate
}

y_transform_dict = {
    'Yes': 1,
    'No': 2
}

for r in ut_x:
    td = list()
    for i in range(len(r)):
        attr = r[i]
        attr_t = x_transform_dict[i][attr]
        td.append(attr_t)
    X.append(td)
print(f'The transformed X:')
pp(X)

# transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
# --> addd your Python code here
for r in ut_y:
    Y.append(y_transform_dict[r])
print('The transformed Y:')
print(Y)

# fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, Y)

# plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes', 'No'], filled=True,
               rounded=True)
plt.show()
