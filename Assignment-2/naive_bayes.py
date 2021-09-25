# -------------------------------------------------------------------------
# AUTHOR: Ta-Wei Chien (Jet)
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv


# reading the training data
# --> add your Python code here
def read_csv_to_ls(fp: str) -> list:
    d = list()
    with open(fp, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                d.append(row)

    return d


training_data = read_csv_to_ls('weather_training.csv')

# transform the original training features to numbers and add to the 4D array X. For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
# --> add your Python code here
x_trs_dict = {
    1: {'Sunny': 1, 'Overcast': 2, 'Rain': 3},  # Outlook
    2: {'Hot': 1, 'Mild': 2, 'Cool': 3},  # Temperature
    3: {'Normal': 1, 'High': 2},  # Humidity
    4: {'Strong': 1, 'Weak': 2},  # Wind
}

y_trs_dict = {
    'Yes': 1,
    'No': 2,
}

y_trs_dict_to_str = {
    1: 'Yes',
    2: 'No',
}

# def transform_x(ut_x:list):
#     result = list()
#     for i in range(len(ut_x)):
#         result.append(x_trs_dict[])

X, Y = list(), list()
for r in training_data:
    r_of_x = list()
    for i in range(len(r)):
        if i == 0:
            continue

        # transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
        if i == 5:
            Y.append(y_trs_dict[r[i]])
            continue

        r_of_x.append(x_trs_dict[i][r[i]])

    X.append(r_of_x)

# fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

# reading the data in a csv file
# --> add your Python code here
test_data = read_csv_to_ls('weather_test.csv')

# printing the header os the solution
print("Day".ljust(15) + "Outlook".ljust(15) + "Temperature".ljust(15) + "Humidity".ljust(15) + "Wind".ljust(
    15) + "PlayTennis".ljust(15) + "Confidence".ljust(15))

# use your test samples to make probabilistic predictions.
# --> add your Python code here
# -->predicted = clf.predict_proba([[3, 1, 2, 1]])[0]

test_X = list()
for r in test_data:
    r_of_x = list()
    for i in range(len(r)):
        if i == 0 or i == 5:
            continue

        r_of_x.append(x_trs_dict[i][r[i]])
    test_X.append(r_of_x)

for i, r_of_test_x in enumerate(test_data):
    # print(r_of_test_x)
    # print(test_X[i])
    predicted = clf.predict_proba([test_X[i]])[0]
    pred_cls = clf.predict([test_X[i]])[0]

    pred_cls_str = y_trs_dict_to_str[pred_cls]
    confidence = round(max(predicted), 2)
    # print(r_of_test_x)
    # print(pred_cls_str, confidence)
    if confidence >= 0.75:
        print(f"{r_of_test_x[0]}".ljust(15) + f"{r_of_test_x[1]}".ljust(15) + f"{r_of_test_x[2]}".ljust(
            15) + f"{r_of_test_x[3]}".ljust(15) + f"{r_of_test_x[4]}".ljust(
            15) + f"{pred_cls_str}".ljust(15) + f"{confidence}".ljust(15))
