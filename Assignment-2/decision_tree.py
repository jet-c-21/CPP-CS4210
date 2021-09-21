# -------------------------------------------------------------------------
# AUTHOR: Ta-Wei Chien (Jet)
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)


    def get_ut_xy(data) -> (list, list):
        ut_x = list()
        ut_y = list()
        for d in data:
            ut_x.append(d[0:4])
            ut_y.append(d[-1])

        return ut_x, ut_y


    ut_x, ut_y = get_ut_xy(dbTraining)


    # transform the original training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    # --> add your Python code here
    def transform_x(r_of_x):
        x_transform_dict = {
            0: {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3},  # Age
            1: {'Myope': 1, 'Hypermetrope': 2},  # Spectacle Prescription
            2: {'Yes': 1, 'No': 2},  # Tear Production Rate
            3: {'Normal': 1, 'Reduced': 2}  # Tear Production Rate
        }
        td = list()
        for i in range(len(r_of_x)):
            attr = r_of_x[i]
            attr_t = x_transform_dict[i][attr]
            td.append(attr_t)

        return td


    def get_transformed_x(ut_x):
        result = list()

        for r in ut_x:
            td = transform_x(r)
            result.append(td)

        return result


    X = get_transformed_x(ut_x)


    # print(X)

    # transform the original training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    # --> add your Python code here
    def transform_y(r_of_y):
        y_transform_dict = {
            'Yes': 1,
            'No': 2
        }
        return y_transform_dict[r_of_y]


    def get_transformed_y(ut_y):
        result = list()

        for r in ut_y:
            td = transform_y(r)
            result.append(td)

        return result


    Y = get_transformed_y(ut_y)


    # print(Y)

    def read_csv_as_ls(fp):
        result = list()
        with open(fp, 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:  # skipping the header
                    result.append(row)
        return result


    # loop your training and test tasks 10 times here
    accuracy_record = list()
    for i in range(10):

        # fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3)
        clf = clf.fit(X, Y)

        # read the test data and add this data to dbTest
        # --> add your Python code here
        dbTest = read_csv_as_ls('contact_lens_test.csv')

        eva_dict = {
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0,
        }

        for data in dbTest:
            test_X = transform_x(data[0:4])
            # print(test_X)
            # transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
            # class_predicted = clf.predict([[3, 1, 2, 1]])[0]           -> [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            # --> add your Python code here
            class_predicted = clf.predict([test_X])[0]
            # print(class_predicted)

            # compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            # --> add your Python code here
            true_label = transform_y(data[4])
            # print(true_label)

            if class_predicted == 1:
                if true_label == 1:
                    eva_dict['tp'] += 1
                else:
                    eva_dict['fp'] += 1
            else:  # class_predicted == 2
                if true_label == 1:
                    eva_dict['fn'] += 1
                else:
                    eva_dict['tn'] += 1


        # print(eva_dict)

        # find the lowest accuracy of this model during the 10 runs (training and test set)
        # --> add your Python code here
        def get_accuracy(eva_dict):
            return (eva_dict['tp'] + eva_dict['tn']) / \
                   (eva_dict['tp'] + eva_dict['tn'] + eva_dict['fp'] + eva_dict['fn'])


        accuracy_record.append(get_accuracy(eva_dict))

    # print the lowest accuracy of this model during the 10 runs (training and test set) and save it.
    # your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    # --> add your Python code here
    tr_res_msg = f"final accuracy when training on {ds}: {min(accuracy_record)}"
    print(tr_res_msg)
