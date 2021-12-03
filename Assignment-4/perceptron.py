# -------------------------------------------------------------------------
# AUTHOR: Ta-Wei Chien (Jet)
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier  # pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]  # learning rate
r = [True, False]  # random state

df = pd.read_csv('optdigits.tra', sep=',', header=None)  # reading the data by using Pandas library

X_training = np.array(df.values)[:, :64]  # getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:, -1]  # getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None)  # reading the data by using Pandas library

X_test = np.array(df.values)[:, :64]  # getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:, -1]  # getting the last field to form the class label for test


def get_clf_accuracy(clf, X_test, y_test) -> float:
    correct_count = 0
    for x, ground_truth in zip(X_test, y_test):
        pred = clf.predict([x])[0]
        # print(f"pred: {pred}<{type(pred)}> gt: {ground_truth}<{type(ground_truth)}>")
        if pred == ground_truth:
            correct_count += 1

    return correct_count / len(y_test)


highest_accuracy = 0
for w in n:  # iterates over n (learning rate options)

    for b in r:  # iterates over r

        for a in range(2):  # iterates over the algorithms

            # Create a Neural Network classifier
            if a == 0:
                clf = Perceptron(eta0=w, random_state=b, max_iter=1000)
                """
                https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html
                
                eta0 : double, default=1
                       Constant by which the updates are multiplied.
                       (learning rate)
                       
                random_state : int, RandomState instance, default=None
                               Used to shuffle the training data, when shuffle is set to True. 
                               Pass an int for reproducible output across multiple function calls.
                
                max_iter : int, default=1000
                           The maximum number of passes over the training data (aka epochs). 
                           It only impacts the behavior in the fit method, and not the partial_fit method.
                           New in version 0.19.
                """
            else:
                clf = MLPClassifier(activation='logistic',
                                    learning_rate_init=w,
                                    hidden_layer_sizes=(25,),
                                    shuffle=b,
                                    max_iter=1000)
                """
                https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
                learning_rate_init : learning rate, 
                hidden_layer_sizes : number of neurons in the ith hidden layer               
                """

            # Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            """
            - make the classifier prediction for each test sample and start computing its accuracy
            
            - hint: to iterate over two collections simultaneously with zip() 
                    Example: for (x_testSample, y_testSample) in zip(X_test, y_test):
            
            - to make a prediction do: clf.predict([x_testSample])
            """
            acc = get_clf_accuracy(clf, X_test, y_test)

            """
            - check if the calculated accuracy is higher than the previously one 
              calculated for each classifier. 
            - If so, update the highest accuracy and print it together with the network hyper-parameters
            - Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, random_state=True"
            - Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, random_state=False"
            """
            if acc > highest_accuracy:
                highest_accuracy = acc
                if a == 0:
                    msg = f"Highest Perceptron accuracy so far: {round(highest_accuracy, 2)}, Parameters: learning rate={w}, random_state={b}"
                    print(msg)
                else:
                    msg = f"Highest MLP accuracy so far: {round(highest_accuracy, 2)}, Parameters: learning rate={w}, random_state={b}"
                    print(msg)
