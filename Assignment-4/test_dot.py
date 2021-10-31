# -------------------------------------------------------------------------
# AUTHOR: Ta-Wei Chien (Jet)
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #4
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: YOU CAN USE ANY PYTHON LIBRARY TO COMPLETE YOUR CODE.

# importing the libraries
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def build_model(n_hidden, n_neurons_hidden, n_neurons_output, learning_rate):
    # --> add your Python code here
    # Creating the Neural Network using the Sequential API
    model = keras.models.Sequential()

    model.add(keras.layers.Flatten(input_shape=[28, 28]))  # input layer

    # iterate over the number of hidden layers to create the hidden layers:
    for _ in range(n_hidden):
        # hidden layer with ReLU activation function
        model.add(keras.layers.Dense(n_neurons_hidden, activation="relu"))

    """
    output layer with one neural for each class 
    and use the softmax activation function since the classes are exclusive
    """
    model.add(keras.layers.Dense(n_neurons_output, activation="softmax"))

    # defining the learning rate
    opt = keras.optimizers.SGD(learning_rate)

    # Compiling the Model specifying the loss function and the optimizer to use.
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


"""
- Using Keras to Load the Dataset. 
- Every image is represented as a 28×28 array rather than a 1D array of size 784. 
- Moreover, the pixel intensities are represented as integers (from 0 to 255) 
  rather than floats (from 0.0 to 255.0).
"""
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# creating a validation set and scaling the features
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

"""
For Fashion MNIST, we need the list of class names to know what we are dealing with. 
For instance, class_names[y_train[0]] = 'Coat'
"""
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


# Iterate here over number of hidden layers, number of neurons in each hidden layer and the learning rate.
# -->add your Python code here
def get_model_acc(model, X_test, y_test):
    correct_pred_count = 0
    class_predictions = np.argmax(model.predict(X_test), axis=-1)
    for pred, gt in zip(class_predictions, y_test):
        if pred == gt:
            correct_pred_count += 1

    return correct_pred_count / len(y_test)


n_hidden = [2]
n_neurons = [100]
l_rate = [0.1]

best_model = None
best_history = None
highest_accuracy = 0
best_param = {
    'h': None,
    'n': None,
    'lr': None,
}
for h in n_hidden:  # looking or the best parameters w.r.t the number of hidden layers
    for n in n_neurons:  # looking or the best parameters w.r.t the number of neurons
        for lr in l_rate:  # looking or the best parameters w.r.t the learning rate

            # build the model for each combination by calling the function:
            model = build_model(h, n, 10, lr)

            # To train the model
            """
            epochs: 
                number of times that the learning algorithm will 
                work through the entire training dataset.
            """
            history = model.fit(X_train, y_train,
                                epochs=5, validation_data=(X_valid, y_valid))

            # Calculate the accuracy of this neural network and store its value if it is the highest so far
            # -->add your Python code here
            model_acc = get_model_acc(model, X_test, y_test)
            # model_score = model.evaluate(X_test, y_test, verbose=True)  # return [v1 v2]

            if model_acc > highest_accuracy:
                highest_accuracy = model_acc
                best_param['h'] = h
                best_param['n'] = n
                best_param['lr'] = lr
                best_model = model
                best_history = history

                msg = f"Highest accuracy so far: {highest_accuracy}"
                print(msg)

                msg = f"Parameters - Number of Hidden Layers: f{h}, Number of Neurons: {n}, Learning Rate: {lr}"
                print(msg)
                print()

# After generating all neural networks, print the final weights and biases of the best model
model = best_model
history = best_history
weights, biases = model.layers[1].get_weights()
print(weights)
print(biases)

"""
The model’s summary() method displays:
    1. all the model’s layers, including each layer’s name 
       (which is automatically generated unless you set it when creating the layer) 
    
    2. its output shape (None means the batch size can be anything)
    
    3. its number of parameters. 
    
    - Note that Dense layers often have a lot of parameters. 
      This gives the model quite a lot of flexibility to fit the training data, 
      but it also means that the model runs the risk of over-fitting, 
      especially when you do not have a lot of training data.
"""
print(model.summary())
img_file = './model_arch.png'
tf.keras.utils.plot_model(model, to_file=img_file,
                          show_shapes=True, show_layer_names=True)

# plotting the learning curves of the best model
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()
