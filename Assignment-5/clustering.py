# -------------------------------------------------------------------------
# AUTHOR: Ta-Wei Chien (Jet)
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

# reading the data by using Pandas library
df = pd.read_csv('training_data.csv', sep=',', header=None)

# assign your training data to X_training feature matrix
X_training = df

# run K-means testing different k values from 2 until 20 clusters
k_ls = list()
sil_coeff_ls = list()
highest_sil_coeff = 0
max_k_val = None
k_means = None
for k in range(2, 20 + 1):
    k_means = KMeans(n_clusters=k, random_state=0)
    k_means.fit(X_training)

    """
    for each k, calculate the silhouette_coefficient by using: 
        - silhouette_score(X_training, k_means.labels_)
        - find which k maximizes the silhouette_coefficient
    """
    k_ls.append(k)
    silhouette_coefficient = silhouette_score(X_training, k_means.labels_)
    sil_coeff_ls.append(silhouette_coefficient)

    if silhouette_coefficient > highest_sil_coeff:
        highest_sil_coeff = silhouette_coefficient
        max_k_val = k

"""
plot the value of the silhouette_coefficient 
for each k value of K-means so that we can see the best k
"""
# --> add your Python code here
plt.plot(k_ls, sil_coeff_ls)
plt.show()

# reading the validation data (clusters) by using Pandas library
# --> add your Python code here
validation_data = pd.read_csv('testing_data.csv', sep=',', header=None)

"""
assign your data labels to vector labels: 
    (you might need to reshape the row vector to a column vector)
    do this: np.array(df.values).reshape(1,<number of samples>)[0]
"""
# --> add your Python code here
labels = np.array(validation_data.values).reshape(1, len(validation_data.values))[0]

# Calculate and print the Homogeneity of this K-means clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, k_means.labels_).__str__())

# rung agglomerative clustering now by using the best value o k calculated before by kmeans
# Do it:
agg = AgglomerativeClustering(n_clusters=max_k_val, linkage='ward')
agg.fit(X_training)

# Calculate and print the Homogeneity of this agglomerative clustering
print("Agglomerative Clustering Homogeneity Score = " + metrics.homogeneity_score(labels, agg.labels_).__str__())
