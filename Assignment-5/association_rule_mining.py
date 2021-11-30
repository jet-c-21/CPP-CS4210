# -------------------------------------------------------------------------
# AUTHOR: Ta-Wei Chien (Jet)
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4200 - Assignment #5
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

# find the unique items all over the data an store them in the set below
item_set = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    item_set = item_set.union(set(items))

# remove nan (empty) values by using:
item_set.remove(np.nan)

"""
- To make use of the apriori module given by mlxtend library, 
  we need to convert the dataset accordingly. 

- Apriori module requires a DataFrame that has either 0 and 1 
  or True and False as data.

- Example:
    Bread Wine Eggs
    1     0    1
    0     1    1
    1     1    1

To do that:
1. create a dictionary (labels) for each transaction
2. store the corresponding values for each item 
   (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
3. and when is completed, append the dictionary to the list encoded_val_ls 
   below (this is done for each transaction)
"""


# -->add your python code below
def get_empty_labels_dict() -> dict:
    return {item: 0 for item in item_set}


encoded_val_ls = []
for r_idx, row in df.iterrows():
    labels = get_empty_labels_dict()
    for c_idx, col in row.iteritems():
        if col in labels.keys():
            labels[col] += 1
    encoded_val_ls.append(labels)

# adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_val_ls)  # ohe_df = one-hot-encoding-DataFrame
print(ohe_df)

# calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

"""
iterate the rules data frame and 
print the apriori algorithm results by using the following format:
    Meat, Cheese -> Eggs
    Support: 0.21587301587301588
    Confidence: 0.6666666666666666
    Prior: 0.4380952380952381
    Gain in Confidence: 52.17391304347825
"""
# -->add your python code below
rules["antecedents"] = rules["antecedents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
rules["consequents"] = rules["consequents"].apply(lambda x: ', '.join(list(x))).astype("unicode")
print(rules)

"""
1. To calculate the prior and gain in confidence, 
   find in how many transactions the consequent of the rule appears 
   (the supportCount below). 

2. Then, use the gain formula provided right after.
   prior (is the number of transactions) = supportCount / len(encoded_val_ls) -> encoded_val_ls 
    print("Gain in Confidence: " + str( 100*(rule_confidence-prior) / prior ))
"""
# -->add your python code below
for i in range(len(rules)):
    print(f"{rules['antecedents'][i]} -> {rules['consequents'][i]}")
    print(f"Support: {rules['support'][i]}")
    print(f"Confidence: {rules['confidence'][i]}")

    temp = rules["consequents"][i]
    supportCount = 0
    for key in encoded_val_ls:
        if key[temp] == 1:
            supportCount += 1

    prior = supportCount / len(encoded_val_ls)
    print(f"Prior: {prior}")

    rule_confidence = rules['confidence'][i]
    print(f"Gain in Confidence: {100 * ((rule_confidence - prior) / prior)}")
    print()

# Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()
