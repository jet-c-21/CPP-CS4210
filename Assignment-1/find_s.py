# -------------------------------------------------------------------------
# AUTHOR: Ta-Wei Chien (Jet)
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4200- Assignment #1
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# importing some Python libraries
import csv

num_attributes = 4
db = list()
print("\n The Given Training Data Set \n")

# reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)
            print(row)

print("\n The initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes  # representing the most specific possible hypothesis
print(hypothesis)

# find the first positive training data in db and assign it to the vector hypothesis
##--> add your Python code here
positive_data = list()
for d in db:
    if d[-1] == 'Yes':
        positive_data.append(d)

first_pd = positive_data.pop(0)
print(f"The first positive training data: {first_pd}")
hypothesis[0], hypothesis[1], hypothesis[2], hypothesis[3] = first_pd[0], first_pd[1], first_pd[2], first_pd[3]
print(f"The hypothesis after first positive training data: {hypothesis}")


# find the maximally specific hypothesis according to your training data in db and assign it to the vector hypothesis (special characters allowed: "0" and "?")
##--> add your Python code here
def hypo_handel(hypo, data):
    for i in range(len(hypo)):
        if hypo[i] == '?':
            continue
        if hypo[i] != data[i]:
            hypo[i] = '?'


print('Find the maximally specific hypothesis:')
for d in positive_data:
    hypo_handel(hypothesis, d)
    print(f'New hypothesis: {hypothesis}')

print("\n The Maximally Specific Hypothesis for the given training examples found by Find-S algorithm:\n")
print(hypothesis)
