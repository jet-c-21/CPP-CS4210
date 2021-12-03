from itertools import combinations

comb = combinations(['Beer', 'Diapers', 'Diaper', 'Milk'], 2)

# Print the obtained combinations
for i in list(comb):
    print(i)