from q1_a import Perceptron

train_X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

train_y = [0, 1, 1, 0]

pct = Perceptron(lr=0.4, initial_weight=1)
pct.fit(train_X, train_y)
print(pct.report)
