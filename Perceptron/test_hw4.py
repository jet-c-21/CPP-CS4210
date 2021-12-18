# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/12/13
"""
from perceptron import Perceptron

if __name__ == '__main__':
    print('Q1 - a')
    train_X = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    train_y = [0, 0, 0, 1]
    i_w = 1
    lr = 0.4

    pct = Perceptron(lr=lr, initial_weight=i_w)
    pct.fit(train_X, train_y)
    print(pct.report, '\n')

    print('Q1 - b')
    train_X = [
        [0],
        [1],
    ]

    train_y = [1, 0]
    i_w = 0
    lr = 0.1

    pct = Perceptron(lr=lr, initial_weight=i_w)
    pct.fit(train_X, train_y)
    print(pct.report, '\n')
