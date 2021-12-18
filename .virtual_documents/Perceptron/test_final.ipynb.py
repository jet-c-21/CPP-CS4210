# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/12/13
"""
from perceptron import Perceptron

if __name__ == '__main__':
    train_X = [
        [-4],
        [-1],
        [3],
        [4],
        [9],
    ]

    train_y = [-1, -1, 1, 1, 1]
    custom_w_dict = {
        'xw': [-0.1, 0.1],  # w1, w2
        'bw': 0.1,  # w0
    }

    pct = Perceptron(lr=0.1, initial_weight=0, act_func='sgn')
    pct.fit(train_X, train_y)


pct.report



