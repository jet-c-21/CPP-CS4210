# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/12/13
"""
from bagging import Bagging

if __name__ == '__main__':
    b = {
        1: -1,
        2: -1,
        3: 1,
    }

    rd_x_ls = [
        [1, 2, 2],
        [1, 3, 3],
        [2, 2, 3]
    ]

    bg = Bagging(b, rd_x_ls)
    bg.view_result()