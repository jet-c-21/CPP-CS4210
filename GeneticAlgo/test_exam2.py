# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/12/13
"""
from genetic_algo import GeneticAlgo

if __name__ == '__main__':
    data_path_1 = 'exam2.csv'

    i_pop_1 = [
        '10101',
        '10010',
    ]

    col_repr_r_1 = [
        {'col': 'Humidity', 'val': '<High, Normal>', 'bin': False},
        {'col': 'Wind', 'val': '<Strong, Weak>', 'bin': False},
        {'col': 'PlayTennis', 'val': {'Yes': '1', 'No': '0'}, 'bin': True},
    ]

    cso_r_1 = {
        1: {'idx': (1, 2), 'mask': '11000'},
    }

    m_r_1 = {
        1: {'chrsm_bit_str': '10010', 'idx': 3}
    }

    data_path = data_path_1
    i_pop = i_pop_1
    col_repr_r = col_repr_r_1
    cso_r = cso_r_1
    m_r = m_r_1

    ga = GeneticAlgo(data_path, i_pop, col_repr_r, cso_r, m_r, select_best_count=0)
    ga.start_generation()
