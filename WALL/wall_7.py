# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/12/13
"""


def replace_char_by_idx(s: str, idx: int, new_char):
    return s[0:idx] + new_char + sample_str[idx+1:]


if __name__ == '__main__':
    sample_str = "This is a sample string"
    x = replace_char_by_idx(sample_str, 2, 'x')
    print(x)
