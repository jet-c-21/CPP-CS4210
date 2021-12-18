# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/12/11
"""
import re


class UniqueList(list):
    def __init__(self, seq_str=None):
        """
        Init by sequence string example:
        '<Sunny, Overcast, Rain>'

        :param seq_str:
        """
        super().__init__()
        if isinstance(seq_str, str):
            seq_str = seq_str.replace('<', '').replace('>', '')
            for el in seq_str.split(','):
                el = el.strip()

                if re.match(r'^[-+]?[0-9]+$', el) is not None:
                    self.add_el(int(el))
                    continue

                if re.match(r'^-?\d+(?:\.\d+)$', el) is not None:
                    self.add_el(float(el))
                    continue

                self.add_el(el)

    def add_el(self, el: object):
        if el not in self:
            self.append(el)

    def update_els(self, seq):
        for el in seq:
            self.add_el(el)

    def to_list(self) -> list:
        return [el for el in self]

    def __repr__(self):
        return f"{[self[i] for i in range(len(self))]}".replace('[', '<').replace(']', '>')


if __name__ == '__main__':
    uls = UniqueList()

    uls.update_els([50, 21, 21, 7, 8, 9, 7, 9])
    uls.sort()

    print(uls)

    uls = UniqueList('<Sunny, Overcast, Rain>')
    print(uls)

    uls = UniqueList('<1,3.7,-2, -100, 200.8>')
    print(uls)
