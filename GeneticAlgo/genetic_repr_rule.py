# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/12/12
"""
from unique_list import UniqueList


class GeneticReprRule(UniqueList):
    def __init__(self, name: str, seq_str=None, is_bin=False, bin_dict=None):
        self.name = name
        self.is_bin = is_bin
        self.bin_dict = bin_dict
        super().__init__(seq_str)

        # print(self.is_bin, self.bin_dict)
        if self.is_bin:
            self.bin_dict: dict
            for k, v in self.bin_dict.items():
                self.add_el(k)

    def __repr__(self):
        if self.is_bin:
            return f"{self.name}(b):{[self[i] for i in range(len(self))]}".replace('[', '<').replace(']', '>')
        else:
            return f"{self.name}:{[self[i] for i in range(len(self))]}".replace('[', '<').replace(']', '>')

    def get_g_bit_str(self, obj) -> str:
        if self.is_bin:
            return self.bin_dict[obj]

        result = ''
        for el in self:
            if obj == el:
                result += '1'
            else:
                result += '0'

        return result
