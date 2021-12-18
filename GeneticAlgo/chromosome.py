# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/12/12
"""
import pandas as pd
from typing import Union
from genetic_repr_rule import GeneticReprRule
from decimal import Decimal as D


class Chromosome:
    def __init__(self, chrsm_id: int, chrsm_bit_str: str,
                 f_col_repr_rule_ls: list, cls_col_repr_rule: GeneticReprRule):
        self.chrsm_id = chrsm_id
        self.name = f"C{self.chrsm_id + 1}"
        self.chrsm_bit_str = chrsm_bit_str
        self.f_col_repr_rule_ls = f_col_repr_rule_ls
        self.cls_col_repr_rule = cls_col_repr_rule
        self.fitness = None
        self.probability = None

        self.feat_bit_str = self.chrsm_bit_str[0:len(self.chrsm_bit_str) - 1]
        self.cls_bit_str = self.chrsm_bit_str[-1]
        self.cls_str = None
        self.flipped_cls_str = None
        for k, v in self.cls_col_repr_rule.bin_dict.items():
            if v == self.cls_bit_str:
                self.cls_str = k
            else:
                self.flipped_cls_str = k

        self.feat_dict = dict()
        self._parse_feat_str()

    def __repr__(self):
        s = f"{self.name}={self.chrsm_bit_str}"
        return s

    @staticmethod
    def replace_char_by_idx(s: str, idx: int, new_char):
        return s[0:idx] + new_char + s[idx + 1:]

    def _parse_feat_str(self):
        # print(self.f_col_repr_rule_ls)
        # print(f"feat str: {self.feat_str}")
        feat_str_parse_idx = 0
        for f_col_repr_rule in self.f_col_repr_rule_ls:
            f_col_repr_rule: GeneticReprRule
            f_name = f_col_repr_rule.name
            f_str = self.feat_bit_str[feat_str_parse_idx:feat_str_parse_idx + len(f_col_repr_rule)]
            # print(f"{f_name} : {f_str}, [{feat_str_parse_idx}:{len(f_col_repr_rule)}]")
            feat_str_parse_idx += len(f_col_repr_rule)

            f_val_set = set()
            for s, el in zip(f_str, f_col_repr_rule):
                if s == '1':
                    f_val_set.add(el)

            self.feat_dict[f_name] = f_val_set
            # break
        # print()
        # print(self.feat_dict)

    def predict(self, x: Union[pd.Series, dict]) -> str:
        # for pd.Series
        for col_name, val in x.iteritems():
            if col_name == self.cls_col_repr_rule.name:
                continue
            f_val_set = self.feat_dict[col_name]
            # print(set(val), f_val_set)
            if val not in f_val_set:
                return self.flipped_cls_str
        return self.cls_str

        # for dict

    def get_fitness(self, df: pd.DataFrame) -> float:
        correct_count = 0
        for _, row in df.iterrows():
            gt = row[self.cls_col_repr_rule.name]
            pred_cls = self.predict(row)
            if pred_cls == gt:
                correct_count += 1

            # break

        acc = D(correct_count) / D(len(df))
        acc = round(acc, 2)
        acc = float(acc)

        return acc

    def set_fitness(self, fitness: float):
        self.fitness = fitness

    def set_probability(self, probability: float):
        self.probability = probability

    def mutate(self, idx: int):
        if self.chrsm_bit_str[idx] == '1':
            new_char = '0'
        else:
            new_char = '1'

        # print(self.chrsm_bit_str)
        self.chrsm_bit_str = self.replace_char_by_idx(self.chrsm_bit_str, idx, new_char)
        # print(self.chrsm_bit_str)

        self.feat_bit_str = self.chrsm_bit_str[0:len(self.chrsm_bit_str) - 1]
        self.cls_bit_str = self.chrsm_bit_str[-1]

        self.cls_str = None
        self.flipped_cls_str = None
        for k, v in self.cls_col_repr_rule.bin_dict.items():
            if v == self.cls_bit_str:
                self.cls_str = k
            else:
                self.flipped_cls_str = k

        self.feat_dict = dict()
        self._parse_feat_str()

