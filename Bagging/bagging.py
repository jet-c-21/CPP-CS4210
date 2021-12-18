# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/12/11
"""
from statistics import mode
from unique_list import UniqueList
from split_rule import SplitRule
import pandas as pd

from pprint import pprint as pp
from decimal import Decimal as D


class Bagging:
    ROUND_RULE_COLS = [
        'Split Point',
        'Left Class',
        'Right Class',
    ]

    ROUND_RULE_INDEX_NAME = 'Round'

    def __init__(self, base: dict, rounds_x_ls: list):
        self.base = base
        self.round_x_ls = rounds_x_ls

        # parsed data
        self.x_val_ls = UniqueList()
        self.x_val_ls.update_els(self.base.keys())
        self.x_val_ls.sort()
        self.x_val_count = len(self.x_val_ls)
        self.min_x = min(self.x_val_ls)
        self.max_x = max(self.x_val_ls)
        self.test_x_ls = self.x_val_ls.to_list()

        self.y_cls_ls = UniqueList()
        self.y_cls_ls.update_els(self.base.values())
        self.y_cls_ls.sort()
        self.y_cls_count = len(self.y_cls_ls)
        self.min_y = min(self.y_cls_ls)
        self.max_y = max(self.y_cls_ls)

        self.round_y_ls = list()
        self.round_rule_ls = list()
        self.round_rule_df = pd.DataFrame(columns=Bagging.ROUND_RULE_COLS)
        self.round_rule_df.index.name = Bagging.ROUND_RULE_INDEX_NAME

        self.round_pred_df_cols = [f"x={x}" for x in self.x_val_ls]
        self.round_pred_df = pd.DataFrame(columns=self.round_pred_df_cols)

        # self.view_meta()
        self._parse()
        # self.round_rule_df_style = self.round_rule_df.style.set_properties(**{'text-align': 'center'})
        # self.round_rule_df_style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])

    @staticmethod
    def view_round_xy(rd_x, rd_y):
        print('x: ', end='')
        for x in rd_x:
            print(f"{str(x).rjust(3)} ", end='')
        print()

        print('y: ', end='')
        for y in rd_y:
            print(f"{str(y).rjust(3)} ", end='')
        print()

    # @staticmethod
    # def _get_split_rule(rd_x: list, rd_y: list, s_idx: int) -> SplitRule:
    #     left_els = rd_y[0:s_idx + 1]
    #     left_cls = mode(left_els)
    #     right_els = rd_y[s_idx: len(rd_y)]
    #     right_cls = mode(right_els)
    #     s_val = (rd_x[s_idx] + rd_x[s_idx + 1]) / 2
    #
    #     return SplitRule(s_val, left_cls, right_cls)

    @staticmethod
    def _get_s_idx_candidate(rd_y: list) -> list:
        result = list()
        if len(set(rd_y)) == 1:
            result.append(-1)
            return result

        i = 0
        while i < len(rd_y) - 1:
            if rd_y[i] != rd_y[i + 1]:
                result.append(i)
            i += 1

        # print(rd_y)
        # print(result)
        return result

    def view_meta(self):
        msg = 'self.base:'
        print(msg)
        pp(self.base)
        print()

        msg = f"self.x_val_ls:\n{self.x_val_ls}\n"
        print(msg)

        msg = f"self.x_val_count:\n{self.x_val_count}\n"
        print(msg)

        msg = f"self.min_x:\n{self.min_x}\n"
        print(msg)

        msg = f"self.max_x:\n{self.max_x}\n"
        print(msg)

        msg = f"self.y_cls_ls:\n{self.y_cls_ls}\n"
        print(msg)

        msg = f"self.y_cls_count:\n{self.y_cls_count}\n"
        print(msg)

        msg = f"self.min_y:\n{self.min_y}\n"
        print(msg)

        msg = f"self.max_y:\n{self.max_y}\n"
        print(msg)

        msg = 'self.round_y_ls:'
        print(msg)
        pp(self.round_y_ls)
        print()

        msg = 'self.round_rule_ls:'
        print(msg)
        pp(self.round_rule_ls)
        print()

        msg = f"self.round_rule_df:\n{self.round_rule_df}\n"
        print(msg)

        msg = f"self.round_pred_df:\n{self.round_pred_df}\n"
        print(msg)

    def _get_spilt_rule_origin_s_idx(self, rd_x: list, rd_y: list) -> list:
        """

        :param rd_x:
        :return: [SpiltRule, SpiltRule]
        """
        result = list()
        s_val = (D(rd_x[0]) + D(0)) / D(2)
        s_val = round(s_val, 2)
        s_val = float(s_val)

        y = mode(rd_y)

        result.append(SplitRule(s_val, y, y))

        # result.append(SplitRule(s_val, self.min_y, self.max_y))
        # result.append(SplitRule(s_val, self.max_y, self.min_y))

        return result

    def _get_split_rule_candidate_by_s_idx(self, rd_x: list, rd_y: list, s_idx: int) -> list:
        """

        :param rd_x:
        :param rd_y:
        :param s_idx:
        :return: [SpiltRule, SpiltRule, ..., SpiltRule]
        """
        if s_idx == -1:
            return self._get_spilt_rule_origin_s_idx(rd_x, rd_y)

        result = list()

        if rd_x[s_idx] == rd_x[s_idx + 1]:
            return result

        s_val = (D(rd_x[s_idx]) + D(rd_x[s_idx + 1])) / D(2)
        s_val = round(s_val, 2)
        s_val = float(s_val)

        left_els = rd_y[0:s_idx + 1]
        left_cls_ls = UniqueList()
        left_cls_ls.update_els(left_els)

        right_els = rd_y[s_idx: len(rd_y)]
        right_cls_ls = UniqueList()
        right_cls_ls.update_els(right_els)

        for left_cls in left_cls_ls:
            for right_cls in right_cls_ls:
                result.append(SplitRule(s_val, left_cls, right_cls))

        return result

    def _get_split_rule_candidate(self, rd_x: list, rd_y: list) -> list:
        result = list()
        s_idx_candidate = self._get_s_idx_candidate(rd_y)

        for s_idx in s_idx_candidate:
            if s_idx == -1:
                src = self._get_spilt_rule_origin_s_idx(rd_x, rd_y)
            else:
                src = self._get_split_rule_candidate_by_s_idx(rd_x, rd_y, s_idx)

            result.extend(src)

        return result

    def _parse(self):
        # fill records for round_y_ls, round_rule_ls, rule_df
        for rd_idx, rd_x in enumerate(self.round_x_ls):
            # if rd_idx != 3:  # debugger
            #     continue

            rd_y = [self.base[x] for x in rd_x]
            self.round_y_ls.append(rd_y)

            split_rule_candidate = self._get_split_rule_candidate(rd_x, rd_y)
            # for r in split_rule_candidate:
            #     print(r, '\n')

            best_rule = split_rule_candidate[0]
            error_rate = best_rule.get_error_rate(rd_x, rd_y)
            for i in range(1, len(split_rule_candidate)):
                new_rule = split_rule_candidate[i]
                new_rule: SplitRule
                new_e_r = new_rule.get_error_rate(rd_x, rd_y)
                if new_e_r < error_rate:
                    error_rate = new_e_r
                    best_rule = new_rule

            self.round_rule_ls.append(best_rule)

            # update round_rule_df
            self.round_rule_df.loc[rd_idx + 1] = best_rule.args

            # update round_pred_df
            pred_ls = best_rule.predict_xs(self.test_x_ls)
            # print(self.test_x_ls, pred_ls)
            self.round_pred_df.loc[f"Round {rd_idx + 1}"] = pred_ls

            # break

        # update sum row of round_pred_df
        self.round_pred_df.loc['Sum'] = self.round_pred_df.sum()
        self.round_pred_df.loc['Sign'] = self.round_pred_df.loc['Sum'].apply(lambda val: 1 if val > 0 else -1)

    def view_result(self):
        # print(self.round_x_ls)
        # print(self.round_y_ls)
        # print(self.round_rule_ls)

        for rd_idx in range(len(self.round_x_ls)):
            msg = f"Round {rd_idx + 1}"
            print(msg)

            rd_x = self.round_x_ls[rd_idx]
            rd_y = self.round_y_ls[rd_idx]
            rd_rule = self.round_rule_ls[rd_idx]
            self.view_round_xy(rd_x, rd_y)
            print(rd_rule)
            print()

        print(self.round_rule_df, '\n')
        print(self.round_pred_df, '\n')


if __name__ == '__main__':
    b_0 = {
        0.1: 1,
        0.2: 1,
        0.3: 1,
        0.4: -1,
        0.5: -1,
        0.6: -1,
        0.7: -1,
        0.8: 1,
        0.9: 1,
        1: 1

    }
    rd_x_ls_0 = [
        [0.1, 0.2, 0.2, 0.3, 0.4, 0.4, 0.5, 0.6, 0.9, 0.9],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.9, 1, 1, 1],
        [0.1, 0.2, 0.3, 0.4, 0.4, 0.5, 0.7, 0.7, 0.8, 0.9],
        [0.1, 0.1, 0.2, 0.4, 0.4, 0.5, 0.5, 0.7, 0.8, 0.9],
        [0.1, 0.1, 0.2, 0.5, 0.6, 0.6, 0.6, 1, 1, 1],

        [0.2, 0.4, 0.5, 0.6, 0.7, 0.7, 0.7, 0.8, 0.9, 1],
        [0.1, 0.4, 0.4, 0.6, 0.7, 0.8, 0.9, 0.9, 0.9, 1],
        [0.1, 0.2, 0.5, 0.5, 0.5, 0.7, 0.7, 0.8, 0.9, 1],
        [0.1, 0.3, 0.4, 0.4, 0.6, 0.7, 0.7, 0.8, 1, 1],
        [0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.8, 0.8, 0.9, 0.9],
    ]

    b_1 = {
        1: -1,
        2: -1,
        3: 1,
        4: 1,
        5: -1
    }

    rd_x_ls_1 = [
        [1, 1, 2, 4, 5],  # r1
        [3, 3, 4, 4, 5],  # r2
        [1, 2, 2, 5, 5],  # r3
        [1, 3, 4, 4, 5],  # r4
        [1, 2, 3, 3, 4],  # r5
    ]

    b_2 = {
        1: -1,
        2: 1,
        3: -1,
    }

    rd_x_ls_2 = [
        [1, 2, 2],
        [2, 2, 3],
        [1, 1, 3]
    ]

    # b = b_0
    # rd_x_ls = rd_x_ls_0

    b = b_1
    rd_x_ls = rd_x_ls_1

    # b = b_2
    # rd_x_ls = rd_x_ls_2

    bg = Bagging(b, rd_x_ls)
    bg.view_result()
