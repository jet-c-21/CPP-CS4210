# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/12/11
"""
from decimal import Decimal as D


class SplitRule:
    def __init__(self, s_val: float, left_cls: int, right_cls: int):
        self.s_val = s_val
        self.left_cls = left_cls
        self.right_cls = right_cls
        self.args = [self.s_val, self.left_cls, self.right_cls]

    def __repr__(self):
        x_r_j = 3
        y_r_j = 2

        s = f"x ≤ {str(self.s_val).rjust(x_r_j)} → y = {str(self.left_cls).rjust(y_r_j)}\n" \
            f"x > {str(self.s_val).rjust(x_r_j)} → y = {str(self.right_cls).rjust(y_r_j)}"

        return s

    def predict(self, x) -> int:
        if x <= self.s_val:
            return self.left_cls

        if x > self.s_val:
            return self.right_cls

    def predict_xs(self, xs) -> list:
        result = list()
        for x in xs:
            result.append(self.predict(x))

        return result

    def get_error_count(self, rd_x: list, rd_y: list) -> int:
        error_count = 0
        for x, gt in zip(rd_x, rd_y):
            pred = self.predict(x)
            if pred != gt:
                error_count += 1

        return error_count

    def get_error_rate(self, rd_x: list, rd_y: list) -> float:
        error_count = self.get_error_count(rd_x, rd_y)

        return error_count / len(rd_x)

    def get_boosting_error_rate(self, rd_x_w_ls: list, rd_x: list, rd_y: list):
        result = 0
        pr = D(1) / D(len(rd_x))

        pred_ls = self.predict_xs(rd_x)
        # print(pred_ls)

        temp = list()
        for x_w, pred, gt in zip(rd_x_w_ls, pred_ls, rd_y):
            # print(pred, gt)
            if pred != gt:
                temp.append((x_w, 1))

        # print(temp)

        if len(temp):
            t = D(0)
            msg = f"{pr} * [0 "
            print(msg, end='')
            for w, v in temp:
                msg = f"+ ({w} * {v})"
                print(msg, end='')
                t += D(w) * D(v)

            result = pr * t
            result = round(result, 3)
            result = float(result)
            msg = f"] = {result}"
            print(msg)

        return result
