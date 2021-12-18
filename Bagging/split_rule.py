# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/12/11
"""


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

    def get_error_rate(self, rd_x: list, rd_y: list) -> float:
        error_count = 0
        for x, gt in zip(rd_x, rd_y):
            pred = self.predict(x)
            if pred != gt:
                error_count += 1

        return error_count / len(rd_x)
