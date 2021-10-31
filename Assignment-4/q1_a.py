from decimal import Decimal as D
from typing import Callable

import pandas as pd


class Perceptron:
    @staticmethod
    def heaviside(z) -> int:
        if z > 0:
            return 1
        else:
            return 0

    AF = {
        'heaviside': heaviside.__get__(object, object)
    }

    def __init__(self, lr=0.4, initial_weight=0.1, act_func='heaviside', x_b=1):
        self.lr = lr
        self.initial_weight = initial_weight
        self.act_func = act_func
        self.x_b = x_b
        self.w_b = initial_weight
        self.delta_w_b = 0
        self.w_ls = list()
        self.delta_w_ls = list()

    def _init_weight_and_report(self, X, y):
        self.X = X
        self.y = y
        self.report_columns = list()

        for i in range(len(self.X[0])):
            self.w_ls.append(self.initial_weight)
            self.delta_w_ls.append(None)
            self.report_columns.append(f"x{i + 1}")
        self.report_columns.append('xb')

        for i in range(len(self.w_ls)):
            self.report_columns.append(f"w{i + 1}")
        self.report_columns.append('wb')

        self.report_columns.append('z')
        self.report_columns.append('y')
        self.report_columns.append('t')
        self.report_columns.append('t-y')

        for i in range(len(self.w_ls)):
            self.report_columns.append(f"Δw{i + 1}")
        self.report_columns.append('Δwb')

        self.report = pd.DataFrame(columns=self.report_columns)

    def fit(self, X, y):
        self._init_weight_and_report(X, y)

        converge_flag = False

        while not converge_flag:
            pred_bias_record = list()

            for inst_x, gt in zip(X, y):
                ty_diff = self._train(inst_x, gt)
                pred_bias_record.append(ty_diff)

            converge_flag = self._is_converge(pred_bias_record)

        self.report.index += 1

    def _is_converge(self, pred_bias_record):
        for r in pred_bias_record:
            if r != 0:
                return False
        return True

    def _train(self, x, gt) -> int:
        z = self._get_z(x)
        pred_y = self._get_pred_y(z)
        ty_diff = gt - pred_y

        # update report
        report_record = self._get_report_record(x, gt, z, pred_y, ty_diff)
        self.report.loc[len(self.report)] = report_record

        # update weights
        for i in range(len(self.w_ls)):
            self.w_ls[i] = self._get_new_weight(self.w_ls[i], self.delta_w_ls[i])
        self.w_b = self._get_new_weight(self.w_b, self.delta_w_b)

        return ty_diff

    def _get_report_record(self, x, gt, z, pred_y, ty_diff) -> list:
        """
        generate report record and update delta-weights at the same time
        - delta-weights : 1. self.delta_w_ls
                          2. delta_w_b
        :param x:
        :param gt:
        :param z:
        :param pred_y:
        :param ty_diff:
        :return:
        """

        report_record = list()
        for fv in x:
            report_record.append(fv)
        report_record.append(self.x_b)

        for w in self.w_ls:
            report_record.append(w)
        report_record.append(self.w_b)

        report_record.append(z)
        report_record.append(pred_y)
        report_record.append(gt)
        report_record.append(ty_diff)

        # update delta weights
        for i in range(len(self.delta_w_ls)):
            fv = x[i]
            self.delta_w_ls[i] = self._get_delta_weight(ty_diff, fv)

        self.delta_w_b = self._get_delta_weight(ty_diff, self.x_b)

        for dw in self.delta_w_ls:
            report_record.append(dw)
        report_record.append(self.delta_w_b)

        return report_record

    def _get_delta_weight(self, ty_diff, fv) -> float:
        # print(f"{self.lr} * {ty_diff} * {fv}")
        d = D(str(self.lr)) * D(str(ty_diff)) * D(str(fv))
        d = float(d)

        if d == 0:
            return abs(d)
        return d

    def _get_new_weight(self, cw, dw) -> float:
        # print(f"{cw} : {type(cw)}  {dw} : {type(dw)}")
        d = D(str(cw)) + D(str(dw))
        d = float(d)

        if d == 0:
            return abs(d)
        return d
        # return cw + dw

    def _get_z(self, x) -> float:
        z = D('0')
        # feature-index feature-value
        for fi, fv in enumerate(x):
            z += D(str(fv)) * D(str(self.w_ls[fi]))

        z += D(str(self.x_b)) * D(str(self.w_b))

        return float(z)

    def _get_pred_y(self, z) -> Callable:
        return Perceptron.AF[self.act_func](z)

    def predict(self, X) -> list:
        result = list()
        for x in X:
            z = self._get_z(x)
            result.append(self._get_pred_y(z))
        return result


if __name__ == '__main__':
    train_X = [
        [20, 10],
        [80, 60],
        [-20, 10],
    ]
    train_y = [0, 1, 1]
    prcp = Perceptron(lr=0.2, initial_weight=0.1)
    prcp.fit(train_X, train_y)
    print(prcp.report)
    print(prcp.predict([[20, 50]])[0])
    print(prcp.predict([[20, 0]])[0])
