from pprint import pprint as pp
import pandas as pd
from typing import Union
from numpy import dot
from numpy import mean
from numpy.linalg import norm
from decimal import Decimal as D


class CollaboratingFiltering:
    def __init__(self, raw_data: Union[str, pd.DataFrame], nature_val=1.5, k=2, thresh=2.0):
        self.raw_df = raw_data
        if isinstance(self.raw_df, str):
            self.raw_df = pd.read_csv(raw_data, index_col=0)
        self.nature_val = nature_val
        self.k = k
        self.thresh = thresh

        self.df = self.raw_df
        self._replace_missing_val()

    @staticmethod
    def get_cos_similarity(seq_1, seq_2) -> float:
        return dot(seq_1, seq_2) / (norm(seq_1) * norm(seq_2))

    @staticmethod
    def to_decimal(number) -> D:
        return D(str(number))

    @staticmethod
    def get_mean(seq, rounded=True, round_d=3) -> float:
        m = float(mean(seq))
        if rounded:
            m = round(m, round_d)
        return m

    def _get_top_cp_ls(self, cp_user_dict: dict) -> list:
        return sorted(cp_user_dict.items(), key=lambda x: x[1]['cs'], reverse=True)[:self.k]

    def _replace_missing_val(self):
        self.df: pd.DataFrame
        self.df = self.df.fillna(self.nature_val)

    def _get_user_base_task_ls(self) -> list:
        task_ls = list()
        task_df = self.df[self.df.isin(['?']).any(axis=1)]
        for row_idx, row in task_df.iterrows():
            for col_name, val in row.iteritems():
                if val == '?':
                    task_ls.append((row_idx, col_name))
        return task_ls

    def _get_item_base_task_ls(self) -> list:
        task_ls = list()
        task_df = self.df.loc[:, (self.df == '?').any()]
        for row_idx, row in task_df.iterrows():
            for col_name, val in row.iteritems():
                if val == '?':
                    task_ls.append((col_name, row_idx))
        return task_ls

    def _calculate_cosine_similarity(self, target_dict: dict, cp_dict: dict, rounded=True, round_d=3):
        target_user = target_dict['name']
        target_user_ft_ls = target_dict['ft_ls']
        print(f"{target_user} : {target_user_ft_ls}")

        for k, v in cp_dict.items():
            cp_user = k
            cp_user_ft_ls = v['ft_ls']

            cs = self.get_cos_similarity(target_user_ft_ls, cp_user_ft_ls)
            if rounded:
                cs = round(cs, round_d)

            msg = f"cos_sim({target_user}, {cp_user}) = {cs}"
            print(msg)

            cp_dict[cp_user]['cs'] = cs
        print()

    def _get_item_pred_rate_user_base(self, target_item: str, target_user_dict: dict, top_cp_user_ls,
                                      rounded=True, round_d=3) -> float:
        target_user_ft_ls = target_user_dict['ft_ls']
        target_user_ft_mean = self.get_mean(target_user_ft_ls)
        target_user_ft_mean = self.to_decimal(target_user_ft_mean)

        sim_w_product = 0
        print(f"Similarity Weight Product = {sim_w_product} ", end='')
        for cp_user, cp_user_data in top_cp_user_ls:
            cs = cp_user_data['cs']
            cs = self.to_decimal(cs)

            cp_user_ft_ls = cp_user_data['ft_ls']
            cp_user_ft_mean = self.get_mean(cp_user_ft_ls)
            cp_user_ft_mean = self.to_decimal(cp_user_ft_mean)

            cp_user_ti_ft = self.df.loc[cp_user][target_item]
            cp_user_ti_ft = self.to_decimal(cp_user_ti_ft)

            msg = f"+ {cs} * ({cp_user_ti_ft} - {cp_user_ft_mean}) "
            print(msg, end='')

            sp = cs * (cp_user_ti_ft - cp_user_ft_mean)
            sim_w_product += sp

        print(f"= {sim_w_product}")

        sim_w_abs_sum = 0
        print(f"Similarity Weight ABS Sum = {sim_w_abs_sum} ", end='')
        for cp_user, cp_user_data in top_cp_user_ls:
            cs = cp_user_data['cs']
            cs = self.to_decimal(cs)
            cs = abs(cs)
            print(f"+ {cs}", end='')
            sim_w_abs_sum += cs

        print(f"= {sim_w_abs_sum}")

        item_pred_rate = target_user_ft_mean + (sim_w_product / sim_w_abs_sum)
        if rounded:
            item_pred_rate = round(item_pred_rate, round_d)
        msg = f"Predicted Rate for Item = {target_user_ft_mean} + ({sim_w_product} / {sim_w_abs_sum}) = {item_pred_rate}"
        print(msg)
        print()

        return float(item_pred_rate)

    def _get_item_pred_rate_item_base(self, target_user: str, target_item_dict: dict, top_cp_item_ls,
                                      rounded=True, round_d=3) -> float:
        target_item = target_item_dict['name']
        target_item_ft_ls = target_item_dict['ft_ls']
        target_item_ft_mean = self.get_mean(target_item_ft_ls)
        target_item_ft_mean = self.to_decimal(target_item_ft_mean)

        sim_w_product = 0
        print(f"Similarity Weight Product = {sim_w_product} ", end='')

        for cp_item, cp_item_data in top_cp_item_ls:
            cs = cp_item_data['cs']
            cs = self.to_decimal(cs)

            cp_item_ft_ls = cp_item_data['ft_ls']
            cp_item_ft_mean = self.get_mean(cp_item_ft_ls)
            cp_item_ft_mean = self.to_decimal(cp_item_ft_mean)

            cp_item_ti_ft = self.df[cp_item][target_user]
            cp_item_ti_ft = self.to_decimal(cp_item_ti_ft)

            msg = f"+ {cs} * ({cp_item_ti_ft} - {cp_item_ft_mean}) "
            print(msg, end='')

            sp = cs * (cp_item_ti_ft - cp_item_ft_mean)
            sim_w_product += sp
        print(f"= {sim_w_product}")

        sim_w_abs_sum = 0
        print(f"Similarity Weight ABS Sum = {sim_w_abs_sum} ", end='')
        for cp_item, cp_item_data in top_cp_item_ls:
            cs = cp_item_data['cs']
            cs = self.to_decimal(cs)
            cs = abs(cs)
            print(f"+ {cs}", end='')
            sim_w_abs_sum += cs

        print(f"= {sim_w_abs_sum}")

        item_pred_rate = target_item_ft_mean + (sim_w_product / sim_w_abs_sum)
        if rounded:
            item_pred_rate = round(item_pred_rate, round_d)
        msg = f"Predicted Rate for Item = {target_item_ft_mean} + ({sim_w_product} / {sim_w_abs_sum}) = {item_pred_rate}"
        print(msg)
        print()

        return float(item_pred_rate)

    def _print_recommend(self, target_user: str, target_item: str, item_pred_rate: float):
        if item_pred_rate > self.thresh:
            msg = f"{item_pred_rate} > {self.thresh}"
            print(msg)
            msg = f"Should recommend <{target_item}> to {target_user}!"
            print(msg)
        else:
            msg = f"{item_pred_rate} <= {self.thresh}"
            print(msg)
            msg = f"DO NOT recommend <{target_item}> to {target_user}!"
            print(msg)

        print('\n' * 2)

    def _user_base_task_helper(self, task: tuple):
        target_user = task[0]
        target_item = task[1]
        print(f"Target User: {target_user}, Target Item: {target_item}")

        data_df: pd.DataFrame
        data_df = self.df.loc[:, ~(self.df == '?').any()]
        data_df = data_df.astype(float)
        # print(data_df)

        target_row: pd.Series
        target_row = self.df.loc[target_user]
        target_row = target_row[~target_row.isin(['?'])]
        target_user_ft_ls = list(target_row.values)
        target_user_dict = {'name': target_user, 'ft_ls': target_user_ft_ls}
        print(f"* {target_user} : {target_user_ft_ls}")

        cp_user_dict = dict()
        # print(data_df)
        for i, row in data_df.iterrows():
            if i == target_user:
                continue
            cp_user = i
            cp_user_ft_ls = list(row.values)
            print(cp_user, cp_user_ft_ls)
            cp_user_dict[cp_user] = {'ft_ls': cp_user_ft_ls}
        print()

        self._calculate_cosine_similarity(target_user_dict, cp_user_dict)
        top_cp_user_ls = self._get_top_cp_ls(cp_user_dict)
        item_pred_rate = self._get_item_pred_rate_user_base(target_item, target_user_dict, top_cp_user_ls)

        self._print_recommend(target_user, target_item, item_pred_rate)

    def user_base_predict(self):
        task_ls = self._get_user_base_task_ls()
        for task in task_ls:
            self._user_base_task_helper(task)
            # break

    def item_base_predict(self):
        task_ls = self._get_item_base_task_ls()
        for task in task_ls:
            self._item_base_task_helper(task)
            # break

    def _item_base_task_helper(self, task: tuple):
        target_item = task[0]
        target_user = task[1]

        print(f"Target Item: {target_item}, Target User: {target_user}")

        data_df: pd.DataFrame
        data_df = self.df[~self.df.isin(['?']).any(axis=1)]
        drop_col = self.df.loc[:, self.df.isin(['?']).any()].columns
        data_df = data_df.drop(columns=drop_col)
        data_df = data_df.astype(float)
        # print(data_df)

        target_col: pd.Series
        target_col = self.df[target_item]
        target_col = target_col[~target_col.isin(['?'])]
        target_col = target_col.astype(float)
        target_item_ft_ls = list(target_col.values)
        target_item_dict = {'name': target_item, 'ft_ls': target_item_ft_ls}
        print(f"* {target_item}: {target_item_ft_ls}")

        cp_item_dict = dict()
        for col_name, col_content in data_df.iteritems():
            cp_item = col_name
            cp_item_ft_ls = list(col_content.values)
            print(cp_item, cp_item_ft_ls)
            cp_item_dict[cp_item] = {'ft_ls': cp_item_ft_ls}
        print()

        self._calculate_cosine_similarity(target_item_dict, cp_item_dict)
        top_cp_item_ls = self._get_top_cp_ls(cp_item_dict)
        item_pred_rate = self._get_item_pred_rate_item_base(target_user, target_item_dict, top_cp_item_ls)

        self._print_recommend(target_user, target_item, item_pred_rate)


if __name__ == '__main__':
    csv_path = 'cf_q13.csv'
    # df = pd.read_csv(csv_path)
    # print(df)
    cf = CollaboratingFiltering(
        csv_path,
        nature_val=1.5,
        k=1,
        thresh=3.0,
    )
    cf.user_base_predict()
    # cf.item_base_predict()
