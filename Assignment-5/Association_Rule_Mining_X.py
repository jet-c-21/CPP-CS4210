from pprint import pprint as pp
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from itertools import combinations
from copy import deepcopy


class AssociationRuleMining:
    def __init__(self, raw_data: dict, min_sup: float, min_conf: float):
        self.raw_data = raw_data
        self.min_sup = min_sup
        self.min_conf = min_conf

        self.item_set_1 = self._get_item_set()
        self.ohe_df = self._get_ohe_df()

        self.freq_item_set = apriori(self.ohe_df,
                                     min_support=self.min_sup,
                                     use_colnames=True,
                                     verbose=1)
        self.freq_item_set: pd.DataFrame

        self.all_freq_item_set = apriori(self.ohe_df,
                                         min_support=0.01,
                                         use_colnames=True,
                                         verbose=1)

        self.rules = association_rules(self.freq_item_set,
                                       metric="confidence",
                                       min_threshold=self.min_conf)

    @staticmethod
    def get_item_set_str(s) -> str:
        t = str(sorted(s)).replace("'", '').replace('[', '{').replace(']', '}')
        return t

    def _get_item_set(self) -> list:
        res = set()
        for v in self.raw_data.values():
            res.update(v)
        return sorted(res)

    def _get_empty_labels_dict(self) -> dict:
        return {item: 0 for item in self.item_set_1}

    def _get_ohe_df(self) -> pd.DataFrame:
        encoded_val_ls = list()
        for k, v in self.raw_data.items():
            labels = self._get_empty_labels_dict()
            for i in v:
                if i in labels.keys():
                    labels[i] += 1
                else:
                    labels[i] = 1
            encoded_val_ls.append(labels)

        return pd.DataFrame(encoded_val_ls)

    def get_freq_item_set(self, k) -> list:
        result = list()
        for i, row in self.freq_item_set.iterrows():
            item_set = row['itemsets']
            # print(item_set, type(item_set))
            if len(item_set) == k:
                item_set = sorted(item_set)
                # item_set = set(item_set)
                result.append(item_set)
        return result

    def print_freq_item_set(self):
        for i, row in self.freq_item_set.iterrows():
            sup = row['support']
            i_set = set(row['itemsets'])

            i_set_str = str(i_set).replace("'", '')
            print(f"{i_set_str} : {sup}")

    def view_freq_k_item_set(self, k: int):
        self.rules: pd.DataFrame
        for i, row in self.rules.iterrows():
            acd = set(row['antecedents'])
            csq = set(row['consequents'])
            if len(acd) + len(csq) == k:
                sup = round(row['support'], 2)
                conf = round(row['confidence'], 2)
                acd_str = self.get_item_set_str(acd)
                csq_str = self.get_item_set_str(csq)

                msg = f"{acd_str} -> {csq_str} : min-support = {sup} min-confidence = {conf}"
                print(msg)

    def _print_fkm1_fkm1(self, freq_item_set_ls):
        for i, s in enumerate(freq_item_set_ls):
            s = set(s)
            s_str = self.get_item_set_str(s)
            if i == 0:
                print(s_str, end='')
            else:
                print(f", {s_str}", end='')

        print('\n X')

        for i, s in enumerate(freq_item_set_ls):
            s = set(s)
            s_str = self.get_item_set_str(s)
            if i == 0:
                print(s_str, end='')
            else:
                print(f", {s_str}", end='')
        print()

    def _get_item_set_comb(self, freq_item_set_ls, n=2):
        set_comb = list(combinations(freq_item_set_ls, 2))
        return set_comb

    def _print_candidate(self, candidate: list):
        for i, s in enumerate(candidate):
            s = set(s)
            s_str = self.get_item_set_str(s)
            if i == 0:
                print(f"Candidates: {s_str}", end='')
            else:
                print(f", {s_str}", end='')

    def view_freq_k_item_set_fkm1_fkm1(self, k: int):
        freq_item_set_ls = self.get_freq_item_set(k - 1)
        self._print_fkm1_fkm1(freq_item_set_ls)

        set_comb = self._get_item_set_comb(freq_item_set_ls)

        candidate = list()
        for c in set_comb:
            for el in c[1]:
                temp = list(c[0])
                temp.append(el)
                temp = set(temp)
                if len(temp) != k:
                    continue
                temp = sorted(temp)
                if temp not in candidate:
                    candidate.append(temp)

        self._print_candidate(candidate)

        for i in candidate:
            print(i)

    def mine(self):
        pass


if __name__ == '__main__':
    tx_dict = {
        # '1': {'a', 'b', 'd', 'e'},
        # '2': {'b', 'c', 'd'},
        # '3': {'a', 'b', 'd', 'e'},
        # '4': {'a', 'c', 'd', 'e'},
        # '5': {'b', 'c', 'd', 'e'},
        # '6': {'b', 'd', 'e'},
        # '7': {'c', 'd'},
        # '8': {'a', 'b', 'c'},
        # '9': {'a', 'd', 'e'},
        # '10': {'b', 'd'},

        # '1': {'A', 'D'},
        # '2': {'A', 'B', 'C'},
        # '3': {'A', 'B', 'D'},
        # '4': {'A', 'C'},

        '1': {'a', 'c', 'd'},
        '2': {'b', 'c'},
        '3': {'a', 'b', 'd'},
        '4': {'a', 'b', 'c', 'd'},
    }

    arm = AssociationRuleMining(tx_dict, 0.01, 0.01)
    arm.mine()
    # arm.print_freq_item_set()
    # arm.view_freq_k_item_set(2)
    arm.view_freq_k_item_set_fkm1_fkm1(3)

    # print(arm.all_freq_item_set)
