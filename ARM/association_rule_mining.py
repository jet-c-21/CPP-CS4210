# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/12/11
"""
from unique_list import UniqueList
from mlxtend.frequent_patterns import apriori, association_rules
from collections import OrderedDict
from itertools import combinations
import pandas as pd

from pprint import pprint as pp


class AssociationRuleMining:
    def __init__(self, raw_data: dict, min_sup: float, min_conf: float, load_all_itemset=True):
        self.raw_data = raw_data
        self.min_sup = min_sup
        self.min_conf = min_conf

        self.itemset_1 = self._get_itemset_1()
        self.ohe_df = self._get_ohe_df()
        self.freq_itemset = apriori(self.ohe_df,
                                    min_support=self.min_sup,
                                    use_colnames=True,
                                    verbose=1)
        self.rules = association_rules(self.freq_itemset,
                                       metric='confidence',
                                       min_threshold=self.min_conf)

        self.all_itemset = None
        if load_all_itemset:
            self.all_itemset = self.get_all_itemset_bf()

    @staticmethod
    def get_itemset_str(itemset: UniqueList) -> str:
        res = str(itemset).replace("'", '').replace('<', '{').replace('>', '}')
        return res

    def _get_itemset_1(self) -> UniqueList:
        res = UniqueList()
        for v in self.raw_data.values():
            res.update_els(v)

        res.sort()
        return res

    def _get_empty_labels_dict(self) -> dict:
        return {item: 0 for item in self.itemset_1}

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

    def view_all_freq_itemset(self):
        msg = 'All frequent itemsets:'
        print(msg)
        for i, row in self.freq_itemset.iterrows():
            sup = row['support']
            itemset = UniqueList()
            itemset.update_els(row['itemsets'])
            itemset.sort()
            itemset_str = self.get_itemset_str(itemset)
            msg = f"{itemset_str}: support = {sup}"
            print(msg)

    def get_freq_k_itemset_ls(self, k: int) -> list:
        result = list()
        self.freq_itemset: pd.DataFrame
        for i, row in self.freq_itemset.iterrows():
            its = row['itemsets']
            if len(its) == k:
                itemset = UniqueList()
                itemset.update_els(its)
                itemset.sort()
                result.append(itemset)

        return result

    def view_freq_k_itemset(self, k: int):
        msg = f"{k}-itemsets:"
        print(msg)
        self.freq_itemset: pd.DataFrame
        for i, row in self.freq_itemset.iterrows():
            its = row['itemsets']
            if len(its) == k:
                itemset = UniqueList()
                itemset.update_els(its)
                itemset.sort()

                itemset_str = self.get_itemset_str(itemset)

                sup = round(row['support'], 2)

                msg = f"{itemset_str} : support = {sup}"
                print(msg)

    def _print_fk_itemset_ls(self, fk_itemset_ls: list):
        k = len(fk_itemset_ls[0])
        msg = f"Frequent {k}-itemsets:"
        print(msg)
        for itemset in fk_itemset_ls:
            itemset_str = self.get_itemset_str(itemset)
            print(itemset_str)

    def view_freq_k_itemset_fkm1_x_f1(self, k: int):
        msg = f"{k}-itemsets:"
        # print(msg)
        if k < 3:
            self.view_freq_k_itemset(k)

        fkm1_itemset_ls = self.get_freq_k_itemset_ls(k - 1)
        self._print_fk_itemset_ls(fkm1_itemset_ls)
        print()

        f1_itemset_ls = self.get_freq_k_itemset_ls(1)
        self._print_fk_itemset_ls(f1_itemset_ls)
        print()

        # get candidate
        msg = 'Generated Candidate:'
        print(msg)
        candidate = UniqueList()
        for itemset in fkm1_itemset_ls:
            for merged_item in f1_itemset_ls:
                if not set(merged_item) <= set(itemset):
                    temp = UniqueList()
                    temp.update_els(itemset)
                    temp.update_els(merged_item)
                    temp.sort()
                    itemset_str = self.get_itemset_str(temp)
                    print(itemset_str)
                    candidate.add_el(temp)
        print()

        # candidate pruning
        cand_s_flag = False
        msg = 'After candidate pruning:'
        print(msg)
        for cand in candidate:
            sup = self.get_support(cand)
            # print(sup)
            if sup >= self.min_sup:
                itemset_str = self.get_itemset_str(cand)
                print(itemset_str)
                cand_s_flag = True

        if not cand_s_flag:
            msg = f"None of the candidates is survived after pruning."
            print(msg)

    def _check_fkm1_x_fkm1_cand(self, cand: UniqueList, k) -> bool:
        comb = combinations(cand, k - 1)
        for isc in list(comb):
            itemset = UniqueList()
            itemset.update_els(isc)
            sup = self.get_support(itemset)
            if sup < self.min_sup:
                return False
        return True

    def view_freq_k_itemset_fkm1_x_fkm1(self, k: int):
        if k < 3:
            self.view_freq_k_itemset(k)

        fkm1_itemset_ls = self.get_freq_k_itemset_ls(k - 1)
        fkm1_itemsets_str = str(fkm1_itemset_ls).replace('[', '').replace(']', '') \
            .replace('<', '{').replace('>', '}').replace('\'', '')

        msg = f"{fkm1_itemsets_str}\nX\n{fkm1_itemsets_str}\n"
        print(msg)

        msg = 'Generated Candidate:'
        print(msg)
        cand_ls = list()
        comb = combinations(fkm1_itemset_ls, k - 1)
        for isc in list(comb):
            itemset_a, itemset_b = isc[0], isc[1]
            if itemset_a[:k - 2] == itemset_b[:k - 2]:
                cand = UniqueList()
                prefix_item = itemset_a[:k - 2]
                cand.update_els(prefix_item)
                merged_item_a = itemset_a[k - 2:]
                cand.update_els(merged_item_a)
                merged_item_b = itemset_b[k - 2:]
                cand.update_els(merged_item_b)
                cand.sort()

                cand_str = self.get_itemset_str(cand)
                print(cand_str)

                cand_ls.append(cand)
        print()

        cand_s_flag = False
        msg = 'After candidate pruning:'
        print(msg)
        for cand in cand_ls:
            if self._check_fkm1_x_fkm1_cand(cand, k):
                cand_str = self.get_itemset_str(cand)
                print(cand_str)
                cand_s_flag = True

        if not cand_s_flag:
            msg = f"None of the candidates is survived after pruning."
            print(msg)

    def view_rules_of_k_itemset(self, k: int):
        msg = f"Rules of {k}-itemsets:"
        print(msg)
        for i, row in self.rules.iterrows():
            acd = UniqueList()
            acd.update_els(row['antecedents'])
            acd.sort()
            csq = UniqueList()
            csq.update_els(row['consequents'])
            csq.sort()
            if len(acd) + len(csq) == k:
                sup = round(row['support'], 2)
                conf = round(row['confidence'], 2)
                acd_str = self.get_itemset_str(acd)
                csq_str = self.get_itemset_str(csq)

                msg = f"{acd_str} -> {csq_str} : support = {sup}, confidence = {conf}"
                print(msg)

    def get_support(self, itemset: UniqueList) -> float:
        sup_count = 0
        for i, row in self.ohe_df.iterrows():
            tx_labels = row[row.isin([1])].index.tolist()
            if set(itemset) <= set(tx_labels):
                sup_count += 1

        return sup_count / len(self.ohe_df)

    def get_all_itemset_bf(self) -> list:
        result = list()
        for k in range(1, len(self.itemset_1) + 1):
            comb = combinations(self.itemset_1, k)
            for isc in list(comb):
                itemset = UniqueList()
                itemset.update_els(isc)
                itemset.sort()

                sup = self.get_support(itemset)
                # print(f"s = {sup} : {itemset}")
                result.append((sup, itemset))

        return result

    def view_all_itemset(self):
        msg = 'All itemsets:'
        print(msg)
        for sup, itemset in self.all_itemset:
            itemset_str = self.get_itemset_str(itemset)
            print(itemset_str)

    def get_immediate_supersets(self, itemset: UniqueList) -> list:
        result = list()
        for _, its in self.all_itemset:
            if len(its) == len(itemset) + 1 and set(itemset) <= set(its):
                result.append(its)
        return result

    def is_maximal(self, itemset: UniqueList) -> bool:
        """
        A frequent itemset is maximal
        if NONE of its immediate supersets is frequent
        :param itemset:
        :return: bool
        """
        # print(f"itemset: {itemset}")
        imd_supersets = self.get_immediate_supersets(itemset)
        for its in imd_supersets:
            sup = self.get_support(its)
            # print(f"imd: {its} - {sup}")
            if sup >= self.min_sup:
                return False
        return True

    def is_closed(self, itemset: UniqueList) -> bool:
        """
        A frequent itemset is closed
        if none of its immediate supersets
        has the same support as itself
        :param itemset:
        :return: bool
        """
        base_sup = self.get_support(itemset)
        imd_supersets = self.get_immediate_supersets(itemset)
        for its in imd_supersets:
            sup = self.get_support(its)
            if sup == base_sup:
                return False
        return True

    def get_itemset_labels(self, itemset: UniqueList, support: float) -> list:
        result = list()
        if support < self.min_sup:  # Infrequent
            result.append('I')
            return result

        if self.is_maximal(itemset):
            result.append('M')

        if self.is_closed(itemset):
            result.append('C')

        if 'M' not in result and 'C' not in result:
            result.append('F')

        return result

    def view_labels_of_all_itemset(self):
        """
        M - maximal frequent itemset:
                A frequent itemset is maximal
                if NONE of its immediate supersets is frequent

        C - closed frequent itemset:
                A frequent itemset is closed
                if none of its immediate supersets
                has the same support as itself

        F - frequent but neither maximal nor closed

        I - infrequent

        :return:
        """
        m_its_ls = list()
        c_its_ls = list()
        f_its_ls = list()
        i_its_ls = list()

        for sup, itemset in self.all_itemset:
            labels = self.get_itemset_labels(itemset, sup)
            msg = f"{itemset} : {labels}"
            print(msg)

            if 'I' in labels:
                i_its_ls.append(itemset)
            else:
                if 'M' in labels:
                    m_its_ls.append(itemset)

                if 'C' in labels:
                    c_its_ls.append(itemset)

                if 'F' in labels:
                    f_its_ls.append(itemset)
        print()

        msg = f"Maximal Frequent Itemset:"
        print(msg)
        for itemset in m_its_ls:
            itemset_str = self.get_itemset_str(itemset)
            print(itemset_str)
        print()

        msg = f"Closed Frequent Itemset:"
        print(msg)
        for itemset in c_its_ls:
            itemset_str = self.get_itemset_str(itemset)
            print(itemset_str)
        print()

        msg = f"Frequent Itemset but Neither Maximal nor Closed:"
        print(msg)
        for itemset in f_its_ls:
            itemset_str = self.get_itemset_str(itemset)
            print(itemset_str)
        print()

        msg = f"Infrequent Itemset:"
        print(msg)
        for itemset in i_its_ls:
            itemset_str = self.get_itemset_str(itemset)
            print(itemset_str)
        print()


if __name__ == '__main__':
    tx_dict_1 = {
        '1': ['Bread', 'Milk'],
        '2': ['Beer', 'Bread', 'Diaper', 'Eggs'],
        '3': ['Beer', 'Coke', 'Diaper', 'Milk'],
        '4': ['Beer', 'Bread', 'Diaper', 'Milk'],
        '5': ['Bread', 'Coke', 'Diaper', 'Milk'],
    }

    tx_dict_2 = {
        '1': ['a', 'b', 'd', 'e'],
        '2': ['b', 'c', 'd'],
        '3': ['a', 'b', 'd', 'e'],
        '4': ['a', 'c', 'd', 'e'],
        '5': ['b', 'c', 'd', 'e'],
        '6': ['b', 'd', 'e'],
        '7': ['c', 'd'],
        '8': ['a', 'b', 'c'],
        '9': ['a', 'd', 'e'],
        '10': ['b', 'd'],
    }

    tx_dict_3 = {
        '1': ['A', 'D'],
        '2': ['A', 'B', 'C'],
        '3': ['A', 'B', 'D'],
        '4': ['A', 'C'],
    }

    # tx_dict = tx_dict_1
    # tx_dict = tx_dict_2
    tx_dict = tx_dict_3

    tx_dict = OrderedDict(tx_dict)
    arm = AssociationRuleMining(tx_dict, 0.5, 0.5)
    # print(arm.freq_itemset)
    # pp(arm.raw_data)
    # print(arm.itemset_1)
    # arm.view_all_freq_itemset()
    # arm.view_rules_of_k_itemset(3)
    # pp(arm.all_itemset)
    # arm.view_all_itemset()
    # arm.view_freq_k_itemset(3)
    # arm.view_freq_k_itemset_fkm1_x_f1(3)
    # arm.view_freq_k_itemset_fkm1_x_fkm1(3)
    arm.view_labels_of_all_itemset()
