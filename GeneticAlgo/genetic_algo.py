# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/12/12
"""
from typing import Union
import pandas as pd
from genetic_repr_rule import GeneticReprRule
from chromosome import Chromosome
from pprint import pprint as pp
from decimal import Decimal as D


class GeneticAlgo:
    def __init__(self, raw_data: Union[str, pd.DataFrame], initial_pop_ls: list,
                 col_repr_rule_ls: list, crossover_rule_dict: dict, mutation_rule_dict: dict,
                 cls_col=None, select_best_count=2, ):
        self.raw_data = raw_data
        self.initial_pop_ls = initial_pop_ls
        self.col_repr_rule_ls = col_repr_rule_ls
        self.crossover_rule_dict = crossover_rule_dict
        self.mutation_rule_dict = mutation_rule_dict

        if isinstance(raw_data, str):
            self.df = pd.read_csv(raw_data)
        else:
            self.df = raw_data

        if cls_col is None:
            self.cls_col = self.df.columns[-1]
        else:
            self.cls_col = cls_col

        self.select_best_count = select_best_count

        self.col_repr_rule_dict = dict()
        self.f_col_repr_rule_ls = list()
        self.cls_col_repr_rule = None
        self.cls_col_repr_rule: GeneticReprRule
        self._gen_col_repr_rules()

        self.chrsm_ls = list()
        self.act_chrsm_ls = list()
        self._gen_initial_chromosome()
        self.chrsm_latest_idx = len(self.act_chrsm_ls) - 1

        # self.g_row_ls = list()
        # self._parse_df_to_g_rows()

    def _gen_col_repr_rules(self):
        for r in self.col_repr_rule_ls:
            col_name = r['col']
            # print(col_name)
            temp = dict()
            temp['bin'] = r['bin']
            if temp['bin']:
                temp['rule_str'] = None
                temp['rule'] = GeneticReprRule(col_name, is_bin=True, bin_dict=r['val'])
                self.col_repr_rule_dict[col_name] = temp
                self.cls_col_repr_rule = temp['rule']
            else:
                temp['rule_str'] = r['val']
                temp['rule'] = GeneticReprRule(col_name, seq_str=temp['rule_str'])
                self.col_repr_rule_dict[col_name] = temp
                self.f_col_repr_rule_ls.append(temp['rule'])

    # # unused
    # def _get_col_repr_rule(self, col_name) -> dict:
    #     for r in self.raw_col_repr_rule_ls:
    #         if col_name == r['col']:
    #             return r

    # # unused
    # def _parse_df_to_g_rows(self):
    #     for i, row in self.df.iterrows():
    #         # parse row to g_row
    #         g_row = ''
    #         for col_name, val in row.iteritems():
    #             col_repr_rule: GeneticReprRule
    #             col_repr_rule = self.col_repr_rule_dict[col_name]['rule']
    #             g_bit_str = col_repr_rule.get_g_bit_str(val)
    #             g_row += g_bit_str
    #             # print(g_row)
    #         # print(g_row)

    def _gen_initial_chromosome(self):
        for chrsm_id, chrsm_bit_str in enumerate(self.initial_pop_ls):
            chrsm = Chromosome(chrsm_id, chrsm_bit_str,
                               self.f_col_repr_rule_ls,
                               self.cls_col_repr_rule)
            self.chrsm_ls.append(chrsm)
            self.act_chrsm_ls.append(chrsm)

    def _print_generation_header(self, gen_idx: int):
        msg = f"#{gen_idx} generation "
        print(msg, end='')

        act_chrsm_str = '('
        for chrsm in self.act_chrsm_ls:
            act_chrsm_str += chrsm.name + ','
        act_chrsm_str = act_chrsm_str[:len(act_chrsm_str) - 1] + '):'
        print(act_chrsm_str)

    def _print_fitness_of_act_chrsms(self):
        for chrsm in self.act_chrsm_ls:
            chrsm: Chromosome
            msg = f"Fitness({chrsm.name}) = {chrsm.fitness}"
            print(msg)
        print()

    def _print_pr_of_act_chrsms(self):
        for i, chrsm in enumerate(self.act_chrsm_ls, start=1):
            chrsm: Chromosome
            msg = f"Pr({chrsm.name}) = {chrsm.probability} (#{i})"
            print(msg)
        print()

    def _get_fitness_of_act_chrsms(self):
        for chrsm in self.act_chrsm_ls:
            chrsm: Chromosome
            fitness = chrsm.get_fitness(self.df)
            chrsm.set_fitness(fitness)
            # print(f"{chrsm.name}: {chrsm.fitness}")
            # break

    def _get_pr_of_act_chrsms(self):
        fit_sum = 0
        chrsm: Chromosome
        for chrsm in self.act_chrsm_ls:
            fit_sum += chrsm.fitness

        for chrsm in self.act_chrsm_ls:
            fit = chrsm.fitness
            pr = D(fit) / D(fit_sum)
            pr = round(pr, 2)
            pr = float(pr)
            chrsm.set_probability(pr)
            # print(f"{chrsm.name}: {chrsm.probability}")

    def _sort_act_chrsm_ls_by_pr(self):
        self.act_chrsm_ls.sort(key=lambda x: x.probability, reverse=True)

    def _sort_act_chrsm_ls_by_id(self):
        self.act_chrsm_ls.sort(key=lambda x: x.chrsm_id)

    @staticmethod
    def _print_crossover_process(ca: Chromosome, ca_child: Chromosome,
                                 cb: Chromosome, cb_child: Chromosome, mask: str):
        msg = f"{ca.name} = "
        print(msg, end='')
        for i, (m_s, c_s) in enumerate(zip(mask, ca.chrsm_bit_str)):
            print(c_s, end='')
            if i < len(mask) - 1 and m_s != mask[i + 1]:
                print('|', end='')
        msg = f" → {ca_child.name} = {ca_child.chrsm_bit_str}"
        print(msg)

        msg = f"{cb.name} = "
        print(msg, end='')
        for i, (m_s, c_s) in enumerate(zip(mask, cb.chrsm_bit_str)):
            print(c_s, end='')
            if i < len(mask) - 1 and m_s != mask[i + 1]:
                print('|', end='')
        msg = f" → {cb_child.name} = {cb_child.chrsm_bit_str}"
        print(msg, '\n')

    def _apply_crossover(self, gen_idx: int):
        csr = self.crossover_rule_dict[gen_idx]
        chrsm_a_idx = csr['idx'][0] - 1
        chrsm_b_idx = csr['idx'][1] - 1
        mask = csr['mask']

        chrsm_a: Chromosome
        chrsm_b: Chromosome
        chrsm_a, chrsm_b = self.act_chrsm_ls[chrsm_a_idx], self.act_chrsm_ls[chrsm_b_idx]

        if chrsm_a.chrsm_id > chrsm_b.chrsm_id:
            chrsm_a, chrsm_b = chrsm_b, chrsm_a

        chrsm_a_child_str, chrsm_b_child_str = '', ''
        for i, s in enumerate(mask):
            if s == '1':
                chrsm_a_child_str += chrsm_a.chrsm_bit_str[i]
                chrsm_b_child_str += chrsm_b.chrsm_bit_str[i]
            else:
                chrsm_a_child_str += chrsm_b.chrsm_bit_str[i]
                chrsm_b_child_str += chrsm_a.chrsm_bit_str[i]

        # print(self.chrsm_latest_idx)
        self.chrsm_latest_idx += 1
        chrsm_a_child = Chromosome(self.chrsm_latest_idx, chrsm_a_child_str,
                                   self.f_col_repr_rule_ls,
                                   self.cls_col_repr_rule)
        self.chrsm_ls.append(chrsm_a_child)

        self.chrsm_latest_idx += 1
        chrsm_b_child = Chromosome(self.chrsm_latest_idx, chrsm_b_child_str,
                                   self.f_col_repr_rule_ls,
                                   self.cls_col_repr_rule)
        self.chrsm_ls.append(chrsm_b_child)

        self._print_crossover_process(chrsm_a, chrsm_a_child, chrsm_b, chrsm_b_child, mask)

        self.act_chrsm_ls = self.act_chrsm_ls[:self.select_best_count]
        self.act_chrsm_ls.append(chrsm_a_child)
        self.act_chrsm_ls.append(chrsm_b_child)
        self._sort_act_chrsm_ls_by_id()
        # print(self.act_chrsm_ls)

    def _apply_mutation(self, gen_idx):
        if gen_idx not in self.mutation_rule_dict.keys():
            return

        m_rule = self.mutation_rule_dict[gen_idx]
        t_cbs = m_rule['chrsm_bit_str']
        t_idx = m_rule['idx'] - 1

        msg = f"Applying mutation on {t_cbs}:"
        print(msg)

        for chrsm in self.act_chrsm_ls:
            chrsm: Chromosome
            if chrsm.chrsm_bit_str == t_cbs:
                print(chrsm, end='')
                chrsm.mutate(t_idx)
                print(f" → {chrsm}")
        print()

    def _has_acc_100_chrsm(self) -> bool:
        for chrsm in self.act_chrsm_ls:
            chrsm: Chromosome
            if chrsm.fitness == 1:
                return True

        return False

    def _get_best_chrsm(self) -> list:
        result = list()
        for chrsm in self.act_chrsm_ls:
            chrsm: Chromosome
            if chrsm.fitness == 1:
                result.append(chrsm)
        return result

    def _print_best_chrsm(self):
        msg = f"The Best Chromosome:"
        print(msg)
        for chrsm in self._get_best_chrsm():
            print(chrsm)

    def start_generation(self):
        gen_idx = 0
        flag = True
        while flag:
            self._get_fitness_of_act_chrsms()
            self._print_fitness_of_act_chrsms()

            gen_idx += 1

            if self._has_acc_100_chrsm():
                self._print_best_chrsm()
                break

            # if gen_idx == 3:
            #     flag = False
            #     break

            self._print_generation_header(gen_idx)

            self._get_pr_of_act_chrsms()
            self._sort_act_chrsm_ls_by_pr()
            self._print_pr_of_act_chrsms()
            self._apply_crossover(gen_idx)
            self._apply_mutation(gen_idx)
            # break


if __name__ == '__main__':
    data_path_1 = 'hw4.csv'

    i_pop_1 = [
        '1001001',
        '0100101',
        '1011000',
        '1101100',
    ]

    col_repr_r_1 = [
        {'col': 'Outlook', 'val': '<Sunny, Overcast, Rain>', 'bin': False},
        {'col': 'Temperature', 'val': '<Hot, Mild, Cool>', 'bin': False},
        {'col': 'PlayTennis', 'val': {'Yes': '1', 'No': '0'}, 'bin': True},
    ]

    cso_r_1 = {
        1: {'idx': (1, 3), 'mask': '1110000'},
        2: {'idx': (1, 2), 'mask': '0001100'},
    }

    m_r_1 = {
        2: {'chrsm_bit_str': '1011000', 'idx': 6}
    }

    data_path = data_path_1
    i_pop = i_pop_1
    col_repr_r = col_repr_r_1
    cso_r = cso_r_1
    m_r = m_r_1

    ga = GeneticAlgo(data_path, i_pop, col_repr_r, cso_r, m_r)
    ga.start_generation()
    # ga.chrsm_ls[0].fitness = 0.69
    # print(ga.chrsm_ls[0].fitness)
    # print(ga.act_chrsm_ls[0].fitness)

    # pp(ga.col_repr_rule_dict)
    # pp(ga.f_col_repr_rule_ls)
    # pp(ga.chrsm_ls)
    # pp(ga.col_repr_rule_dict)
    # print(ga.df.)

    # for i, row in ga.df.iterrows():
    #     print(row.to_dict())
