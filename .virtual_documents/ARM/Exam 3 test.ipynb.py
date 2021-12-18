from association_rule_mining import AssociationRuleMining


tx_dict = {
    '1' : ['A', 'D'],
    '2' : ['A', 'B', 'C'],
    '3' : ['A', 'B', 'D'],
    '4' : ['A', 'C'],
}


arm = AssociationRuleMining(tx_dict, 0.5, 0.5)


arm.freq_itemset


arm.view_labels_of_all_itemset()


arm.all_itemset


tx_dict = {
    '1': ['a', 'c', 'd'],
    '2': ['b', 'c'],
    '3': ['a', 'b', 'd'],
    '4': ['a', 'b'],
    '5': ['a', 'b', 'c', 'd'],
}


arm = AssociationRuleMining(tx_dict, 0.6, 0.7)


arm.view_all_freq_itemset()


arm.view_rules_of_k_itemset(2)


arm.view_freq_k_itemset_fkm1_x_fkm1(3)






