from association_rule_mining import AssociationRuleMining


tx_dict = {
    '1' : ['b', 'd'],
    '2' : ['b', 'c', 'd'],
    '3' : ['a', 'b', 'd'],
    '4' : ['a', 'b'],
    '5' : ['a', 'b', 'c', 'd'],
}


arm = AssociationRuleMining(tx_dict, 0.6, 0.7)


arm.view_all_freq_itemset()


arm.view_rules_of_k_itemset(2)


arm.view_labels_of_all_itemset()



