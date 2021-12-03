from pprint import pprint as pp
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

tx_dict = {
    '1': {'a', 'b', 'd', 'e'},
    '2': {'b', 'c', 'd'},
    '3': {'a', 'b', 'd', 'e'},
    '4': {'a', 'c', 'd', 'e'},
    '5': {'b', 'c', 'd', 'e'},
    '6': {'b', 'd', 'e'},
    '7': {'c', 'd'},
    '8': {'a', 'b', 'c'},
    '9': {'a', 'd', 'e'},
    '10': {'b', 'd'},
}


def get_item_set(tx_dict: dict) -> list:
    res = set()
    for v in tx_dict.values():
        res.update(v)
    return sorted(res)


def get_empty_labels_dict() -> dict:
    return {item: 0 for item in ITEM_SET}


def tx_dict_to_ohe_df(tx_dict: dict) -> pd.DataFrame:
    encoded_val_ls = list()
    for k, v in tx_dict.items():
        labels = get_empty_labels_dict()
        for i in v:
            if i in labels.keys():
                labels[i] += 1
            else:
                labels[i] = 1
        encoded_val_ls.append(labels)

    return pd.DataFrame(encoded_val_ls)


def print_freq_items(freq_items: pd.DataFrame):
    for i, row in freq_items.iterrows():
        sup = row['support']
        i_set = set(row['itemsets'])

        i_set_str = str(i_set).replace("'", '')
        print(f"{i_set_str} : {sup}")


if __name__ == '__main__':
    ITEM_SET = get_item_set(tx_dict)
    ohe_df = tx_dict_to_ohe_df(tx_dict)
    freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
    print(type(freq_items))

    print_freq_items(freq_items)

    rules = association_rules(freq_items, metric="confidence", min_threshold=0)
    print(rules)
