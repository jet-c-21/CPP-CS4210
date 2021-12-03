from pprint import pprint as pp
import pandas as pd

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


def item_set_dict_to_df(item_set_dict: dict, cols=None):
    if cols is None:
        cols = ['item', 'count']

    return pd.DataFrame(
        list(item_set_dict.items()),
        columns=cols,
    ).sort_values(by=cols[0]).reset_index(drop=True)


def get_item_set_1(d: dict, min_sup=0.3) -> dict:
    min_sup_count = len(d) * min_sup
    res = dict()
    for k, v in d.items():
        for i in v:
            if i in res.keys():
                res[i] += 1
            else:
                res[i] = 1

    # pp(res)

    for k, v in res.items():
        if v < min_sup_count:
            res.pop(k)

    # pp(res)

    return res


def get_item_set_2():
    pass


item_set_1 = get_item_set_1(tx_dict)
item_set_1_df = item_set_dict_to_df(item_set_1)
print(item_set_1_df)
