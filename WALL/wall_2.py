s = {5, 2, 7, 1, 8}


def sort_set(s) -> set:
    return sorted([s])[0]


def get_item_set_str(s) -> str:
    t = str(sorted(s)).replace("'", '').replace('[', '{').replace(']', '}')
    return t


s = sort_set(s)
print(s)
