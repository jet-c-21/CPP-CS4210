import math


def get_entropy_by_pn(p, n):
    total = p + n
    p_p = p / total
    n_p = n / total

    # positive
    if p_p != 0:
        positive = p_p * math.log(p_p, 2)
    else:
        positive = 0

    # negative
    if n_p != 0:
        negative = n_p * math.log(n_p, 2)
    else:
        negative = 0

    return - positive - negative


def get_entropy_by_pn_rd(p, n, r=3):
    return round(get_entropy_by_pn(p, n), r)


entropy = get_entropy_by_pn_rd(1, 2)
print(entropy)
