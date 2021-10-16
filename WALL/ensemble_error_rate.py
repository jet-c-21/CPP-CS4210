import math
from scipy.stats import binom


def ensemble_error_rate(clf_count: int, wrong_clf_count: int, error_rate: float):
    res = 0
    for i in range(wrong_clf_count, clf_count + 1):
        p = error_rate
        q = 1 - p
        c = math.factorial(clf_count) / (math.factorial(i) * math.factorial(clf_count - i))
        e = c * (p ** i) * (q ** (clf_count - i))
        res += e

    return res


eer = ensemble_error_rate(25, 13, 0.35)
print(eer)

x = binom.pmf(13, 25, (1 - 0.35))
print(x)
