import pandas as pd
import math
from collections import Counter

df = pd.read_csv('rgb.csv')
# print(df)

ct_id = 10
ct_red = 154
ct_green = 205
ct_blue = 50


def get_dist(d1: tuple, d2: tuple):
    d1_label = d1[0]
    d1_r = d1[1]
    d1_g = d1[2]
    d1_b = d1[3]

    d2_label = d2[0]
    d2_r = d2[1]
    d2_g = d2[2]
    d2_b = d2[3]

    r_diff = (d1_r - d2_r) ** 2
    g_diff = (d1_g - d2_g) ** 2
    b_diff = (d1_b - d2_b) ** 2

    rgb_diff = r_diff + g_diff + b_diff
    d = round(math.sqrt(rgb_diff), 3)

    msg = f"d(#{d1_label}, #{d2_label}) = (({d1_r} - {d2_r})² + ({d1_g} - {d2_g})² + ({d1_b} - {d2_b})²)½ = ({r_diff} + {g_diff} + {b_diff})½ = ({rgb_diff})½ = {d}"
    print(msg)

    return d


def knn_handel(dist_d: list, k=3) -> list:
    knn = dist_d[0:k]
    return knn


def print_knn_res(ct_id, knn):
    print(f"{len(knn)}NN:")

    cls_ls = []
    for nn in knn:
        cls_ls.append(nn[2])
        msg = f"d(#{ct_id}, #{nn[1]}) = {nn[0]}, nn-class: {nn[2]}"
        print(msg)

    pred_cls = Counter(cls_ls).most_common(1)[0][0]

    msg = f"\nThe most common class for the prediction is '{pred_cls}'.\n"
    print(msg)
    #
    # return pred_res


d_record = list()
for i, row in df.iterrows():
    cp_id = row['ID']
    cp_red = row['Red']
    cp_green = row['Green']
    cp_blue = row['Blue']
    predicted_label = row['Class']

    d = get_dist((ct_id, ct_red, ct_green, ct_blue), (cp_id, cp_red, cp_green, cp_blue))
    d_record.append((d, cp_id, predicted_label))

d_record.sort(key=lambda x: x[0])
knn = knn_handel(d_record)
pred_res = print_knn_res(ct_id, knn)
