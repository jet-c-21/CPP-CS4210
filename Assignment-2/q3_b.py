import pandas as pd
import math
from collections import Counter

df = pd.read_csv('binary_points.csv')
df['id'] = df.index + 1
df = df[['id', 'x', 'y', 'class']]


def get_dist(d1: tuple, d2: tuple):
    d1_label = d1[0]
    d1_x = d1[1]
    d1_y = d1[2]

    d2_label = d2[0]
    d2_x = d2[1]
    d2_y = d2[2]

    a = (d1_x - d2_x) ** 2
    b = (d1_y - d2_y) ** 2
    c = a + b
    d = round(math.sqrt(c), 3)

    msg = f"d(#{d1_label}, #{d2_label}) = (({d1_x} - {d2_x})² + ({d1_y} - {d2_y})²)½ = ({a} + {b})½ = ({c})½ = {d}"
    print(msg)

    return d


def knn_handel(dist_d: list, k=3) -> list:
    knn = dist_d[0:k]
    return knn

def print_knn_res(ct_id, trur_label, knn):
    print(f"{len(knn)}NN:")

    cls_ls = []
    for nn in knn:
        cls_ls.append(nn[2])
        msg = f"d(#{ct_id}, #{nn[1]}) = {nn[0]}, nn-class: {nn[2]}"
        print(msg)

    pred_cls = Counter(cls_ls).most_common(1)[0][0]

    if true_label == pred_cls:
        pred_res_str = 'Correct'
        pred_res = True

    else:
        pred_res_str = 'Wrong'
        pred_res = False

    msg = f"\nThe most common class for the prediction is '{pred_cls}', truth label is {true_label}, so the prediction for this data is {pred_res_str}.\n"
    print(msg)

    return pred_res



cw_dict = {
    'Correct': 0,
    'Wrong': 0,
}

for i in range(len(df)):
    counted_row = df.iloc[i]
    rest_rows = df.drop(i)

    ct_id = counted_row['id']
    ct_x = counted_row['x']
    ct_y = counted_row['y']
    true_label = counted_row['class']

    print(f"Data #{ct_id}")

    d_record = []

    for _, row in rest_rows.iterrows():
        cp_id = row['id']
        cp_x = row['x']
        cp_y = row['y']
        predicted_label = row['class']

        d = get_dist((ct_id, ct_x, ct_y), (cp_id, cp_x, cp_y))
        # break
        d_record.append((d, cp_id, predicted_label))

    d_record.sort(key=lambda x: x[0])
    knn = knn_handel(d_record)
    pred_res = print_knn_res(ct_id, true_label, knn)

    if pred_res:
        cw_dict['Correct'] += 1
    else:
        cw_dict['Wrong'] += 1

print()
print(cw_dict)
