import pandas as pd
import math
from collections import Counter

df = pd.read_csv('binary_points.csv')
df['id'] = df.index + 1
df = df[['id', 'x', 'y', 'class']]


cw_dict = {
    'Correct': 0,
    'Wrong': 0,
}

for i in range(len(df)):
    counted_row = df.iloc[i]
    rest_rows = df.drop(i)

    ct_id = counted_row['id']
    true_label = counted_row['class']

    print(f"Data #{ct_id}:")

    pred_cls_ls = []

    for _, row in rest_rows.iterrows():
        cp_id = row['id']
        predicted_label = row['class']
        pred_cls_ls.append(predicted_label)

        print(f"#{cp_id} : '{predicted_label}'")

    pred_cls_ct = Counter(pred_cls_ls)
    pred_cls_dict = dict(pred_cls_ct)

    msg = f"We got {pred_cls_dict['+']} of '+' and {pred_cls_dict['-']} of '-'."
    print(msg)

    pred_cls = pred_cls_ct.most_common(1)[0][0]

    # print(true_label, pred_cls)

    if true_label == pred_cls:
        pred_res = 'Correct'
        cw_dict['Correct'] += 1
    else:
        pred_res = 'Wrong'
        cw_dict['Wrong'] += 1

    msg = f"Hence, the most common class for the prediction is '{pred_cls}', the truth label is '{true_label}'. Thus, the prediction is {pred_res}.\n"
    print(msg)


print()
print(cw_dict)
