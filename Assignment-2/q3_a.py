import pandas as pd
import math

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
    res = d_record.pop(0)

    nearest_d = res[1]
    pred_cls = res[2]

    if true_label == pred_cls:
        pred_res = 'Correct'
        cw_dict['Correct'] += 1
    else:
        pred_res = 'Wrong'
        cw_dict['Wrong'] += 1

    msg = f"\nThe nearest data is #{nearest_d}, the predicted class is {pred_cls}, truth label is {true_label}, so the prediction for this data is {pred_res}.\n"
    print(msg)


print()
print(cw_dict)
