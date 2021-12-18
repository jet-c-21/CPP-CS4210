# coding: utf-8
"""
author: Jet Chien
GitHub: https://github.com/jet-chien
Create Date: 2021/12/13
"""
from collaborating_filtering import CollaboratingFiltering

if __name__ == '__main__':
    csv_path = 'final_q18.csv'
    # df = pd.read_csv(csv_path)
    # print(df)
    cf = CollaboratingFiltering(
        csv_path,
        nature_val=1.5,
        k=2,
        thresh=3.0,
    )
    cf.user_base_predict()
    # cf.item_base_predict()
