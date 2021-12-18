import math
import pandas as pd
from decimal import Decimal as D


class Point:
    def __init__(self, name: str, x, y):
        self.name = name
        self.x = x
        self.y = y
        self.coord = (self.x, self.y)

    def __repr__(self):
        return f"{self.name}:({self.x}, {self.y})"

    def update(self, x, y):
        self.x = x
        self.y = y
        self.coord = (self.x, self.y)


def get_dist(p1: Point, p2: Point, rounded=True, round_d=2) -> D:
    dist_square = get_dist_square(p1, p2)
    d = dist_square.sqrt()
    if rounded:
        d = round(d, round_d)

    return D(d)


def get_dist_square(p1: Point, p2: Point) -> D:
    x1 = p1.x
    x2 = p2.x
    y1 = p1.y
    y2 = p2.y

    x1 = D(x1)
    x2 = D(x2)
    y1 = D(y1)
    y2 = D(y2)

    d = (x1 - x2) ** 2 + (y1 - y2) ** 2

    return d


def get_nearest_centroid(p: Point, c_ls: list) -> (Point, list):
    nc = None
    nd = math.inf
    d_ls = list()
    for c in c_ls:
        d = get_dist(p, c)
        d_ls.append(d)
        if d < nd:
            nd = d
            nc = c

    return nc, d_ls


def get_iteration_record(pt_ls: list, c_ls: list) -> list:
    result = list()
    for p in pt_ls:
        nearest_c, d_ls = get_nearest_centroid(p, c_ls)
        d_ls.append(nearest_c.name)
        r = (p.name, d_ls)
        result.append(r)
    return result


def get_iteration_df(pt_ls: list, c_ls: list) -> (pd.DataFrame, list):
    cols = [f"{c.name} dist" for c in c_ls]
    cols.append('Cluster Assigned')
    df = pd.DataFrame(columns=cols)
    iter_record = get_iteration_record(pt_ls, c_ls)
    for r in iter_record:
        r_idx_name = r[0]
        df.loc[r_idx_name] = r[1]

    partition = list()
    for c_name, c_group in df.groupby('Cluster Assigned'):
        c_p_name_ls = list(c_group.index)
        c_pts = get_points(c_p_name_ls, pt_ls)
        p = (c_name, c_pts)
        partition.append(p)

    return df, partition


def print_iteration_table(iter_name, iter_df: pd.DataFrame):
    print(f"{iter_name} iteration : ")
    print(iter_df.T)
    print()


def get_new_c_coord(c_point_ls: list, pt_ls: list, round_d=1):
    x_sum, y_sum = 0, 0
    for c_p_name in c_point_ls:
        p = get_point(c_p_name, pt_ls)
        x_sum += p.x
        y_sum += p.y

    x = x_sum / len(c_point_ls)
    y = y_sum / len(c_point_ls)
    return round(x, round_d), round(y, round_d)


def get_new_c_coord_ls(iter_df: pd.DataFrame, pt_ls: list) -> list:
    res = list()
    cluster_gb = iter_df.groupby('Cluster Assigned')
    for c_name, c_group in cluster_gb:
        # print(c_name)
        c_point_ls = list(c_group.index)
        # print(c_point_ls)
        new_c_coord = get_new_c_coord(c_point_ls, pt_ls)
        res.append(new_c_coord)

    return res


def get_point(p_name: str, p_ls: list) -> Point:
    for p in p_ls:
        if p_name == p.name:
            return p


def get_points(p_name_list, pt_ls: list) -> list:
    res = list()
    for p_name in p_name_list:
        p = get_point(p_name, pt_ls)
        res.append(p)
    return res


def need_update_c(c_ls: list, new_c_coord_ls: list):
    for old_c, new_coord in zip(c_ls, new_c_coord_ls):
        old_coord = old_c.coord
        if old_coord != new_coord:
            return True
    return False


def update_centroid_ls(c_ls, new_c_coord_ls):
    for old_c, new_coord in zip(c_ls, new_c_coord_ls):
        old_c.update(new_coord[0], new_coord[1])


def print_centroid_ls(iter_name, c_ls):
    print(f"#{iter_name} iteration centroid: {c_ls}")


def get_sse(partition, c_ls: list, rounded=True, round_d=2):
    e = 0
    print('SSE = 0', end='')
    for c_name, c_p_ls in partition:
        c = get_point(c_name, c_ls)
        for c_p in c_p_ls:
            d_square = get_dist_square(c, c_p)
            if rounded:
                d_square = round(d_square, round_d)

            print(f" + {d_square}", end='')
            e += d_square

    print(f'= {e}')
    return e


def k_means(pt_ls: list, c_ls: list):
    result = list()
    flag = True
    iteration = 0
    while flag:
        iteration += 1
        iter_df, partition = get_iteration_df(pt_ls, c_ls)
        # print(partition)
        print_centroid_ls(f"{iteration}", c_ls)
        print_iteration_table(f"#{iteration}", iter_df)

        new_c_coord_ls = get_new_c_coord_ls(iter_df, pt_ls)
        result = partition
        sse = get_sse(partition, c_ls)
        print(f"SSE : {sse} \n")

        if need_update_c(c_ls, new_c_coord_ls):
            update_centroid_ls(c_ls, new_c_coord_ls)
        else:
            iteration += 1
            print(f"The centroid will be the same in #{iteration} iteration")

            iter_df, partition = get_iteration_df(pt_ls, c_ls)
            # print(partition)
            print_centroid_ls(f"{iteration}", c_ls)
            print_iteration_table(f"#{iteration}", iter_df)

            new_c_coord_ls = get_new_c_coord_ls(iter_df, pt_ls)
            result = partition
            sse = get_sse(partition, c_ls)
            print(f"SSE : {sse} \n")

            return result

    return result


if __name__ == '__main__':
    point_ls = [
        Point('A1', 2, 8),
        Point('A2', 6, 8),
        Point('A3', 4, 5),
        Point('A4', 4, 4),
        Point('A5', 4, 0),

    ]

    centroid_ls = [
        Point('C1', 2, 8),  # A1
        Point('C2', 4, 4),  # A4

    ]

    K = len(centroid_ls)

    y = k_means(point_ls, centroid_ls)
    # print(y)
