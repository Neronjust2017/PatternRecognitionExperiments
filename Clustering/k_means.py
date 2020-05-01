import numpy  as np
from collections import defaultdict
from random import uniform
from math import sqrt
from utils import load_data, plot_data, plot_data_clustering, print_nparray,eucliDist

def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    NB. points can have more dimensions than 2

    Returns a new point which is the center of all the points.
    """
    dimensions = len(points[0])

    new_center = []

    for dimension in range(dimensions):
        dim_sum = 0  # dimension sum
        for p in points:
            dim_sum += p[dimension]

        # average of each dimension
        new_center.append(dim_sum / float(len(points)))

    return new_center


def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers where `k` is the number of unique assignments.
    """
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)

    for points in new_means.values():
        centers.append(point_avg(points))

    return centers


def assign_points(data_points, centers):
    """
    Given a data set and a list of points betweeen other points,
    assign each point to an index that corresponds to the index
    of the center point on it's proximity to that point.
    Return a an array of indexes of centers that correspond to
    an index in the data set; that is, if there are N points
    in `data_set` the list we return will have N elements. Also
    If there are Y points in `centers` there will be Y unique
    possible values within the returned list.
    """
    assignments = []
    for point in data_points:
        shortest =  float('inf')  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    """
    dimensions = len(a)

    _sum = 0
    for dimension in range(dimensions):
        difference_sq = (a[dimension] - b[dimension]) ** 2
        _sum += difference_sq
    return sqrt(_sum)


def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    find the minimum and maximum for each coordinate, a range.
    Generate `k` random points between the ranges.
    Return an array of the random points within the ranges.
    """
    centers = []
    dimensions = len(data_set[0])
    min_max = defaultdict(int)

    for point in data_set:
        for i in range(dimensions):
            val = point[i]
            min_key = 'min_%d' % i
            max_key = 'max_%d' % i
            if min_key not in min_max or val < min_max[min_key]:
                min_max[min_key] = val
            if max_key not in min_max or val > min_max[max_key]:
                min_max[max_key] = val

    for _k in range(k):
        rand_point = []
        for i in range(dimensions):
            min_val = min_max['min_%d' % i]
            max_val = min_max['max_%d' % i]

            rand_point.append(uniform(min_val, max_val))

        centers.append(rand_point)

    return centers


def k_means(dataset, k):
    k_points = generate_k(dataset, k)
    print_nparray(k_points)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    centers = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
        centers = new_centers
    return zip(assignments, dataset), centers

if __name__ == '__main__':


    # 两类 2个特征 k =2
    # data1, data2 = load_data(Cla_1=1, Cla_2=2)
    # data1 = data1[:, :2]
    # data2 = data2[:, :2]
    # data1 = np.unique(data1, axis=0)
    # data2 = np.unique(data2, axis=0)
    # plot_data(data1, data2)
    #
    # k =2
    #
    # data = np.concatenate((data1,data2),axis=0)
    # s, centers = k_means(data, k)
    #
    # s = list(s)
    #
    # center1 = centers[0]
    # center2 = centers[1]
    #
    # data1_new = []
    # data2_new = []
    # for i in range(len(s)):
    #     if s[i][0] == 0:
    #         data1_new.append(s[i][1])
    #     else:
    #         data2_new.append(s[i][1])
    #
    # data1_new = np.array(data1_new)
    # data2_new = np.array(data2_new)
    #
    # plot_data_clustering(data1_new,center1, data2_new,center2)


    # 两类 2个特征 k=3
    # data1, data2 = load_data(Cla_1=1, Cla_2=2)
    # data1 = data1[:, :2]
    # data2 = data2[:, :2]
    # data1 = np.unique(data1, axis=0)
    # data2 = np.unique(data2, axis=0)
    # plot_data(data1, data2)
    #
    # k = 3
    #
    # data = np.concatenate((data1, data2), axis=0)
    # s, centers = k_means(data, k)
    #
    # s = list(s)
    #
    # center1 = centers[0]
    # center2 = centers[1]
    # center3 = centers[2]
    #
    # data1_new = []
    # data2_new = []
    # data3_new = []
    # for i in range(len(s)):
    #     if s[i][0] == 0:
    #         data1_new.append(s[i][1])
    #     elif s[i][0] == 1:
    #         data2_new.append(s[i][1])
    #     else:
    #         data3_new.append(s[i][1])
    #
    # data1_new = np.array(data1_new)
    # data2_new = np.array(data2_new)
    # data3_new = np.array(data3_new)
    #
    #
    # plot_data_clustering(data1_new, center1, data2_new, center2, data3_new, center3)


    # 两类 4个特征 k=2
    # data1, data2 = load_data(Cla_1=1, Cla_2=2)
    #
    # k =2
    #
    # data = np.concatenate((data1,data2),axis=0)
    # s, centers = k_means(data, k)
    #
    # s = list(s)
    #
    # center1 = centers[0]
    # center2 = centers[1]
    #
    # print("聚类中心距离：")
    # print(eucliDist(center1,center2))
    #
    # data1_new = []
    # data2_new = []
    # for i in range(len(s)):
    #     if s[i][0] == 0:
    #         data1_new.append(s[i][1])
    #     else:
    #         data2_new.append(s[i][1])
    #
    # data1_new = np.array(data1_new)
    # data2_new = np.array(data2_new)
    #
    # print("聚类域样本数目：")
    # print(len(data1_new), len(data2_new))
    #
    # print("类间标准差：")
    # print(np.std(data1_new, axis=0))
    # print(np.std(data2_new, axis=0))


    # 三类 2个特征 k =3
    # data1, data2, data3 = load_data(Cla_1=1, Cla_2=2, Cla_3=3)
    # data1 = data1[:, 0:3:2]
    # data2 = data2[:, 0:3:2]
    # data3 = data3[:, 0:3:2]
    # data1 = np.unique(data1, axis=0)
    # data2 = np.unique(data2, axis=0)
    # data3 = np.unique(data3, axis=0)
    # plot_data(data1, data2, data3,label1 ='sepal length (cm)', label2='petal length (cm)')
    #
    # k =3
    #
    # data = np.concatenate((data1,data2,data3),axis=0)
    # s, centers = k_means(data, k)
    #
    # s = list(s)
    #
    # center1 = centers[0]
    # center2 = centers[1]
    # center3 = centers[2]
    #
    # data1_new = []
    # data2_new = []
    # data3_new = []
    #
    # for i in range(len(s)):
    #     if s[i][0] == 0:
    #         data1_new.append(s[i][1])
    #     elif s[i][0] == 1:
    #         data2_new.append(s[i][1])
    #     else:
    #         data3_new.append(s[i][1])
    #
    # data1_new = np.array(data1_new)
    # data2_new = np.array(data2_new)
    # data3_new = np.array(data3_new)
    #
    # plot_data_clustering(data1_new,center1, data2_new,center2, data3_new,center3)


    # # 三类 2个特征 k =2
    # data1, data2, data3 = load_data(Cla_1=1, Cla_2=2, Cla_3=3)
    # data1 = data1[:, 0:3:2]
    # data2 = data2[:, 0:3:2]
    # data3 = data3[:, 0:3:2]
    # data1 = np.unique(data1, axis=0)
    # data2 = np.unique(data2, axis=0)
    # data3 = np.unique(data3, axis=0)
    # plot_data(data1, data2, data3,label1 ='sepal length (cm)', label2='petal length (cm)')
    #
    # k =2
    #
    # data = np.concatenate((data1,data2,data3),axis=0)
    # s, centers = k_means(data, k)
    #
    # s = list(s)
    #
    # center1 = centers[0]
    # center2 = centers[1]
    #
    # data1_new = []
    # data2_new = []
    #
    # for i in range(len(s)):
    #     if s[i][0] == 0:
    #         data1_new.append(s[i][1])
    #     else:
    #         data2_new.append(s[i][1])
    #
    #
    # data1_new = np.array(data1_new)
    # data2_new = np.array(data2_new)
    #
    # plot_data_clustering(data1_new,center1, data2_new,center2)


    # 三类 4个特征 k =3
    data1, data2, data3 = load_data(Cla_1=1, Cla_2=2, Cla_3=3)

    k =3

    data = np.concatenate((data1,data2,data3),axis=0)
    s, centers = k_means(data, k)

    s = list(s)

    center1 = centers[0]
    center2 = centers[1]
    center3 = centers[2]

    data1_new = []
    data2_new = []
    data3_new = []

    for i in range(len(s)):
        if s[i][0] == 0:
            data1_new.append(s[i][1])
        elif s[i][0] == 1:
            data2_new.append(s[i][1])
        else:
            data3_new.append(s[i][1])

    data1_new = np.array(data1_new)
    data2_new = np.array(data2_new)
    data3_new = np.array(data3_new)

    print("聚类中心距离：")
    print("1,2", eucliDist(center1,center2))
    print("1,3", eucliDist(center1, center3))
    print("2,3", eucliDist(center2, center3))

    print("聚类域样本数目：")
    print(len(data1_new), len(data2_new), len(data3_new))

    print("类间标准差：")
    print(np.std(data1_new, axis=0))
    print(np.std(data2_new, axis=0))
    print(np.std(data3_new, axis=0))


