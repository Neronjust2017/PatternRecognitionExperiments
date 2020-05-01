import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
def load_data(Cla_1=1 ,Cla_2=2, Cla_3=None):
    data_size = 150
    data = load_iris()
    #鸢尾花的四个特征
    # 'sepal length (cm)'
    # 'sepal width (cm)'
    # 'petal length (cm)'
    # 'petal width (cm)'
    data_feature = data.feature_names
    print("Features:")
    print(data_feature)
    Iris_data = data.data
    print("Datas:")
    print(Iris_data)

    #鸢尾花的三个类别
    #'setosa' 1
    # 'versicolor' 2
    # 'virginica'  3
    target_names =data.target_names
    print("Target_names:")
    print(target_names)
    Iris_label=data.target
    print("Targets:")
    print(Iris_label)
    #
    setosad_data = Iris_data[:50]
    setosad_label = Iris_label[:50]
    versicolor_data = Iris_data[50:100]
    versicolor_label = Iris_label[50:100]
    virginica_data = Iris_data[100:]
    virginica_label = Iris_label[100:]

    if not Cla_3 is None:
        return setosad_data, versicolor_data, virginica_data
    if Cla_1 == 1 and Cla_2 == 2:
        return setosad_data,  versicolor_data
    elif Cla_1 == 1 and Cla_2 == 3:
        return setosad_data,  virginica_data
    elif Cla_1 == 2 and Cla_2 == 3:
        return versicolor_data, virginica_data
    else:
        raise ValueError('...')

def plot_data(data1, data2, data3 = None, label1 = 'sepal length (cm)', label2 = 'sepal width (cm)', cla1 = 'setosa' , cla2 = 'versicolor', cla3 = 'virginica'):
    plt.scatter(data1[:,0], data1[:, 1], color='blue', marker='x', label=cla1)
    plt.scatter(data2[:, 0], data2[:, 1], color='red', marker='o', label=cla2)
    if not data3 is None:
        plt.scatter(data3[:, 0], data3[:, 1], color='green', marker='s', label=cla3)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.legend(loc='upper left')
    plt.show()

def plot_data_clustering(data1,center1, data2, center2, data3 = None, center3=None, cla1 = 'Cluster1' , cla2 = 'Cluster2', cla3 = 'Cluster3'):
    plt.scatter(data1[:,0], data1[:, 1], color='blue', marker='o', label=cla1, s=10)
    plt.scatter(center1[0],center1[1], color='blue', marker='X', s=100)
    plt.scatter(data2[:, 0], data2[:, 1], color='red', marker='o', label=cla2, s=10)
    plt.scatter(center2[0],center2[1], color='red', marker='X', s=100)

    if not data3 is None:
        plt.scatter(data3[:, 0], data3[:, 1], color='green', marker='o', label=cla3, s=10)
        plt.scatter(center3[0], center3[1], color='green', marker='X', s=100)

    plt.legend(loc='upper left')
    plt.show()

def plot_data_clustering_2(data_list, centers):
    num_cluster = len(data_list)
    color = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink']
    for i in range(num_cluster):
        cluster = np.array(data_list[i])
        center = centers[i]
        plt.scatter(cluster[:, 0], cluster[:, 1], color=color[i], marker='o', label='cla'+str(i+1), s=10)
        plt.scatter(center[0], center[1], color=color[i], marker='X', s=100)
    plt.legend(loc='upper left')
    plt.show()

def print_nparray(array):
    array = np.array(array)
    np.set_printoptions(precision=3)
    print(array)


def eucliDist(A,B):
    return  round(math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)])),3)


