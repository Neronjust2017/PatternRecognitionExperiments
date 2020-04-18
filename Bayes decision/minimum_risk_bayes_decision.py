import numpy as np
from sklearn.datasets import load_iris
from collections import  Counter
from numpy import *

def load_data(Cla_1=1 ,Cla_2=2):
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

    if Cla_1 == 1 and Cla_2 == 2:
        return setosad_data,  versicolor_data
    elif Cla_1 == 1 and Cla_2 == 3:
        return setosad_data,  virginica_data
    elif Cla_1 == 2 and Cla_2 == 3:
        return versicolor_data, virginica_data
    else:
        raise ValueError('...')

def get_train_test_split(data, label, split = 0.4):
    ind = np.random.permutation(len(data))
    split_ind = int(len(data)*split)
    data = data[ind]
    label = label[ind]
    train_data = data[:split_ind]
    train_label = label[:split_ind]
    test_data = data[split_ind:]
    test_label = label[split_ind:]
    return train_data, train_label, test_data, test_label

def class_pior_prob(train_label):

    nb_cla1 = sum(train_label == 0)
    nb_cla2 = sum(train_label == 1)
    return nb_cla1/(nb_cla1+nb_cla2), nb_cla2/(nb_cla1+nb_cla2)

def coVariance(X):  # 数据的每一行是一个样本，每一列是一个特征
    ro, cl = X.shape
    row_mean = np.mean(X, axis=0)
    X_Mean = np.zeros_like(X)
    X_Mean[:] = row_mean  # 把向量赋值给每一行
    X_Minus = X - X_Mean
    covarMatrix = np.zeros((cl, cl))
    for i in range(cl):
        for j in range(cl):
            covarMatrix[i, j] = (X_Minus[:, i].dot(X_Minus[:, j].T)) / (ro - 1)
    return covarMatrix

def gussion_paramters(train_data_cla):
    mean  =  np.mean(train_data_cla, axis=0)
    cov = coVariance(train_data_cla)
    return  mean, cov

def prob_gussion(x, mean, cov):
        cov = mat(cov)
        r = mat([x[0] - mean[0], x[1] - mean[1], x[2]-mean[2], x[3]-mean[3]])
        multi = r * cov.I * r.T
        multi = float(multi)  # 1乘1矩阵取内容
        k = exp(-multi / 2)  # .I求逆,.T转置
        k /= 2 * math.pi * linalg.det(cov) ** (1 / 2)  # linalg.det求行列式的值
        return k

def post_prob(x, mean1, cov1, mean2, cov2, pior_prob_1, pior_prob_2):
    k1 = prob_gussion(x, mean1, cov1) * pior_prob_1
    k2 = prob_gussion(x, mean2, cov2) * pior_prob_2
    return k1 / (k1 + k2), k2 / (k1 + k2)

if __name__ == '__main__':
    accs = []
    err_1s = []
    err_2s = []
    for i in range(500):
        Cla1, Cla2 = load_data(2,3)
        data = np.concatenate((Cla1,Cla2), axis=0)
        label = np.zeros((100,))
        label[50:100] = 1
        train_data, train_label, test_data, test_label = get_train_test_split(data, label, split=0.2)
        pior_prob_1, pior_prob_2 = class_pior_prob(train_label)
        train_data_cla1 = []
        train_data_cla2 = []
        for i in range(len(train_data)):
            if train_label[i] == 0:
                train_data_cla1.append(train_data[i])
            else:
                train_data_cla2.append(train_data[i])
        train_data_cla1 = np.array(train_data_cla1)
        train_data_cla2 = np.array(train_data_cla2)
        mu1, cov1 = gussion_paramters(train_data_cla1)
        mu2, cov2 = gussion_paramters(train_data_cla2)

        λ11 = 0
        λ12 = 1
        λ21 = 100
        λ22 = 0
        count = 0
        err_1 = 0
        err_2 = 0
        sum_1 = 0
        sum_2 = 0
        for i in range(len(test_data)):
            post_prob_1, post_prob_2 = post_prob(test_data[i],mu1,cov1,mu2,cov2,pior_prob_1,pior_prob_2)
            print("post_prob_1:")
            print(post_prob_1)
            print("post_prob_2:")
            print(post_prob_2)
            if post_prob_1*λ21 > post_prob_2*λ12:   # post_prob_1*λ21: R2
                pred = 0
            else:
                pred = 1
            if pred == test_label[i]:
                count += 1
            if test_label[i] == 0:
                sum_1 += 1
            else:
                sum_2 += 1
            if test_label[i] == 0 and pred == 1:
                err_1 += 1
            if test_label[i] == 1 and pred == 0:
                err_2 += 1
        acc = count / len(test_label)
        print("**********")
        print(count)
        print(acc)
        print(err_1/sum_1)
        print(err_2/sum_2)
        err_1s.append(err_1/sum_1)
        err_2s.append(err_2/sum_2)
        accs.append(acc)
    print(np.mean(np.array(accs)))
    print(np.mean(np.array(err_1s)))
    print(np.mean(np.array(err_2s)))