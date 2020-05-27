import numpy as np
from data import load_data
from sklearn import linear_model

if __name__ == '__main__':
    Data, label = load_data(Cla_1=1, Cla_2=2, Cla_3=3)

    l = [ [0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]]

    accss = []

    for k in range(len(l)):
        data = Data[:, l[k]]
        accs =[]
        for i in range(500):
            ind = np.random.permutation(len(data))

            ratio = 0.6

            ind_cut = int(len(data)*ratio)

            X_train = data[ind[0:ind_cut]]
            y_train = label[ind[0:ind_cut]]
            X_test = data[ind[ind_cut:]]
            y_test = label[ind[ind_cut:]]

            logistic = linear_model.LogisticRegression()

            logistic.fit(X_train, y_train)

            print("测试集数量:%d" %len(X_test))
            acc = logistic.score(X_test,y_test)
            print(acc)
            accs.append(acc)
        print(l[k])
        accs_mean = np.mean(np.array(accs))
        print(accs_mean)
        accss.append(accs_mean)
    np.set_printoptions(precision=4)
    print(np.array(accss))
