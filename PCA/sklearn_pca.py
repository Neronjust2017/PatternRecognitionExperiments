import numpy as np
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
from data import load_data
from sklearn import linear_model
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score

if __name__ == '__main__':

    Data, label = load_data(Cla_1=1, Cla_2=2, Cla_3=3)

    accss = []
    precisionss = []
    recallss = []
    f1ss = []
    for i in range(4):
        accs = []
        precisions = []
        recalls = []
        f1s = []
        for j in range(500):
            pca = KernelPCA(kernel="rbf", n_components=i+1)
            data = pca.fit_transform(Data)
            print(pca.lambdas_)
            ind = np.random.permutation(len(data))
            ratio = 0.6

            ind_cut = int(len(data) * ratio)
            X_train = data[ind[0:ind_cut]]
            y_train = label[ind[0:ind_cut]]
            X_test = data[ind[ind_cut:]]
            y_test = label[ind[ind_cut:]]

            logistic = linear_model.LogisticRegression()


            logistic.fit(X_train, y_train)
            print("测试集数量:%d" % len(X_test))
            acc = logistic.score(X_test, y_test)
            print(acc)

            y_pred = logistic.predict(X_test)
            # cr=classification_report(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")
            f1 = f1_score(y_test, y_pred, average="macro")
            accs.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        accss.append(np.mean(accs))
        precisionss.append(np.mean(precisions))
        recallss.append(np.mean(recalls))
        f1ss.append(np.mean(f1s))

    np.set_printoptions(precision=4)
    print(np.array(accss))
    print(np.array(precisionss))
    print(np.array(recallss))
    print(np.array(f1ss))
    # kpca = KernelPCA(n_components=1, kernel='linear', kernel_para=0.1)
    # X_new_2 = kpca.fit_transform(X)
    print(2)

