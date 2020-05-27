import numpy as np
from scipy.linalg import svd, eig
from sklearn.datasets import make_circles
import math
from data import load_data
from sklearn import linear_model
from sklearn.metrics import classification_report,f1_score,precision_score,recall_score

class PCA:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.n_features = X.shape[1]
        _, sigmas, VT = svd(X - self.mean_, full_matrices=False)
        V = VT.T
        a = np.argsort(sigmas)
        # a = np.array([0,3,2,1])
        V = V[:, np.argsort(sigmas)[::-1]]
        # V = V[:, a[::-1]]
        # V = V[:, [3,0,1,2]]
        b = np.linalg.norm(V, axis=0)
        V /= np.linalg.norm(V, axis=0)
        self.eigenvalue = sigmas**2 / X.shape[0]
        self.scalings_ = V

    def transform(self, X):
        if not hasattr(self, 'scalings_'):
            raise Exception('Please run `fit` before transform')
        if not hasattr(self, 'mean_'):
            raise Exception('Please run `fit` before transform')
        assert X.shape[1] == self.n_features, 'X.shape[1] != self.n_features'
        if self.n_components is None:
            self.n_components = self.n_features
        return ((X - self.mean_) @ self.scalings_)[:, :self.n_components]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class KernelPCA:
    def __init__(self, n_components=None, kernel='linear', kernel_para=0.1):
        self.__kernels = {'linear': self.__kernel_linear,
                          'rbf': self.__kernel_rbf}
        assert kernel in self.__kernels, 'arg kernel =\'' + kernel + '\' is not available'
        self.n_components = n_components
        self.kernel = kernel
        self.kernel_para = kernel_para

    def __kernel_linear(self, x, y):
        return x.T @ y

    def __kernel_rbf(self, x, y):
        result = np.zeros((x.shape[1], y.shape[1]))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.exp(-self.kernel_para * (x[:, i] - y[:, j]).T @ (x[:, i] - y[:, j]))
        return result

    def fit(self, X):
        self.n_samples, self.n_features = X.shape
        K = self.__kernels[self.kernel](X.T, X.T)
        one_M = np.ones((self.n_samples, self.n_samples)) / self.n_samples
        K = K - one_M @ K - K @ one_M + one_M @ K @ one_M
        e_vals, e_vecs = eig(K)
        e_vals, e_vecs = np.real(e_vals), np.real(e_vecs)
        e_vecs /= np.linalg.norm(e_vecs, axis=0)
        e_vecs = e_vecs[:, np.argsort(e_vals)[::-1]] / np.sqrt(np.sort(e_vals)[::-1])
        self.scalings_ = e_vecs
        self.e_vals_ = e_vals
        self.__X = X

    def transform(self, X):
        if not hasattr(self, 'scalings_'):
            raise Exception('Please run `fit` before transform')
        assert X.shape[1] == self.n_features, 'X.shape[1] != self.n_features'
        if self.n_components is None:
            self.n_components = X.shape[1]
        K = self.__kernels[self.kernel](X.T, self.__X.T)
        one_M = np.ones((self.n_samples, self.n_samples)) / self.n_samples
        K = K - one_M @ K - K @ one_M + one_M @ K @ one_M
        return (K @ self.scalings_)[:, :self.n_components]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

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
            pca = KernelPCA(n_components=i+1)
            data = pca.fit_transform(Data)
            print(pca.e_vals_)
            ind = np.random.permutation(len(data))
            ratio = 0.6

            ind_cut = int(len(data) * ratio)
            X_train = data[ind[0:ind_cut]]
            y_train = label[ind[0:ind_cut]]
            X_test = data[ind[ind_cut:]]
            y_test = label[ind[ind_cut:]]

            logistic = linear_model.LogisticRegression()

            try:
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
            except ValueError as e:
                print("ValueError")
            finally:
                pass
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