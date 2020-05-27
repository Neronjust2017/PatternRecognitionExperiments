import numpy as np
from scipy.linalg import eig
from sklearn.datasets import make_circles
from data import load_data

class LinearDiscriminantAnalysis:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y):
        self.n_features = X.shape[1]
        u1 = X[y == 1, :].mean(axis=0).reshape((-1, 1))
        u0 = X[y == 0, :].mean(axis=0).reshape((-1, 1))
        Sb = (u1 - u0) @ (u1 - u0).T
        Sw = (X[y == 1, :].T - u1) @ (X[y == 1, :].T - u1).T + (X[y == 0, :].T - u0) @ (X[y == 0, :].T - u0).T
        e_vals, e_vecs = eig(np.linalg.pinv(Sw) @ Sb)
        e_vals, e_vecs = np.real(e_vals), np.real(e_vecs)
        e_vecs = e_vecs[:, np.argsort(e_vals)[::-1]]
        e_vecs /= np.linalg.norm(e_vecs, axis=0)
        self.scalings_ = e_vecs

    def transform(self, X):
        if not hasattr(self, 'scalings_'):
            raise Exception('Please run `fit` before transform')
        assert X.shape[1] == self.n_features, 'X.shape[1] != self.n_features'
        if self.n_components is None:
            self.n_components = self.n_features
        return (X @ self.scalings_)[:, :self.n_components]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

if __name__ == '__main__':
    # X, y = make_circles(n_samples=400, factor=.3, noise=.05, random_state=0)
    # lda = LinearDiscriminantAnalysis(n_components=1)
    # X_new = lda.fit_transform(X,y)
    # print(2)
    Data, Label = load_data(Cla_1=1, Cla_2=2, Cla_3=3)
    X1 = Data[:50]
    X2 = Data[50:100]
    X3 = Data[100:]
    y = np.zeros(100)
    y[50:] = 1
    lda = LinearDiscriminantAnalysis(n_components=1)
    X_new = lda.fit_transform(np.concatenate((X2,X3), axis=0), y)

    u1 = X_new[y == 1, :].mean(axis=0).reshape((-1, 1))
    u0 = X_new[y == 0, :].mean(axis=0).reshape((-1, 1))

    u = (u1 + u0) / 2
    print(u1, u0, u)

    flag = 1
    if u1 > u0:
        flag = 0

    y_pred = np.zeros(len(y))
    for i in range(len(y)):
        if X_new[i] < u :
            y_pred[i] = flag
        else:
            y_pred[i] = 1 - flag
        print(y[i], y_pred[i])
        if y[i] != y_pred[i]:
            print(i)

