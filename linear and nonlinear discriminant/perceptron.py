import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
class Perception(object):
    """
    eta:学习率
    n_iter:权重向量的训练次数
    w_:神经分叉权重向量
    error_:用于记录神经元判断出错次数
    """
    def __init__(self, eta = 0.01, n_iter = 10):
        self.eta = eta
        self.n_iter = n_iter
        pass

    def fit(self, x, y):
        """
        输入训练数据，培训神经元，x输入样本向量，y对应样本分类

        x:shape[n_samples, n_features]
        x:[[1, 2, 3], [4, 5, 6]]
        n_samples:2
        n_features:3

        y:[1, -1]
        """

        """
        初始化权重向量为0
        加一是因为前面算法提到的w0，也就是步调函数的阈值
        """
        # self.w_ = np.zeros(1 + x.shape[1])
        self.w_ = np.random.randn(1 + x.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            # errors = 0
            """
            x:[[1, 2, 3], [4, 5, 6]]
            y:[1, -1]
            zip(x,y) = [[1, 2, 3, 1], [4, 5, 6, -1]]
            """
            for xi, target in zip(x, y):
                """
                update = η * (y - y')
                """
                # z = self.predict(xi)
                update = self.eta * (target - self.predict(xi))
                """
                xi是一个向量
                update * xi 等价：
                [▽w[1]=x[1]*update,▽w[2]=x[2]*update,▽w[3]=x[3]*update]
                """
                self.w_[1:] += update * xi
                self.w_[0] += update

                # errors += int(update != 0.0)
                # self.errors_.append(errors)
                # pass

            # pass
            errors = 82 - np.sum((self.predict(x) * y) == 1)
            if errors == 0:
                self.plot_decision_regions(x, y, _)
                print(_)
                return _
            self.errors_.append(errors)
            print(errors)
            # if _ % 10 == 0 :
            #     # plot_data(x[:38], x[38:], self.w_, _)
            #     print("iter:", _)
            #     print(self.w_)
            #     self.plot_decision_regions(x, y, _)
            if _+1 == self.n_iter:
                self.plot_decision_regions(x, y, _)

        pass
    def net_input(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]
        pass
    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0, 1, -1)
        pass

    def plot_decision_regions(self, x, y, iter, label1 = 'sepal length (cm)', label2 = 'sepal width (cm)', resolution=0.02):
        marker = ('s', 'x', 'o', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max()
        x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max()

        # print(x1_min, x1_max)
        # print(x2_min, x2_max)

        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                               np.arange(x2_min, x2_max, resolution))

        z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

        # print(xx1.ravel())
        # print(xx2.ravel())
        # print(z)

        z = z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=x[y == cl, 0], y=x[y == cl, 1], alpha=0.8, c=cmap(idx),
                        marker=marker[idx], label=cl)
        plt.xlabel(label1)
        plt.ylabel(label2)
        plt.legend(loc='upper left')
        plt.title("iter%d" %iter)
        plt.show()

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

def plot_data(data1, data2, w=None, iter=0, label1 = 'sepal length (cm)', label2 = 'sepal width (cm)', cla1 = 'setosa' , cla2 = 'versicolor'):
    plt.scatter(data1[:,0], data1[:, 1], color='blue', marker='x', label=cla1)
    plt.scatter(data2[:, 0], data2[:, 1], color='red', marker='o', label=cla2)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.legend(loc='upper left')
    if  not w is None:
        y = (w[1] * x + w[0]) / -w[2]
        plt.plot(x, y , color='black')
        plt.title('iter_'+str(iter))
    plt.show()



if __name__ == '__main__':

    # 'setosa' 1 'versicolor' 2 'virginica'  3
    # 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'
    data1, data2 = load_data(Cla_1=1, Cla_2=2)
    data1 = data1[:,:2]
    data2 = data2[:,:2]
    data1 = np.unique(data1,axis=0)
    data2 = np.unique(data2,axis=0)
    plot_data(data1, data2)
    # print(data1[:, 0:3:2])
    # plot_data(data1[:, 0:3:2], data2[:, 0:3:2],label1 ='sepal length (cm)', label2='petal length (cm)')

    # x = np.concatenate((data1[:, 0:3:2], data2[:, 0:3:2]))
    x = np.concatenate((data1, data2))
    y = np.ones((len(data1)+len(data2),))
    y[len(data1):] = -1

    # y[40] = 1

    iters = []
    for i in range(1):
        ppn = Perception(eta=0.1, n_iter=1000)
        iter = ppn.fit(x, y)
        w = ppn.w_
        errors = ppn.errors_
        print(w)
        # plt.plot(errors)
        # plot_data(data1[:,:2], data2[:,:2],'sepal length (cm)', 'sepal width (cm)', 'setosa', 'versicolor',w)
        # plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
        # plt.xlabel('Epoch')
        # plt.ylabel('错误分类次数')
        # plt.show()
        iters.append(iter)
    print(np.mean(np.array(iters)))

