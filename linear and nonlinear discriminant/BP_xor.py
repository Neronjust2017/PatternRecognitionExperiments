import numpy as np
from sklearn.datasets import load_iris
from keras.utils import to_categorical
import random
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
# 双曲正切函数,该函数为奇函数
def tanh(x):
    return np.tanh(x)
# tanh导函数性质:f'(t) = 1 - f(x)^2
def tanh_prime(x):
    return 1.0 - tanh(x)**2
class NeuralNetwork:
    def __init__(self, layers, activation = 'tanh'):
        """
        :参数layers: 神经网络的结构(输入层-隐含层-输出层包含的结点数列表)
        :参数activation: 激活函数类型
        """
        if activation == 'tanh':    # 也可以用其它的激活函数
            self.activation = tanh
            self.activation_prime = tanh_prime
        else:
            pass

        # 存储权值矩阵
        self.weights = []

        # range of weight values (-1,1)
        # 初始化输入层和隐含层之间的权值
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1     # add 1 for bias node
            self.weights.append(r)

            # 初始化输出层权值
            r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
            self.weights.append(r)

    def fit(self, X, Y, learning_rate=0.2, epochs=10000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        X = np.hstack([np.ones((X.shape[0],1)),X])

        for k in range(epochs):     # 训练固定次数
            if k % 1000 == 0: print('epochs:', k)

            # Return random integers from the discrete uniform distribution in the interval [0, low).
            i = np.random.randint(X.shape[0],high=None)
            a = [X[i]]   # 从m个输入样本中随机选一组

            for l in range(len(self.weights)):
                dot_value = np.dot(a[l], self.weights[l])   # 权值矩阵中每一列代表该层中的一个结点与上一层所有结点之间的权值
                activation = self.activation(dot_value)
                a.append(activation)

             # 反向递推计算delta:从输出层开始,先算出该层的delta,再向前计算
            error = Y[i] - a[-1]    # 计算输出层delta
            deltas = [error * self.activation_prime(a[-1])]

            # 从倒数第2层开始反向计算delta
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()    # 逆转列表中的元素

            # backpropagation
            # 1. Multiply its output delta and input activation to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):  # 逐层调整权值
                layer = np.atleast_2d(a[i])     # View inputs as arrays with at least two dimensions
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * np.dot(layer.T, delta) # 每输入一次样本,就更新一次权值

    def predict(self, x):
        a = np.concatenate((np.ones(1), np.array(x)))       # a为输入向量(行向量)
        for l in range(0, len(self.weights)):               # 逐层计算输出
            a = self.activation(np.dot(a, self.weights[l]))
        return a

def random_point(x, y, radius, N):
    a = 2 * math.pi * np.array([random.random() for _ in range(N)])
    r = np.array([random.random() for _ in range(N)])
    x1 = radius * np.sqrt(r) * np.cos(a) + x
    y1 = radius * np.sqrt(r) * np.sin(a) + y
    plt.scatter(x1, y1, s=1)
    plt.show()
    return x1, y1

if __name__ == '__main__':
    n = 1000
    x00, y00 = random_point(0,0, 0.1, n)
    x01, y01 = random_point(0,1, 0.1, n)
    x10, y10 = random_point(1,0, 0.1, n)
    x11, y11 = random_point(1,1, 0.1, n)

    x00 = x00.reshape([x00.shape[0],1])
    x01 = x01.reshape([x01.shape[0], 1])
    x10 = x10.reshape([x10.shape[0], 1])
    x11 = x11.reshape([x11.shape[0], 1])
    y00 = y00.reshape([y00.shape[0], 1])
    y01 = y01.reshape([y01.shape[0], 1])
    y10 = y10.reshape([y10.shape[0], 1])
    y11 = y11.reshape([y11.shape[0], 1])

    data = np.concatenate((np.concatenate((x00, y00),axis=1),
                           np.concatenate((x01, y01),axis=1),
                           np.concatenate((x10, y10),axis=1),
                           np.concatenate((x11, y11),axis=1)
                          ),axis=0)

    plt.scatter(np.array([x00, x11]),np.array([y00, y11]), s=1,  color='blue', label='0')
    plt.scatter(np.array([x01, x10]), np.array([y01, y10]), s=1, color='red', label='1')
    plt.legend(loc='upper left')
    plt.show()
    # plt.scatter(data[:,0], data[:,1], s=1)
    # plt.show()

    label = np.zeros((4*n,))
    label[n:2*n] = 1

    accs = []
    for i in range(500):
        ind = np.random.permutation(len(data))

        ratio = 0.6

        ind_cut = int(len(data) * ratio)

        X_train = data[ind[0:ind_cut]]
        y_train = label[ind[0:ind_cut]]
        X_test = data[ind[ind_cut:]]
        y_test = label[ind[ind_cut:]]

        nn = NeuralNetwork([2, 2, 1])  # 网络结构: 2输入1输出,1个隐含层(包含2个结点)

        nn.fit(X_train, y_train, learning_rate=0.1, epochs=10000)  # 训练网络

        print('w:', nn.weights)  # 调整后的权值列表

        print("测试集数量:%d" % len(X_test))

        count = 0

        print("测试样本  预测标签  真实标签")

        y_test_true = []
        y_test_false = []

        i = 0
        for x, y in zip(X_test, y_test):
            pred = nn.predict(x)
            if pred >= 0.5:
                pred = 1
            else:
                pred = 0
            if pred == y:
                count += 1
                y_test_true.append(X_test[i])
            else:
                y_test_false.append(X_test[i])
            i += 1
            print(x, pred, y)
        acc = count / len(X_test)
        print("acc", acc)

        y_test_true = np.array(y_test_true)
        y_test_false = np.array(y_test_false)
        # plt.scatter(y_test_true[:,0],y_test_true[:,1], s=1, color='blue',marker='o', label='true')
        # if acc != 1.0:
        #     plt.scatter(y_test_false[:,0],y_test_false[:,1], s=1, color='red',marker='x', label='flase')
        # plt.legend(loc='upper left')
        # plt.show()
        accs.append(acc)

    print("mean acc:")
    print(np.mean(accs))

