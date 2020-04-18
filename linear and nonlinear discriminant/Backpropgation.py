import numpy as np
from sklearn.datasets import load_iris
from keras.utils import to_categorical
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
        return Iris_data, Iris_label
    if Cla_1 == 1 and Cla_2 == 2:
        return setosad_data,  versicolor_data
    elif Cla_1 == 1 and Cla_2 == 3:
        return setosad_data,  virginica_data
    elif Cla_1 == 2 and Cla_2 == 3:
        return versicolor_data, virginica_data
    else:
        raise ValueError('...')

if __name__ == '__main__':
    data, label = load_data(Cla_1=1, Cla_2=2, Cla_3=3)

    accs =[]
    for i in range(500):
        ind = np.random.permutation(len(data))

        ratio = 0.6

        ind_cut = int(len(data)*ratio)

        X_train = data[ind[0:ind_cut]]
        y_train_true = label[ind[0:ind_cut]]
        X_test = data[ind[ind_cut:]]
        y_test_true = label[ind[ind_cut:]]

        y_train = to_categorical(y_train_true)
        y_test = to_categorical(y_test_true)
        # nn = NeuralNetwork([2,2,1])     # 网络结构: 2输入1输出,1个隐含层(包含2个结点)
        #
        #
        # X = np.array([[0, 0],           # 输入矩阵(每行代表一个样本,每列代表一个特征)
        #                 [0, 1],
        #                 [1, 0],
        #                 [1, 1]])
        # Y = np.array([0, 1, 1, 0])      # 期望输出

        nn = NeuralNetwork([4, 2, 3])  # 网络结构: 2输入1输出,1个隐含层(包含2个结点)

        nn.fit(X_train, y_train, learning_rate=0.1, epochs=5000)                    # 训练网络

        print ('w:', nn.weights)      # 调整后的权值列表

        print("测试集数量:%d" %len(X_test))

        count = 0

        print("测试样本  预测标签  真实标签")
        for x, y  in zip(X_test, y_test_true):
            pred = nn.predict(x)
            pred = np.argmax(pred)
            if pred == y:
                count += 1
            print(x, pred, y)
        acc = count / len(X_test)
        print("acc", acc)
        accs.append(acc)
    print(np.mean(np.array(accs)))