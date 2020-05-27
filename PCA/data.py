from sklearn.datasets import load_iris
from keras.utils import to_categorical

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