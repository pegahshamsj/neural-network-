import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

e = 0.5
input_nn = []
train_mse = []
test_mse = []

# read data from excel
df = pd.read_excel('Carbon dioxide.xlsx')
x_array = np.array(df)

# normalization
x = df.values
min_max = preprocessing.MinMaxScaler()
x_scaled = min_max.fit_transform(x)
df = pd.DataFrame(x_scaled)

# columns
df2 = df.iloc[1:]
df3 = df.iloc[2:]
df4 = df.iloc[3:]
df5 = df.iloc[4:]
df6 = df.iloc[5:]
df7 = df.iloc[6:]
df8 = df.iloc[7:]
df9 = df.iloc[8:]
df10 = df.iloc[9:]
df11 = df.iloc[10:]
df12 = df.iloc[11:]
df13 = df.iloc[12:]
df1 = df.values
df2 = df2.values
df3 = df3.values
df4 = df4.values
df5 = df5.values
df6 = df6.values
df7 = df7.values
df8 = df8.values
df9 = df9.values
df10 = df10.values
df11 = df11.values
df12 = df12.values
df13 = df13.values

# initialization
w1 = np.random.randint(0, 1, size=(18, 12))
w2 = np.random.randint(0, 1, size=(12, 18))
w3 = np.random.randint(0, 1, size=(6, 12))
w4 = np.random.randint(0, 1, size=(3, 6))
w5 = np.random.randint(0, 1, size=(2, 3))
w6 = np.random.randint(0, 1, size=(1, 2))


# activation function
def gaussian(net):
    return np.exp(-np.power(net - 0, 2.) / (2 * np.power(1, 2.)))


# activation derivative
def gaussian_derivative(net):
    net_t = np.array(net)
    o = []
    for i in range(len(net_t)):
        temp = -1 * (np.exp(-np.power((float(-net_t[i])) - 0, 2.) / (2 * np.power(1, 2.))))
        o.append(temp)
    net = np.matrix(o).transpose()
    return net


# neuron function
def activation_function(x, w):
    net = np.dot(w, x)
    net_t = np.array(net)
    o = []
    for i in range(len(net_t)):
        temp = gaussian(float(net_t[i]))
        o.append(temp)
    o = np.matrix(o).transpose()
    return o, net


for i in range(len(df13)):
    input_nn.append([float(df1[i]), float(df2[i]), float(df3[i]), float(df4[i]), float(df5[i]), float(df6[i]), float(df7[i]), float(df8[i]), float(df9[i]), float(df10[i]), float(df11[i]), float(df12[i]), float(df12[i])])


for i in range(1000):

    g = i + 1
    n = int(len(df13) * 0.75)
    m = len(df13)
    v = m-n
    train_error = []
    test_error = []
    train_out = []
    test_out = []

    for j in range(n):

        x = [input_nn[j][0], input_nn[j][1], input_nn[j][2], input_nn[j][3],input_nn[j][4], input_nn[j][5], input_nn[j][6], input_nn[j][7], input_nn[j][8], input_nn[j][9], input_nn[j][10], input_nn[j][11]]
        x = np.matrix(x).transpose()
        y = input_nn[j][12]

        o1, net1 = activation_function(x, w1)
        o2, net2 = activation_function(o1, w2)
        o3, net3 = activation_function(o2, w3)
        o4, net4 = activation_function(o3, w4)
        o5, net5 = activation_function(o4, w5)
        o6, net6 = activation_function(o5, w6)
        '''net6 = np.dot(w6, o5)
        o6 = net6'''

        error = y - o6
        f6d = gaussian_derivative(net6)
        f5d = gaussian_derivative(net5)
        f4d = gaussian_derivative(net4)
        f3d = gaussian_derivative(net3)
        f2d = gaussian_derivative(net2)
        f1d = gaussian_derivative(net1)

        constant = e * error

        w1 = w1 - np.dot(np.multiply(np.dot(w2.transpose(), np.multiply(np.dot(w3.transpose(), np.multiply(np.dot(w4.transpose(), np.multiply(np.dot(w5.transpose(), np.multiply(np.dot(w6.transpose(), np.dot(constant, f6d)), f5d)), f4d)), f3d)), f2d)), f1d), x.transpose())
        w2 = w2 - np.dot(np.multiply(np.dot(w3.transpose(), np.multiply(np.dot(w4.transpose(), np.multiply(np.dot(w5.transpose(), np.multiply(np.dot(w6.transpose(), np.dot(constant, f6d)), f5d)), f4d)), f3d)), f2d), o1.transpose())
        w3 = w3 - np.dot(np.multiply(np.dot(w4.transpose(), np.multiply(np.dot(w5.transpose(), np.multiply(np.dot(w6.transpose(), np.dot(constant, f6d)), f5d)), f4d)), f3d), o2.transpose())
        w4 = w4 - np.dot(np.multiply(np.dot(w5.transpose(), np.multiply(np.dot(w6.transpose(), np.dot(constant, f6d)), f5d)), f4d), o3.transpose())
        w5 = w5 - np.dot(np.multiply(np.dot(w6.transpose(), np.dot(constant, f6d)), f5d), o4.transpose())
        w6 = w6 - np.dot(np.dot(constant, f6d), o5.transpose())

    for j in range(n):

        x = [input_nn[j][0], input_nn[j][1], input_nn[j][2], input_nn[j][3], input_nn[j][4], input_nn[j][5],
             input_nn[j][6], input_nn[j][7], input_nn[j][8], input_nn[j][9], input_nn[j][10], input_nn[j][11]]
        x = np.matrix(x).transpose()
        y = input_nn[j][12]

        o1, net1 = activation_function(x, w1)
        o2, net2 = activation_function(o1, w2)
        o3, net3 = activation_function(o2, w3)
        o4, net4 = activation_function(o3, w4)
        o5, net5 = activation_function(o4, w5)
        o6, net6 = activation_function(o5, w6)

        train_error.append(float(y))
        train_out.append(float(o6))

    for j in range(n, m):

        x = [input_nn[j][0], input_nn[j][1], input_nn[j][2], input_nn[j][3], input_nn[j][4], input_nn[j][5],
             input_nn[j][6], input_nn[j][7], input_nn[j][8], input_nn[j][9], input_nn[j][10], input_nn[j][11]]
        x = np.matrix(x).transpose()
        y = input_nn[j][12]

        o1, net1 = activation_function(x, w1)
        o2, net2 = activation_function(o1, w2)
        o3, net3 = activation_function(o2, w3)
        o4, net4 = activation_function(o3, w4)
        o5, net5 = activation_function(o4, w5)
        o6, net6 = activation_function(o5, w6)

        test_error.append(float(y))
        test_out.append(float(o6))

    r = np.linspace(0, n, n)
    plt.subplot(2, 2, 1)
    plt.title('Comparision of MLP and target')
    plt.xlabel('Train samples')
    plt.ylabel('Outputs')
    plt.plot(r, train_error)
    plt.plot(r, train_out)

    t = np.linspace(0, v, v)
    plt.subplot(2, 2, 2)
    plt.title('Comparision of MLP and target')
    plt.xlabel('Test samples')
    plt.ylabel('Outputs')
    plt.plot(t, test_error)
    plt.plot(t, test_out)

    train_mse.append(mean_squared_error(train_error, train_out))
    test_mse.append(mean_squared_error(test_error, test_out))

    d = np.linspace(0, g, g)
    plt.subplot(2, 2, 3)
    plt.title('Train error mse')
    plt.xlabel('Epoch')
    plt.ylabel('Error mse')
    plt.plot(d, train_mse)

    plt.subplot(2, 2, 4)
    plt.title('Test error mse')
    plt.xlabel('Epoch')
    plt.ylabel('Error mse')
    plt.plot(d, test_mse)
    plt.show()
