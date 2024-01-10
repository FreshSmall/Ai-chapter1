import numpy as np
import json
import matplotlib.pyplot as plt

# 读入训练数据
from network import Network
from data import DataSource
from network_v1 import NetworkV1


def test3():
    train_data, test_data = data_source.load_data()
    x = train_data[:, :-1]
    y = train_data[:, -1:]
    # 创建网络
    net = Network(13)
    num_iterations = 1000
    # 启动训练
    losses = net.train(x, y, iterations=num_iterations, eta=0.01)

    # 画出损失函数的变化趋势
    plot_x = np.arange(num_iterations)
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()


def test1():
    training_data, test_data = data_source.load_data()
    x = training_data[:, :-1]
    y = training_data[:, -1:]
    w = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, -0.1, -0.2, -0.3, -0.4, 0.0]
    w = np.array(w).reshape([13, 1])
    net = Network(13)
    x1 = x[0]
    y1 = y[0]
    z = net.forward(x1)
    print(z)
    net = Network(13)
    # 此处可以一次性计算多个样本的预测值和损失函数
    x1 = x[0:3]
    y1 = y[0:3]
    z = net.forward(x1)
    print('predict: ', z)
    loss = net.loss(z, y1)
    print('loss:', loss)
    x1 = x[0]
    y1 = y[0]
    z1 = net.forward(x1)
    print('x1 {}, shape {}'.format(x1, x1.shape))
    print('y1 {}, shape {}'.format(y1, y1.shape))
    print('z1 {}, shape {}'.format(z1, z1.shape))

    net = Network(13)
    # 设置[w5, w9] = [-100., -100.]
    net.w[5] = -100.0
    net.w[9] = -100.0

    z = net.forward(x)
    loss = net.loss(z, y)
    gradient_w, gradient_b = net.gradient(x, y)
    gradient_w5 = gradient_w[5][0]
    gradient_w9 = gradient_w[9][0]
    print('point {}, loss {}'.format([net.w[5][0], net.w[9][0]], loss))
    print('gradient {}'.format([gradient_w5, gradient_w9]))


def test2():
    # 在[w5, w9]平面上，沿着梯度的反方向移动到下一个点P1
    # 定义移动步长 eta
    train_data, test_data = data_source.load_data()
    x = train_data[:, :-1]
    y = train_data[:, -1:]
    net = Network(13)
    gradient_w, gradient_b = net.gradient(x, y)
    gradient_w5 = gradient_w[5][0]
    gradient_w9 = gradient_w[9][0]
    eta = 0.1
    # 更新参数w5和w9
    net.w[5] = net.w[5] - eta * gradient_w5
    net.w[9] = net.w[9] - eta * gradient_w9
    # 重新计算z和loss
    z = net.forward(x)
    loss = net.loss(z, y)
    gradient_w, gradient_b = net.gradient(x, y)
    gradient_w5 = gradient_w[5][0]
    gradient_w9 = gradient_w[9][0]
    print('point {}, loss {}'.format([net.w[5][0], net.w[9][0]], loss))
    print('gradient {}'.format([gradient_w5, gradient_w9]))


def test4():
    # 获取数据
    train_data, test_data = data_source.load_data()

    # 创建网络
    net = NetworkV1(13)
    # 启动训练
    losses = net.train(train_data, num_epochs=50, batch_size=100, eta=0.1)

    # 画出损失函数的变化趋势
    plot_x = np.arange(len(losses))
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()


if __name__ == '__main__':
    data_source = DataSource()
    test4()
