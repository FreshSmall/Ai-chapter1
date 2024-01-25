# 加载飞桨和相关类库
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt


def data_source():
    train_dataset = paddle.vision.datasets.MNIST(mode='train')
    return train_dataset


def draw_img():
    train_dataset = data_source()
    train_data_0 = np.array(train_dataset[0][0])
    train_label_0 = np.array(train_dataset[0][1])
    plt.figure("Image")  # 图像窗口名称
    plt.figure(figsize=(2, 2))
    plt.imshow(train_data_0, cmap=plt.cm.binary)
    plt.axis('on')  # 关掉坐标轴为 off
    plt.title('image')  # 图像题目
    plt.show()

    print("图像数据形状和对应数据为:", train_data_0.shape)
    print("图像标签形状和对应数据为:", train_label_0.shape, train_label_0)
    print("\n打印第一个batch的第一个图像，对应标签数字为{}".format(train_label_0))

def train(model):
    # 启动训练模式
    model.train()
    # 加载训练集 batch_size 设为 16
    train_loader = paddle.io.DataLoader(paddle.vision.datasets.MNIST(mode='train'),
                                        batch_size=16,
                                        shuffle=True)
    # 定义优化器，使用随机梯度下降SGD优化器，学习率设置为0.001
    opt = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())


# 图像归一化函数，将数据范围为[0, 255]的图像归一化到[0, 1]
def norm_img(img):
    # 验证传入数据格式是否正确，img的shape为[batch_size, 28, 28]
    assert len(img.shape) == 3
    batch_size, img_h, img_w = img.shape[0], img.shape[1], img.shape[2]
    # 归一化图像数据
    img = img / 255
    # 将图像形式reshape为[batch_size, 784]
    img = paddle.reshape(img, [batch_size, img_h * img_w])

    print(batch_size)
    print(img_h)
    print(img_w)

    return img




if __name__ == '__main__':
    draw_img()